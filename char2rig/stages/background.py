"""Этап 1: силуэт персонажа.

Порядок попыток:
1. в исходнике уже есть осмысленная альфа — берём её (ничего лучше нет);
2. rembg (u2net/BiRefNet), если установлен;
3. заливка от рамки по цвету фона — грубый фоллбек для однотонного фона.
"""

from __future__ import annotations

import numpy as np

from . import note_fallback

try:  # cv2 — часть ядра, но пусть падает внятно
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("нужен opencv: pip install -r requirements.txt") from exc


def _clean(mask: np.ndarray) -> np.ndarray:
    """Оставить крупные компоненты и залить дырки внутри силуэта."""
    binary = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = int(areas.max())
        # мелкий мусор выкидываем, но серьги/уши/кончик хвоста оставляем
        keep = {i + 1 for i, a in enumerate(areas) if a >= max(64, biggest * 0.02)}
        binary = np.isin(labels, list(keep)).astype(np.uint8)

    filled = binary.copy()
    h, w = filled.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 1)
    holes = (filled == 0).astype(np.uint8)
    return ((binary | holes) * 255).astype(np.uint8)


def _has_alpha(rgba: np.ndarray) -> bool:
    alpha = rgba[..., 3]
    transparent = float((alpha < 16).mean())
    return 0.02 < transparent < 0.98


def _try_rembg(rgba: np.ndarray) -> np.ndarray | None:
    """Альфа от rembg; None, если библиотека не установлена или упала."""
    try:
        from rembg import remove as rembg_remove
    except ImportError:
        note_fallback("фон", "rembg не установлен")
        return None
    try:
        out = np.array(rembg_remove(np.ascontiguousarray(rgba[..., :3])))
    except Exception as exc:  # модель не скачалась, нет сети и т.п.
        note_fallback("фон", f"rembg упал: {type(exc).__name__}: {exc}")
        return None
    if out.ndim == 3 and out.shape[2] == 4:
        return out[..., 3]
    note_fallback("фон", "rembg вернул кадр без альфы")
    return None


def _flood_from_border(rgba: np.ndarray) -> np.ndarray:
    """Фон = то, что связно с рамкой и близко к ней по цвету."""
    rgb = rgba[..., :3].astype(np.uint8)
    h, w = rgb.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    work = rgb.copy()
    tol = (18, 18, 18)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        cv2.floodFill(work, mask, seed, (0, 0, 0), tol, tol, 4 | (255 << 8))
    background = mask[1:-1, 1:-1] > 0
    return ((~background) * 255).astype(np.uint8)


def extract_contour(alpha: np.ndarray, simplify: float = 0.0015) -> list[list[float]]:
    """Внешний контур силуэта как многоугольник — то, что человек будет править.

    Упрощение подобрано так, чтобы узлов было сотня-полторы: по узлу на
    каждый изгиб уха или пальца, но не по узлу на пиксель.
    """
    binary = (alpha > 16).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    biggest = max(contours, key=cv2.contourArea)
    epsilon = simplify * cv2.arcLength(biggest, True)
    points = cv2.approxPolyDP(biggest, epsilon, True).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in points]


def rasterize_contour(
    points: list[list[float]], shape: tuple[int, int], feather: float = 1.0
) -> np.ndarray:
    """Многоугольник → маска со сглаженным краем."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        polygon = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
    return mask


def _apply_strokes(alpha: np.ndarray, strokes: np.ndarray | None) -> np.ndarray:
    """Наложить ручные мазки: 255 — это тело, 128 — это фон.

    Правка живёт отдельным слоем и переприменяется поверх нового результата:
    перегенерация силуэта её не стирает (принцип №1 в DESIGN.md).
    """
    if strokes is None:
        return alpha
    result = alpha.copy()
    result[strokes > 200] = 255
    result[(strokes > 64) & (strokes <= 200)] = 0
    return result


def run(
    rgba: np.ndarray,
    use_ml: bool = True,
    strokes: np.ndarray | None = None,
    contour: list[list[float]] | None = None,
) -> tuple[np.ndarray, str, bool]:
    """Вернуть (альфа uint8, название метода, был ли это фоллбек).

    Чистка бинарная, но сглаженный край исходной альфы сохраняется: по нему
    потом режутся слои, и лесенка на контуре вылезла бы в каждом кадре.

    Поправленный руками контур заменяет автоматику целиком: человек, который
    подвинул узлы, знает про этот арт больше, чем сеть.
    """
    if contour:
        drawn = rasterize_contour(contour, rgba.shape[:2])
        return _apply_strokes(drawn, strokes), "hand_contour", True

    if _has_alpha(rgba):
        soft = rgba[..., 3]
        return _apply_strokes(np.minimum(_clean(soft), soft), strokes), (
            "source_alpha"
        ), True

    if use_ml:
        soft = _try_rembg(rgba)
        if soft is not None:
            return _apply_strokes(np.minimum(_clean(soft), soft), strokes), (
                "rembg"
            ), False

    flooded = _clean(_flood_from_border(rgba))
    return _apply_strokes(flooded, strokes), "border_flood", True
