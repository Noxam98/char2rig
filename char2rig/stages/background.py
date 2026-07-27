"""Этап 1: силуэт персонажа.

Порядок попыток:
1. в исходнике уже есть осмысленная альфа — берём её (ничего лучше нет);
2. rembg (u2net/BiRefNet), если установлен;
3. заливка от рамки по цвету фона — грубый фоллбек для однотонного фона.
"""

from __future__ import annotations

import numpy as np

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
        return None
    try:
        out = np.array(rembg_remove(rgba[..., :3]))
    except Exception:  # модель не скачалась, нет сети и т.п.
        return None
    if out.ndim == 3 and out.shape[2] == 4:
        return out[..., 3]
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


def run(rgba: np.ndarray, use_ml: bool = True) -> tuple[np.ndarray, str, bool]:
    """Вернуть (альфа uint8, название метода, был ли это фоллбек).

    Чистка бинарная, но сглаженный край исходной альфы сохраняется: по нему
    потом режутся слои, и лесенка на контуре вылезла бы в каждом кадре.
    """
    if _has_alpha(rgba):
        soft = rgba[..., 3]
        return np.minimum(_clean(soft), soft), "source_alpha", True

    if use_ml:
        soft = _try_rembg(rgba)
        if soft is not None:
            return np.minimum(_clean(soft), soft), "rembg", False

    return _clean(_flood_from_border(rgba)), "border_flood", True
