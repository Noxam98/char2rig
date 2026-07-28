"""Этап 4: нарезка на слои с нахлёстом и растушёванной альфой.

Ключевая идея против швов на полосатом мехе: часть не обрезается ровно по
своей маске, а «заезжает» на соседей на `overlap` пикселей. Заезд
непрозрачный, и только последние `feather` пикселей гаснут — тогда соседний
слой ложится сверху и стык превращается в градиент, а когда конечность
отходит, из-под неё видно плотный мех, а не полупрозрачный призрак.

Ширина заезда взята из геометрии: при повороте на θ вокруг сустава радиуса R
открывается полоса примерно 2·R·sin(θ/2); для θ=40° из DESIGN это ~0.7·R.

Заезд даётся не во все стороны, а шапкой вокруг суставов, которыми часть
соединена с соседями. Иначе торс, отрастив 30 пикселей по всему контуру,
при взмахе руки выставляет наружу прямоугольный обрубок; шапка же уходит
ровно туда, где сустав и открывается.

Пиксели заезда на прямого соседа (родителя или ребёнка) берутся из
исходника: на стыке настоящий мех соседа честнее любого inpaint. А вот
область, закрытую чужой частью спереди — рука, легшая поперёк торса, —
восстановить нечем, и её достраивает LaMa; без модели там остаются
исходные пиксели чужой части. Узкая кайма за силуэтом всегда достраивается
ближайшим пикселем, чтобы при отведении конечности не выглядывал фон.

Границу отданной модели области можно двигать руками — мазками из редактора
(`layers.overrides/<часть>.png`): 255 — «дорисуй здесь заново», 128 —
«оставь как есть».
"""

from __future__ import annotations

import cv2
import numpy as np

from . import note_fallback
from ..geometry import pixel_grid
from ..template import Bone, Template

RING_PX = 2.0  # ширина каймы за силуэтом
SEAM_PX = 2.0  # минимальный заезд вдали от суставов — против волосяных швов


def _overlap_px(bone, unit: float, own: np.ndarray) -> float:
    """Сколько пикселей плотного заезда дать кости.

    Толщина берётся у самой части, а не у капсулы скелета: маска получает и
    те пиксели, до которых капсула не дотянулась (шерсть, пушистый хвост), и
    именно они при повороте уезжают дальше всего. На реальном коте хвост от
    этого разваливался на куски — заезда, посчитанного по тонкой капсуле,
    не хватало на толстую часть.
    """
    distance = cv2.distanceTransform(own.astype(np.uint8), cv2.DIST_L2, 3)
    thick = float(np.percentile(distance[own], 90)) if own.any() else 0.0
    return float(np.clip(0.7 * thick * bone.overlap, 4.0, 0.06 * unit))


def _connection_points(
    template: Template,
    bone: Bone,
    joints: dict[str, tuple[float, float]],
    unit: float,
    radii: dict[str, tuple[float, float]] | None = None,
) -> list[tuple[tuple[float, float], float]]:
    """Суставы, которыми кость соединена с соседями: (точка, радиус сустава)."""

    def radius(item: Bone) -> float:
        return radii[item.name][0] if radii else item.ra * unit

    points = [(joints[bone.a], radius(bone))]
    points.extend(
        (joints[child.a], radius(child))
        for child in template.bones
        if child.parent == bone.name
    )
    return points


def _overlap_map(
    shape: tuple[int, int],
    points: list[tuple[tuple[float, float], float]],
    overlap: float,
    grid: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Карта «сколько заезда разрешено в этой точке»: шапки вокруг суставов."""
    xs, ys = grid
    result = np.full(shape, SEAM_PX, dtype=np.float32)
    for (jx, jy), radius in points:
        reach = overlap + radius
        near = np.clip((reach - np.hypot(xs - jx, ys - jy)) / (0.4 * reach), 0.0, 1.0)
        result = np.maximum(result, SEAM_PX + (overlap - SEAM_PX) * near)
    return result


def _occluders(template: Template, bone: Bone) -> list[str]:
    """Части, которые лежат поверх этой и не являются ей роднёй.

    Родитель и дети не считаются: их нахлёст на стыке — это тот самый
    настоящий мех, ради которого нарезка и делается с заездом.
    """
    kin = {bone.parent, bone.name} | {
        b.name for b in template.bones if b.parent == bone.name
    }
    return [b.name for b in template.bones if b.z > bone.z and b.name not in kin]


_LAMA = None


def _try_lama(rgb: np.ndarray, hidden: np.ndarray) -> np.ndarray | None:
    """Достроить скрытую область через LaMa; None, если модели нет."""
    global _LAMA
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        note_fallback("слои", "simple-lama-inpainting не установлен")
        return None
    try:
        if _LAMA is None:
            _LAMA = SimpleLama()
        out = _LAMA(rgb, (hidden * 255).astype(np.uint8))
    except Exception as exc:
        note_fallback("слои", f"LaMa упала: {type(exc).__name__}: {exc}")
        return None
    # LaMa дополняет вход до кратности восьми и отдаёт кадр вместе с добавкой
    height, width = rgb.shape[:2]
    return np.array(out)[:height, :width, :3]


def _fill_outside(rgb: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """Кайма за силуэтом — цветом ближайшего известного пикселя.

    Ближайший пиксель, а не диффузия cv2.inpaint: кайма узкая, а диффузия
    размазывает тонкие тёмные детали (усы) в жирные полосы. Настоящий
    inpaint скрытых областей — LaMa на шаге «ML-обвязка».
    """
    if not unknown.any():
        return rgb
    _, labels = cv2.distanceTransformWithLabels(
        unknown.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    known = ~unknown
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    flat_index = np.flatnonzero(known)
    lut[labels[known]] = flat_index
    filled = rgb.reshape(-1, 3)[lut[labels]].reshape(rgb.shape)
    return np.where(unknown[..., None], filled, rgb).astype(np.uint8)


def extended_silhouette(alpha: np.ndarray) -> np.ndarray:
    """Альфа силуэта плюс узкая кайма за его краем."""
    silhouette = alpha > 16
    soft = alpha.astype(np.float32) / 255.0
    dist_out = cv2.distanceTransform((~silhouette).astype(np.uint8), cv2.DIST_L2, 3)
    return np.maximum(soft, np.clip(1.0 - dist_out / RING_PX, 0.0, 1.0))


def part_fill(
    template: Template,
    bone: Bone,
    alpha: np.ndarray,
    joints: dict[str, tuple[float, float]],
    masks: dict[str, np.ndarray],
    unit: float,
    radii: dict[str, tuple[float, float]] | None = None,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    silhouette_ext: np.ndarray | None = None,
) -> dict | None:
    """Где лежит часть и какие её пиксели придётся выдумать.

    Возвращает ``{"alpha", "region", "ring", "hidden"}`` либо None, если
    части нет вовсе. Отсюда же берёт данные редактор достройки: считать
    выдуманные пиксели двумя разными кусками кода — верный способ показать
    человеку не то, что попадёт в слой.
    """
    shape = alpha.shape
    own = masks[bone.name] > 127
    if not own.any():
        return None
    if grid is None:
        grid = pixel_grid(shape)
    if silhouette_ext is None:
        silhouette_ext = extended_silhouette(alpha)

    overlap = _overlap_map(
        shape,
        _connection_points(template, bone, joints, unit, radii),
        _overlap_px(bone, unit, own),
        grid,
    )
    feather = max(2.0, 0.005 * unit)
    dist_own = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 3)
    falloff = np.clip((overlap + feather - dist_own) / feather, 0.0, 1.0)
    part_alpha = falloff * silhouette_ext
    region = part_alpha > 1e-3
    if not region.any():
        return None

    # область, закрытую чужой частью спереди, честно достроить нечем —
    # там лежат её пиксели, а не наши; отдаём LaMa, если она есть
    covers = [masks[name] > 127 for name in _occluders(template, bone)]
    hidden = region & np.logical_or.reduce(covers or [np.zeros(shape, dtype=bool)])
    return {
        "alpha": part_alpha,
        "region": region,
        "ring": region & ~(alpha > 16),
        "hidden": hidden,
    }


def _apply_redraw(
    hidden: np.ndarray, region: np.ndarray, strokes: np.ndarray | None
) -> np.ndarray:
    """Ручная правка достройки: 255 — дорисовать заново, 128 — не трогать.

    Смотреть на достроенное человеку проще, чем эвристике: LaMa то размажет
    полосу, то, наоборот, зря сотрёт годные пиксели соседа. Мазок только
    двигает границу области, которую отдают модели, — сами пиксели никто
    руками не рисует (принцип №1 в DESIGN.md).
    """
    if strokes is None:
        return hidden
    again = (strokes > 200) & region
    keep = (strokes > 64) & (strokes <= 200)
    return (hidden | again) & ~keep


def run(
    template: Template,
    rgba: np.ndarray,
    alpha: np.ndarray,
    joints: dict[str, tuple[float, float]],
    masks: dict[str, np.ndarray],
    unit: float,
    radii: dict[str, tuple[float, float]] | None = None,
    use_ml: bool = True,
    redraw: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, dict], str, bool, dict]:
    """Вернуть (слои, метод, фоллбек ли, статистика).

    Слой: {"image": RGBA-кроп, "offset": (x0, y0)}.
    `redraw` — ручные мазки по достройке, по маске на часть.
    """
    shape = alpha.shape
    grid = pixel_grid(shape)
    silhouette = alpha > 16
    silhouette_ext = extended_silhouette(alpha)

    rgb = np.ascontiguousarray(rgba[..., :3])
    coverage = np.zeros(shape, dtype=bool)
    layers: dict[str, dict] = {}
    filled_px = 0
    hidden_px = 0
    redraw_px = 0
    lama_used = False

    for bone in template.bones:
        fill = part_fill(
            template, bone, alpha, joints, masks, unit, radii, grid, silhouette_ext
        )
        if fill is None:
            continue
        own = masks[bone.name] > 127
        part_alpha, region = fill["alpha"], fill["region"]

        unknown = fill["ring"]
        filled_px += int(unknown.sum())
        part_rgb = _fill_outside(rgb, unknown) if unknown.any() else rgb

        strokes = (redraw or {}).get(bone.name)
        hidden = _apply_redraw(fill["hidden"], region, strokes)
        redraw_px += int((hidden != fill["hidden"]).sum())
        hidden_px += int(hidden.sum())
        if use_ml and hidden.any():
            restored = _try_lama(part_rgb, hidden)
            if restored is not None:
                part_rgb = np.where(hidden[..., None], restored, part_rgb)
                lama_used = True

        ys, xs = np.nonzero(region)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        crop[..., :3] = part_rgb[y0:y1, x0:x1]
        crop[..., 3] = (part_alpha[y0:y1, x0:x1] * 255).astype(np.uint8)

        layers[bone.name] = {"image": crop, "offset": (x0, y0)}
        coverage |= own

    uncovered = int((silhouette & ~coverage).sum())
    stats = {
        "layers": len(layers),
        "uncovered_px": uncovered,
        "uncovered_ratio": round(uncovered / max(int(silhouette.sum()), 1), 5),
        "ring_px": filled_px,
        "hidden_px": hidden_px,
        "redraw_px": redraw_px,
    }
    method = "overlap_cut+lama" if lama_used else "overlap_cut+nearest_fill"
    return layers, method, not lama_used, stats
