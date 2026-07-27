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

Пиксели заезда берутся прямо из исходника (там настоящий мех соседа), а не
достраиваются — на стыке это честнее любого inpaint. Достраивается только
узкая кайма за пределами силуэта, чтобы при отведении конечности из-под неё
не выглядывал фон. Полноценный inpaint скрытых областей (LaMa) — шаг
«ML-обвязка» в PLAN.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry import pixel_grid
from ..template import Bone, Template

RING_PX = 2.0  # ширина каймы за силуэтом
SEAM_PX = 2.0  # минимальный заезд вдали от суставов — против волосяных швов


def _overlap_px(bone, unit: float) -> float:
    """Сколько пикселей плотного заезда дать кости."""
    thin = min(bone.ra, bone.rb) * unit
    return float(np.clip(0.7 * thin * bone.overlap, 4.0, 0.05 * unit))


def _connection_points(
    template: Template,
    bone: Bone,
    joints: dict[str, tuple[float, float]],
    unit: float,
) -> list[tuple[tuple[float, float], float]]:
    """Суставы, которыми кость соединена с соседями: (точка, радиус сустава)."""
    points = [(joints[bone.a], bone.ra * unit)]
    points.extend(
        (joints[child.a], child.ra * unit)
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


def run(
    template: Template,
    rgba: np.ndarray,
    alpha: np.ndarray,
    joints: dict[str, tuple[float, float]],
    masks: dict[str, np.ndarray],
    unit: float,
) -> tuple[dict[str, dict], str, bool, dict]:
    """Вернуть (слои, метод, фоллбек ли, статистика).

    Слой: {"image": RGBA-кроп, "offset": (x0, y0)}.
    """
    shape = alpha.shape
    grid = pixel_grid(shape)
    silhouette = alpha > 16
    soft = alpha.astype(np.float32) / 255.0

    dist_out = cv2.distanceTransform(
        (~silhouette).astype(np.uint8), cv2.DIST_L2, 3
    )
    silhouette_ext = np.maximum(soft, np.clip(1.0 - dist_out / RING_PX, 0.0, 1.0))

    rgb = np.ascontiguousarray(rgba[..., :3])
    coverage = np.zeros(shape, dtype=bool)
    layers: dict[str, dict] = {}
    filled_px = 0

    for bone in template.bones:
        own = masks[bone.name] > 127
        if not own.any():
            continue
        overlap = _overlap_map(
            shape,
            _connection_points(template, bone, joints, unit),
            _overlap_px(bone, unit),
            grid,
        )
        feather = max(2.0, 0.005 * unit)
        dist_own = cv2.distanceTransform((~own).astype(np.uint8), cv2.DIST_L2, 3)
        falloff = np.clip((overlap + feather - dist_own) / feather, 0.0, 1.0)
        part_alpha = falloff * silhouette_ext
        region = part_alpha > 1e-3
        if not region.any():
            continue

        unknown = region & ~silhouette
        filled_px += int(unknown.sum())
        part_rgb = _fill_outside(rgb, unknown) if unknown.any() else rgb

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
        "inpainted_px": filled_px,
    }
    return layers, "overlap_cut+nearest_fill", True, stats
