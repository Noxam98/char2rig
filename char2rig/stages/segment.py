"""Этап 3: разбиение силуэта на части тела.

Фоллбек без моделей: каждый пиксель силуэта достаётся кости, к капсуле
которой он ближе (нормированное расстояние `d - r`). Так конфликты решаются
сами собой, а непокрытых пикселей не остаётся вовсе — что важно: дырка в
покрытии позже превращается в дырку в анимации.

SAM2 (ultralytics) подключается сверху: маски от него уточняют границы,
а спорные пиксели всё равно разводятся по капсулам.
"""

from __future__ import annotations

import numpy as np

from ..geometry import capsule_field, pixel_grid
from ..template import Template


def capsule_owner(
    shape: tuple[int, int],
    template: Template,
    joints: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, list[str]]:
    """Карта «какой кости принадлежит пиксель» + порядок имён костей."""
    grid = pixel_grid(shape)
    unit = pixels_per_unit(joints, template)
    names = [b.name for b in template.bones]
    fields = np.stack(
        [
            capsule_field(
                shape,
                joints[b.a],
                joints[b.b],
                b.ra * unit,
                b.rb * unit,
                grid=grid,
            )
            for b in template.bones
        ]
    )
    return np.argmin(fields, axis=0).astype(np.int16), names


def pixels_per_unit(
    joints: dict[str, tuple[float, float]], template: Template
) -> float:
    """Пикселей на единицу нормировки шаблона (рост персонажа).

    Радиусы в шаблоне заданы в долях роста, а суставы уже в пикселях —
    масштаб восстанавливаем по длинам костей. Берём медиану, а не максимум:
    шаблон мог быть растянут по ширине, и горизонтальные кости тогда врут.
    """
    ratios = []
    for bone in template.bones:
        ax, ay = joints[bone.a]
        bx, by = joints[bone.b]
        nx, ny = template.joints[bone.a]
        mx, my = template.joints[bone.b]
        norm = float(np.hypot(mx - nx, my - ny))
        if norm < 1e-3:
            continue
        ratios.append(float(np.hypot(bx - ax, by - ay)) / norm)
    return float(np.median(ratios)) if ratios else 1.0


def _try_sam2(
    rgb: np.ndarray,
    joints: dict[str, tuple[float, float]],
    template: Template,
) -> dict[str, np.ndarray] | None:
    """Маски частей от SAM2 по подсказкам-точкам из костей; None если нет модели."""
    try:
        from ultralytics import SAM
    except ImportError:
        return None
    try:
        model = SAM("sam2_b.pt")
        masks: dict[str, np.ndarray] = {}
        for bone in template.bones:
            ax, ay = joints[bone.a]
            bx, by = joints[bone.b]
            point = [[(ax + bx) / 2, (ay + by) / 2]]
            result = model(rgb, points=point, labels=[1], verbose=False)[0]
            if result.masks is None or len(result.masks.data) == 0:
                return None
            masks[bone.name] = result.masks.data[0].cpu().numpy() > 0.5
        return masks
    except Exception:
        return None


def run(
    template: Template,
    rgba: np.ndarray,
    alpha: np.ndarray,
    joints: dict[str, tuple[float, float]],
    use_ml: bool = True,
) -> tuple[dict[str, np.ndarray], str, bool, dict]:
    """Вернуть (маски частей, метод, фоллбек ли, статистика)."""
    shape = alpha.shape
    silhouette = alpha > 16
    owner, names = capsule_owner(shape, template, joints)

    sam_masks = _try_sam2(rgba[..., :3], joints, template) if use_ml else None
    if sam_masks is not None:
        # SAM2 даёт границы, капсулы — арбитраж: где маска ровно одна, верим
        # ей; где их несколько или ни одной — остаётся ближайшая капсула.
        votes = np.stack([sam_masks[n] for n in names])
        count = votes.sum(axis=0)
        owner = np.where(count == 1, np.argmax(votes, axis=0), owner).astype(np.int16)
        method, fallback = "sam2+capsules", False
    else:
        method, fallback = "capsules", True

    masks = {
        name: (silhouette & (owner == i)).astype(np.uint8) * 255
        for i, name in enumerate(names)
    }
    stats = {
        "parts": len(masks),
        "empty_parts": sorted(n for n, m in masks.items() if m.max() == 0),
        "silhouette_px": int(silhouette.sum()),
    }
    return masks, method, fallback, stats
