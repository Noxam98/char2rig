"""Этап 3: разбиение силуэта на части тела.

Фоллбек без моделей: пиксель достаётся верхней по z капсуле из тех, что его
накрыли, а если не накрыла ни одна — ближайшей. Непокрытых пикселей не
остаётся вовсе, что важно: дырка в покрытии позже превращается в дырку в
анимации.

SAM2 (ultralytics) подключается сверху: маски от него уточняют границы,
а спорные пиксели всё равно разводятся по капсулам.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import note_fallback
from ..geometry import capsule_field, pixel_grid
from ..template import Template, pixels_per_unit


def capsule_fields(
    shape: tuple[int, int],
    template: Template,
    joints: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, list[str]]:
    """Поля расстояний до капсул всех костей + порядок имён."""
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
    return fields, names


def capsule_owner(
    fields: np.ndarray, template: Template
) -> np.ndarray:
    """Карта «какой кости принадлежит пиксель» по капсулам скелета.

    Правило — «верхняя по z из накрывших», а не «ближайшая»: арт рисуется
    слоями, и пиксель принадлежит той части, что лежит спереди. Разница не
    косметическая: на сгенерированном наборе с истинной разметкой ближайшая
    капсула даёт IoU 0.67 (бёдра 0.30 — их съедает толстый торс), а верхняя
    накрывшая — 0.91 при попиксельной точности 0.96.

    Пиксели, до которых не дотянулась ни одна капсула (клочья шерсти, уши,
    одежда), достаются ближайшей — там угадывать больше нечем.
    """
    owner = np.argmin(fields, axis=0)
    for index in np.argsort([bone.z for bone in template.bones]):  # снизу вверх
        owner = np.where(fields[index] < 0, index, owner)
    return owner.astype(np.int16)




_SAM = None

# SAM2 по подсказке из одной кости часто отдаёт весь силуэт целиком: на
# замерах по демо-коту так вышло для торса и плеча (98% силуэта), зато
# кисти и стопы он ловит с IoU 0.86–0.95. Поэтому маска принимается, только
# если она правдоподобна как часть — иначе тут решают капсулы.
SAM_MAX_AREA = 2.0  # во сколько раз маска может быть больше своей капсулы
SAM_MIN_IOU = 0.5  # и насколько обязана совпасть с тем, где мы ждём часть

# веса кладём в models/ (в .gitignore): ultralytics по умолчанию сыплет их
# в текущую папку, а 154 МБ в корне репы GitHub не принимает
SAM_WEIGHTS = Path("models/sam2_b.pt")


def _try_sam2(
    rgb: np.ndarray,
    joints: dict[str, tuple[float, float]],
    template: Template,
    capsules: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    """Правдоподобные маски частей от SAM2 (может вернуть часть костей)."""
    global _SAM
    try:
        from ultralytics import SAM
    except ImportError:
        note_fallback("части", "ultralytics не установлен")
        return None
    try:
        if _SAM is None:
            SAM_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            _SAM = SAM(str(SAM_WEIGHTS))
        rgb = np.ascontiguousarray(rgb)
        masks: dict[str, np.ndarray] = {}
        for bone in template.bones:
            ax, ay = joints[bone.a]
            bx, by = joints[bone.b]
            axis = [
                [ax + (bx - ax) * t, ay + (by - ay) * t] for t in (0.3, 0.5, 0.7)
            ]
            result = _SAM(rgb, points=axis, labels=[1, 1, 1], verbose=False)[0]
            if result.masks is None or len(result.masks.data) == 0:
                continue
            mask = result.masks.data[0].cpu().numpy() > 0.5
            capsule = capsules[bone.name]
            area, expected = int(mask.sum()), max(int(capsule.sum()), 1)
            iou = int((mask & capsule).sum()) / max(int((mask | capsule).sum()), 1)
            if area <= SAM_MAX_AREA * expected and iou >= SAM_MIN_IOU:
                masks[bone.name] = mask
        return masks or None
    except Exception as exc:
        note_fallback("части", f"SAM2 упал: {type(exc).__name__}: {exc}")
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
    fields, names = capsule_fields(shape, template, joints)
    owner = capsule_owner(fields, template)
    capsules = {n: (fields[i] < 0) & silhouette for i, n in enumerate(names)}

    sam_masks = (
        _try_sam2(rgba[..., :3], joints, template, capsules) if use_ml else None
    )
    accepted: list[str] = []
    if sam_masks:
        # SAM2 знает, что «это лапа», но не знает, что рогом вращения будет
        # голеностоп: его граница части проходит не по суставу, и перенос
        # пикселей через неё ломает риг — часть уезжает не вокруг того сустава.
        # Поэтому разрезы по суставам остаются за капсулами, а SAM2 решает
        # только на окраинах, куда скелет не дотянулся: клочья шерсти, уши,
        # одежда. Там правило «ближайшая капсула» — всё равно гадание.
        outskirts = fields.min(axis=0) > 0
        accepted = [n for n in names if n in sam_masks]
        index = np.array([names.index(n) for n in accepted], dtype=np.int16)
        votes = np.stack([sam_masks[n] for n in accepted])
        single = (votes.sum(axis=0) == 1) & outskirts
        owner = np.where(single, index[np.argmax(votes, axis=0)], owner).astype(
            np.int16
        )
        method, fallback = "sam2+capsules", False
    else:
        method, fallback = "capsules", True

    masks = {
        name: (silhouette & (owner == i)).astype(np.uint8) * 255
        for i, name in enumerate(names)
    }
    stats = {
        "parts": len(masks),
        "sam_parts": accepted,
        "empty_parts": sorted(n for n, m in masks.items() if m.max() == 0),
        "silhouette_px": int(silhouette.sum()),
    }
    return masks, method, fallback, stats
