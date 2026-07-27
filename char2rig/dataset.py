"""Генератор обучающего набора: персонажи с точной разметкой частей.

Зачем свой набор. Замеры на фазе 0 показали, что SAM2 не даёт риггуемых
частей: он размечает по смыслу («это лапа»), а ригу нужен разрез по суставу
(«это вращается вокруг голеностопа»). Публичных наборов с такой разметкой
нет — ни в HumanArt, ни в AP-10K, ни в разметках частей тела на фотографиях.
Зато мы умеем рисовать персонажа сами и знаем про него всё: какой пиксель
какой кости принадлежит и где каждый сустав.

Что рандомизируется:

- **пропорции** — длина конечностей и хвоста, размер головы, ширина плеч,
  толщина костей: сеть не должна выучить один силуэт;
- **поза** — суставы разводятся в пределах правдоподобного, включая позы,
  далёкие от канонической A-позы (ради них DWPose и нужен);
- **окрас** — цвет шерсти и полос, частота и сила полос вплоть до гладкой
  шерсти, светлые «носки», уши, глаза, зернистость;
- **кадр** — размер холста и положение персонажа в нём.

На выходе на каждый образец: RGBA-картинка, карта частей (0 — фон, i+1 —
индекс кости) и JSON с суставами, костями и всеми параметрами розыгрыша.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from . import template as templates
from .demo import Look, draw, fit_to_canvas, place_template
from .template import Bone, Template, posed_joints

# кости, которые двигаем в позе, и разумный размах в градусах
POSE_RANGE: dict[str, float] = {
    "head": 14.0,
    "arm_upper_l": 45.0,
    "arm_fore_l": 35.0,
    "hand_l": 20.0,
    "arm_upper_r": 45.0,
    "arm_fore_r": 35.0,
    "hand_r": 20.0,
    "thigh_l": 28.0,
    "shin_l": 30.0,
    "foot_l": 15.0,
    "thigh_r": 28.0,
    "shin_r": 30.0,
    "foot_r": 15.0,
    "tail_1": 40.0,
    "tail_2": 40.0,
    "tail_3": 40.0,
}

# цепочки, длину которых тянем целиком: сустав едет вместе с потомками
CHAINS: dict[str, tuple[str, ...]] = {
    "arm_l": ("shoulder_l", "elbow_l", "wrist_l", "hand_l"),
    "arm_r": ("shoulder_r", "elbow_r", "wrist_r", "hand_r"),
    "leg_l": ("hip_l", "knee_l", "hock_l", "toe_l"),
    "leg_r": ("hip_r", "knee_r", "hock_r", "toe_r"),
    "tail": ("tail_a", "tail_b", "tail_c", "tail_d"),
    "head": ("neck", "head_top"),
}


def _stretch(
    joints: dict[str, tuple[float, float]], chain: tuple[str, ...], factor: float
) -> dict[str, tuple[float, float]]:
    """Растянуть цепочку от её первого сустава, не трогая направление."""
    out = dict(joints)
    for parent, child in zip(chain, chain[1:]):
        if parent not in out or child not in out:
            continue
        px, py = out[parent]
        cx, cy = out[child]
        out[child] = (px + (cx - px) * factor, py + (cy - py) * factor)
    return out


def random_template(rng: np.random.Generator, base: Template) -> Template:
    """Шаблон со случайными пропорциями."""
    joints = dict(base.joints)

    limbs = float(rng.uniform(0.88, 1.16))
    legs = limbs * float(rng.uniform(0.94, 1.08))
    joints = _stretch(joints, CHAINS["arm_l"], limbs)
    joints = _stretch(joints, CHAINS["arm_r"], limbs)
    joints = _stretch(joints, CHAINS["leg_l"], legs)
    joints = _stretch(joints, CHAINS["leg_r"], legs)
    joints = _stretch(joints, CHAINS["tail"], float(rng.uniform(0.75, 1.3)))
    head = float(rng.uniform(0.85, 1.2))
    joints = _stretch(joints, CHAINS["head"], head)

    shoulders = float(rng.uniform(0.88, 1.18))
    for name in ("shoulder_l", "shoulder_r"):
        if name in joints:
            x, y = joints[name]
            joints[name] = (0.5 + (x - 0.5) * shoulders, y)

    girth = float(rng.uniform(0.82, 1.22))
    head_girth = girth * head * float(rng.uniform(0.92, 1.1))
    bones = tuple(
        replace(
            bone,
            ra=bone.ra * (head_girth if bone.name == "head" else girth),
            rb=bone.rb * (head_girth if bone.name == "head" else girth),
        )
        for bone in base.bones
    )
    return replace(base, joints=joints, bones=bones)


def random_look(rng: np.random.Generator) -> Look:
    """Случайный окрас. Полосы иногда выключаются целиком — бывают и гладкие."""

    def hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
        i = int(h * 6.0)
        f = h * 6.0 - i
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        r, g, b = [
            (v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)
        ][i % 6]
        return (int(r * 255), int(g * 255), int(b * 255))

    hue = float(rng.uniform(0.02, 0.14))  # рыжий → песочный → серо-бурый
    if rng.random() < 0.2:
        hue = float(rng.uniform(0.5, 0.75))  # изредка «неземной» окрас
    saturation = float(rng.uniform(0.25, 0.75))
    value = float(rng.uniform(0.6, 0.95))

    fur = hsv(hue, saturation, value)
    stripe = hsv(
        (hue + float(rng.uniform(-0.04, 0.04))) % 1.0,
        min(1.0, saturation * float(rng.uniform(1.0, 1.5))),
        value * float(rng.uniform(0.45, 0.8)),
    )
    light = hsv(hue, saturation * 0.25, min(1.0, value * 1.15))
    return Look(
        fur=fur,
        stripe=stripe,
        light=light,
        ear=hsv(float(rng.uniform(0.92, 1.0)) % 1.0, 0.35, 0.9),
        eye=hsv(float(rng.uniform(0.08, 0.45)), 0.6, 0.75),
        nose=hsv(float(rng.uniform(0.93, 1.0)) % 1.0, 0.45, 0.85),
        dark=hsv(hue, 0.4, float(rng.uniform(0.08, 0.22))),
        stripe_density=float(rng.uniform(0.035, 0.11)),
        stripe_strength=0.0 if rng.random() < 0.18 else float(rng.uniform(0.5, 1.2)),
        sock=0.0 if rng.random() < 0.3 else float(rng.uniform(0.3, 0.7)),
        belly=float(rng.uniform(0.0, 1.0)),
        rim=float(rng.uniform(0.1, 0.55)),
        grain=float(rng.uniform(0.0, 0.12)),
        seed=int(rng.integers(0, 2**31 - 1)),
        whiskers=bool(rng.random() < 0.8),
        ears=bool(rng.random() < 0.9),
        fluff=float(rng.uniform(0.008, 0.032)),
        fluff_scale=float(rng.uniform(0.5, 1.8)),
        toes=int(rng.integers(0, 4)),
        ruff=float(rng.uniform(0.0, 1.4)),
    )


def random_pose(rng: np.random.Generator, template: Template) -> dict[str, float]:
    """Случайные углы костей в радианах."""
    calm = float(rng.uniform(0.25, 1.0))  # часть образцов — почти A-поза
    return {
        bone.name: math.radians(
            float(rng.normal(0.0, POSE_RANGE[bone.name] * calm / 2))
        )
        for bone in template.bones
        if bone.name in POSE_RANGE
    }


def _one_piece(rgba: np.ndarray, tolerance: float = 0.97) -> bool:
    """Персонаж — одно целое, а не тело плюс отвалившийся кусок хвоста.

    Так бывает честно: средний сегмент хвоста целиком уходит за ногу, и в
    кадре остаётся плавающий кончик. Для конвейера это правда, для обучающего
    набора — мусор.
    """
    import cv2

    mask = (rgba[..., 3] > 128).astype(np.uint8)
    total = int(mask.sum())
    if total == 0:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return False
    biggest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return biggest >= total * tolerance


def sample(
    rng: np.random.Generator,
    base: Template,
    size: tuple[int, int] | None = None,
    attempts: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Один образец: (RGBA, карта частей, метаданные)."""
    width, height = size or (
        int(rng.choice([384, 448, 512])),
        int(rng.choice([576, 640, 768])),
    )
    for attempt in range(attempts):
        template = random_template(rng, base)
        joints, unit = place_template(template, (width, height), margin=0.06)
        deltas = random_pose(rng, template)
        moved, corr = posed_joints(template, joints, deltas)
        moved, unit = fit_to_canvas(template, moved, unit, (width, height))

        # сдвиг в кадре — но уже внутри запаса, который оставил fit_to_canvas
        shift = (rng.normal(0, 0.015 * width), rng.normal(0, 0.015 * height))
        moved = {k: (x + shift[0], y + shift[1]) for k, (x, y) in moved.items()}

        look = random_look(rng)
        rgba, labels = draw(template, (width, height), moved, corr, look, unit)
        if _one_piece(rgba) or attempt == attempts - 1:
            break

    # часть может оказаться целиком за другой — в разметке её тогда нет,
    # и обучению полезно знать об этом заранее, а не выяснять по пустой маске
    counts = np.bincount(labels.ravel(), minlength=len(template.bones) + 1)

    meta = {
        "size": [width, height],
        "template": base.name,
        "unit": round(unit, 2),
        "one_piece": bool(_one_piece(rgba)),
        "bones": [
            {
                "name": bone.name,
                "label": i + 1,
                "parent": bone.parent,
                "joints": [bone.a, bone.b],
                "z": bone.z,
                "radius_a": round(bone.ra * unit, 2),
                "radius_b": round(bone.rb * unit, 2),
                "angle": round(float(deltas.get(bone.name, 0.0)), 4),
                "visible_px": int(counts[i + 1]),
            }
            for i, bone in enumerate(template.bones)
        ],
        "joints": {k: [round(v[0], 2), round(v[1], 2)] for k, v in moved.items()},
        "look": {
            "fur": list(look.fur),
            "stripe": list(look.stripe),
            "stripe_strength": round(look.stripe_strength, 3),
            "stripe_density": round(look.stripe_density, 4),
            "seed": look.seed,
        },
    }
    return rgba, labels, meta


def contact_sheet(
    samples: list[np.ndarray], columns: int = 8, cell: int = 160
) -> np.ndarray:
    """Лист с превью — чтобы разнообразие набора было видно глазами."""
    import cv2

    tiles = []
    for rgba in samples:
        h, w = rgba.shape[:2]
        scale = cell / max(h, w)
        small = cv2.resize(
            rgba, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.zeros((cell, cell, 4), dtype=np.uint8)
        y0 = (cell - small.shape[0]) // 2
        x0 = (cell - small.shape[1]) // 2
        tile[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
        tiles.append(tile)

    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros((cell, cell, 4), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows) if rows else np.zeros((cell, cell, 4), np.uint8)


def generate(
    out: Path,
    count: int,
    seed: int = 1,
    template_name: str = templates.DEFAULT_TEMPLATE,
    size: tuple[int, int] | None = None,
    preview: int = 24,
    on_progress=None,
) -> dict:
    """Сгенерировать набор в папку `out`. Вернуть сводку."""
    base = templates.get(template_name)
    rng = np.random.default_rng(seed)
    images, parts, meta_dir = out / "images", out / "parts", out / "meta"
    for path in (images, parts, meta_dir):
        path.mkdir(parents=True, exist_ok=True)

    previews: list[np.ndarray] = []
    index_lines: list[str] = []
    for i in range(count):
        rgba, labels, meta = sample(rng, base)
        name = f"{i:06d}"
        Image.fromarray(rgba, mode="RGBA").save(images / f"{name}.png")
        Image.fromarray(labels, mode="L").save(parts / f"{name}.png")
        meta["id"] = name
        (meta_dir / f"{name}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        index_lines.append(
            json.dumps(
                {
                    "id": name,
                    "image": f"images/{name}.png",
                    "parts": f"parts/{name}.png",
                    "meta": f"meta/{name}.json",
                    "size": meta["size"],
                    "hidden": [
                        b["name"] for b in meta["bones"] if b["visible_px"] == 0
                    ],
                },
                ensure_ascii=False,
            )
        )
        if len(previews) < preview:
            previews.append(rgba)
        if on_progress and (i + 1) % 25 == 0:
            on_progress(i + 1, count)

    (out / "index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    if previews:
        Image.fromarray(contact_sheet(previews), mode="RGBA").save(
            out / "preview.png"
        )
    summary = {
        "count": count,
        "seed": seed,
        "template": template_name,
        "labels": len(base.bones) + 1,
    }
    (out / "dataset.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
