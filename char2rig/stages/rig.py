"""Этап 5: сборка рига.

Пивот части — начальный сустав её кости, родитель части — родительская кость.
Всё остальное (позы, анимации) считается из этого: rig.json самодостаточен,
рендеру не нужны ни шаблон, ни исходник.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..character import Character
from ..geometry import angle_of, length_of
from ..template import Template


def build(
    template: Template,
    joints: dict[str, tuple[float, float]],
    layers: dict[str, dict],
    size: tuple[int, int],
    unit: float,
    radii: dict[str, tuple[float, float]] | None = None,
) -> dict:
    bones = []
    for bone in template.ordered_bones():
        if bone.name not in layers:
            continue
        a, b = joints[bone.a], joints[bone.b]
        layer = layers[bone.name]
        bones.append(
            {
                "name": bone.name,
                "parent": bone.parent,
                "joints": [bone.a, bone.b],
                "rest_a": [round(a[0], 2), round(a[1], 2)],
                "rest_b": [round(b[0], 2), round(b[1], 2)],
                "rest_angle": round(angle_of(a, b), 6),
                "length": round(length_of(a, b), 2),
                "radius_a": round(
                    radii[bone.name][0] if radii else bone.ra * unit, 2
                ),
                "radius_b": round(
                    radii[bone.name][1] if radii else bone.rb * unit, 2
                ),
                "z": bone.z,
                "image": f"layers/{bone.name}.png",
                "offset": [int(layer["offset"][0]), int(layer["offset"][1])],
                "pivot": [round(a[0], 2), round(a[1], 2)],
            }
        )
    return {
        "template": template.name,
        "size": [int(size[0]), int(size[1])],
        "unit": round(unit, 2),
        "chains": {k: list(v) for k, v in template.chains.items()},
        "joints": {k: [round(v[0], 2), round(v[1], 2)] for k, v in joints.items()},
        "bones": bones,
    }


def run(
    char: Character,
    template: Template,
    joints: dict[str, tuple[float, float]],
    layers: dict[str, dict],
    size: tuple[int, int],
    unit: float,
    radii: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Записать слои и rig.json на диск, вернуть риг."""
    for name, layer in layers.items():
        char.write_rgba(char.layers_dir / f"{name}.png", layer["image"])
    rig = build(template, joints, layers, size, unit, radii)
    char.write_json(char.rig, rig)
    return rig


def load_images(char: Character, rig: dict) -> dict[str, np.ndarray]:
    """Слои с диска, ключ — имя кости."""
    images: dict[str, np.ndarray] = {}
    for bone in rig["bones"]:
        path = Path(char.root) / bone["image"]
        images[bone["name"]] = np.array(Image.open(path).convert("RGBA"))
    return images
