"""Этап 6: FK-сборка персонажа, стресс-свинг, preview.gif + спрайт-лента.

FK без матриц по цепочке: для каждой кости хранится только «поправка» —
насколько её мировой угол отличается от покоя. Тогда

    corr[кость]  = corr[родитель] + delta[кость]
    pos[кость]   = pos[родитель] + rot(corr[родитель]) · (rest_a - rest_a_род)

а часть рисуется поворотом на corr вокруг своего пивота со сдвигом в pos.
Никаких накопленных ошибок и никаких отдельных «мировых матриц».
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from ..character import Character
from ..geometry import capsule_field, pixel_grid, rot

# Стресс-свинг: амплитуда в градусах и фаза в радианах.
# Стороны в противофазе — так шов в суставе видно на всех фазах качания.
SWING: dict[str, tuple[float, float]] = {
    "head": (8.0, 0.5),
    "arm_upper_l": (35.0, 0.0),
    "arm_fore_l": (20.0, 0.8),
    "hand_l": (12.0, 1.4),
    "arm_upper_r": (35.0, math.pi),
    "arm_fore_r": (20.0, math.pi + 0.8),
    "hand_r": (12.0, math.pi + 1.4),
    "thigh_l": (30.0, math.pi),
    "shin_l": (20.0, math.pi + 0.7),
    "foot_l": (10.0, math.pi + 1.2),
    "thigh_r": (30.0, 0.0),
    "shin_r": (20.0, 0.7),
    "foot_r": (10.0, 1.2),
    "tail_1": (25.0, 0.0),
    "tail_2": (25.0, 0.6),
    "tail_3": (25.0, 1.2),
}

BG_COLOR = (34, 34, 40)
FRAMES = 24
PAD_RATIO = 0.18


def swing_deltas(phase: float) -> dict[str, float]:
    """Углы поворота костей (радианы) для момента фазы."""
    return {
        name: math.radians(amp) * math.sin(phase + shift)
        for name, (amp, shift) in SWING.items()
    }


def solve_fk(
    rig: dict, deltas: dict[str, float]
) -> dict[str, tuple[float, np.ndarray]]:
    """Для каждой кости: (поправка угла, позиция начального сустава)."""
    by_name = {b["name"]: b for b in rig["bones"]}
    out: dict[str, tuple[float, np.ndarray]] = {}
    for bone in rig["bones"]:  # порядок «родитель раньше ребёнка» из rig.py
        rest_a = np.array(bone["rest_a"], dtype=float)
        parent = bone["parent"]
        delta = float(deltas.get(bone["name"], 0.0))
        if parent is None or parent not in out:
            out[bone["name"]] = (delta, rest_a)
            continue
        corr_p, pos_p = out[parent]
        rest_ap = np.array(by_name[parent]["rest_a"], dtype=float)
        pos = pos_p + rot(corr_p) @ (rest_a - rest_ap)
        out[bone["name"]] = (corr_p + delta, pos)
    return out


def render_frame(
    rig: dict,
    images: dict[str, np.ndarray],
    deltas: dict[str, float],
    canvas: tuple[int, int],
    pad: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Собрать кадр. Вернуть (RGB float 0..1, накопленная альфа 0..1)."""
    height, width = canvas
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    acc = np.zeros((height, width), dtype=np.float32)
    pose = solve_fk(rig, deltas)

    for bone in sorted(rig["bones"], key=lambda b: b["z"]):
        image = images.get(bone["name"])
        if image is None:
            continue
        corr, pos = pose[bone["name"]]
        rotation = rot(corr)
        offset = np.array(bone["offset"], dtype=float)
        pivot = np.array(bone["pivot"], dtype=float)
        shift = rotation @ (offset - pivot) + pos + np.array(pad)
        matrix = np.hstack([rotation, shift.reshape(2, 1)]).astype(np.float32)

        warped = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        a = warped[..., 3].astype(np.float32) / 255.0
        rgb = rgb * (1.0 - a[..., None]) + warped[..., :3].astype(np.float32) * a[
            ..., None
        ]
        acc = acc * (1.0 - a) + a

    return rgb / 255.0, acc


GAP_SCALE = 0.5  # проверка щелей считается в половинном разрешении
GAP_SHRINK = 0.85  # капсулу поджимаем, чтобы не ловить сглаженный край


def _gaps(
    rig: dict,
    pose: dict[str, tuple[float, np.ndarray]],
    acc: np.ndarray,
    pad: tuple[float, float],
) -> tuple[int, int]:
    """(непокрытые пиксели тела, всего пикселей тела) для кадра.

    Тело в текущей позе — объединение капсул скелета. Всё, что скелет
    считает телом, а слои не закрыли, и есть щель в суставе. Замкнутые
    просветы (подмышка, просвет между ног) при этом не считаются дефектом:
    капсул там нет.
    """
    small = cv2.resize(
        acc, None, fx=GAP_SCALE, fy=GAP_SCALE, interpolation=cv2.INTER_AREA
    )
    shape = small.shape
    grid = pixel_grid(shape)
    body = np.zeros(shape, dtype=bool)
    for bone in rig["bones"]:
        corr, position = pose[bone["name"]]
        rest_a = np.array(bone["rest_a"], dtype=float)
        rest_b = np.array(bone["rest_b"], dtype=float)
        end = position + rot(corr) @ (rest_b - rest_a)
        a = (position + np.array(pad)) * GAP_SCALE
        b = (end + np.array(pad)) * GAP_SCALE
        field = capsule_field(
            shape,
            (a[0], a[1]),
            (b[0], b[1]),
            bone["radius_a"] * GAP_SCALE * GAP_SHRINK,
            bone["radius_b"] * GAP_SCALE * GAP_SHRINK,
            grid=grid,
        )
        body |= field < 0
    return int((body & (small < 0.5)).sum()), int(body.sum())


def swing(
    char: Character,
    rig: dict,
    images: dict[str, np.ndarray],
    frames: int = FRAMES,
    strip_frames: int = 8,
    strip_height: int = 320,
    gif_height: int = 600,
) -> dict:
    """Прогнать стресс-свинг, записать preview.gif и preview_strip.png."""
    width, height = rig["size"]
    pad_x = pad_y = round(max(width, height) * PAD_RATIO)
    canvas = (height + 2 * pad_y, width + 2 * pad_x)

    rendered: list[tuple[np.ndarray, np.ndarray]] = []
    gap_px, worst_ratio, worst_frame = 0, 0.0, 0
    for i in range(frames):
        deltas = swing_deltas(2 * math.pi * i / frames)
        rgb, acc = render_frame(rig, images, deltas, canvas, (pad_x, pad_y))
        rendered.append((rgb, acc))
        gaps, body = _gaps(rig, solve_fk(rig, deltas), acc, (pad_x, pad_y))
        if gaps / max(body, 1) > worst_ratio:
            gap_px, worst_ratio, worst_frame = gaps, gaps / max(body, 1), i

    background = np.array(BG_COLOR, dtype=np.float32) / 255.0
    gif_scale = min(1.0, gif_height / canvas[0])
    gif = []
    for rgb, acc in rendered:
        flat = np.clip(rgb + background * (1.0 - acc[..., None]), 0, 1) * 255
        flat = flat.astype(np.uint8)
        if gif_scale < 1.0:
            flat = cv2.resize(
                flat,
                (round(canvas[1] * gif_scale), round(canvas[0] * gif_scale)),
                interpolation=cv2.INTER_AREA,
            )
        image = Image.fromarray(flat)
        gif.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    gif[0].save(
        char.preview,
        save_all=True,
        append_images=gif[1:],
        duration=int(1000 / frames),
        loop=0,
        disposal=2,
    )

    step = max(1, frames // strip_frames)
    picked = rendered[::step][:strip_frames]
    scale = strip_height / canvas[0]
    tiles = []
    for rgb, acc in picked:
        rgba = np.dstack([np.clip(rgb, 0, 1) * 255, np.clip(acc, 0, 1) * 255])
        tile = cv2.resize(
            rgba.astype(np.uint8),
            (max(1, round(canvas[1] * scale)), strip_height),
            interpolation=cv2.INTER_AREA,
        )
        tiles.append(tile)
    char.write_rgba(char.preview_strip, np.hstack(tiles))

    return {
        "frames": frames,
        "gap_px": int(gap_px / (GAP_SCALE * GAP_SCALE)),
        "gap_ratio": round(worst_ratio, 5),
        "worst_frame": worst_frame,
    }


def triage(checks: dict) -> str:
    """Светофор автотриажа (принцип №3 в DESIGN.md)."""
    gap = checks.get("gap_ratio", 0.0)
    uncovered = checks.get("uncovered_ratio", 0.0)
    if gap < 0.002 and uncovered < 0.0005:
        return "green"
    if gap < 0.01 and uncovered < 0.005:
        return "yellow"
    return "red"
