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
from ..geometry import capsule_field, pixel_grid, rot, solve_chain

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
    order = [b["name"] for b in rig["bones"]]  # родители раньше детей — из rig.py
    parent_of = {b["name"]: b["parent"] for b in rig["bones"]}
    rest = {b["name"]: np.array(b["rest_a"], dtype=float) for b in rig["bones"]}
    corr, pos = solve_chain(order, parent_of, rest, deltas)
    return {name: (corr[name], pos[name]) for name in order}


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


def _write_preview(
    char: Character,
    rendered: list[tuple[np.ndarray, np.ndarray]],
    canvas: tuple[int, int],
    frames: int,
    strip_frames: int,
    strip_height: int,
    gif_height: int,
) -> None:
    """Записать preview.gif и preview_strip.png из готовых кадров."""
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
        gif.append(
            Image.fromarray(flat).convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=128
            )
        )
    gif[0].save(
        char.preview,
        save_all=True,
        append_images=gif[1:],
        duration=int(1000 / frames),
        loop=0,
        disposal=2,
    )

    step = max(1, frames // strip_frames)
    scale = strip_height / canvas[0]
    tiles = []
    for rgb, acc in rendered[::step][:strip_frames]:
        rgba = np.dstack([np.clip(rgb, 0, 1) * 255, np.clip(acc, 0, 1) * 255])
        tiles.append(
            cv2.resize(
                rgba.astype(np.uint8),
                (max(1, round(canvas[1] * scale)), strip_height),
                interpolation=cv2.INTER_AREA,
            )
        )
    char.write_rgba(char.preview_strip, np.hstack(tiles))


def swing_mesh(
    char: Character,
    rig: dict,
    texture: np.ndarray,
    alpha: np.ndarray,
    frames: int = FRAMES,
    strip_frames: int = 8,
    strip_height: int = 320,
    gif_height: int = 600,
) -> dict:
    """Стресс-свинг деформацией сетки — без нарезки на жёсткие части."""
    from .. import mesh

    width, height = rig["size"]
    pad_x = pad_y = round(max(width, height) * PAD_RATIO)
    canvas = (height + 2 * pad_y, width + 2 * pad_x)

    unit = float(rig.get("unit", max(width, height)))
    points, triangles = mesh.build(alpha, unit)
    weights = mesh.skin_weights(points, rig)

    rendered = []
    for i in range(frames):
        deltas = swing_deltas(2 * math.pi * i / frames)
        moved = mesh.deform(points, weights, rig, solve_fk(rig, deltas))
        frame = mesh.render(
            texture, points, triangles, moved, canvas, (pad_x, pad_y)
        )
        rgb = frame[..., :3].astype(np.float32) / 255.0
        acc = frame[..., 3].astype(np.float32) / 255.0
        rendered.append((rgb, acc))

    _write_preview(char, rendered, canvas, frames, strip_frames, strip_height,
                   gif_height)
    return {
        "frames": frames,
        "mesh_points": int(len(points)),
        "mesh_triangles": int(len(triangles)),
    }


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

    # поза покоя — точка отсчёта: щели в ней означают не швы, а криво
    # севший скелет (капсула торчит туда, где арта нет)
    rest = {name: 0.0 for name in SWING}
    _, rest_acc = render_frame(rig, images, rest, canvas, (pad_x, pad_y))
    rest_gaps, rest_body = _gaps(rig, solve_fk(rig, rest), rest_acc, (pad_x, pad_y))
    fit_ratio = rest_gaps / max(rest_body, 1)

    rendered: list[tuple[np.ndarray, np.ndarray]] = []
    gap_px, worst_ratio, worst_frame = 0, 0.0, 0
    for i in range(frames):
        deltas = swing_deltas(2 * math.pi * i / frames)
        rgb, acc = render_frame(rig, images, deltas, canvas, (pad_x, pad_y))
        rendered.append((rgb, acc))
        gaps, body = _gaps(rig, solve_fk(rig, deltas), acc, (pad_x, pad_y))
        if gaps / max(body, 1) > worst_ratio:
            gap_px, worst_ratio, worst_frame = gaps, gaps / max(body, 1), i

    _write_preview(
        char, rendered, canvas, frames, strip_frames, strip_height, gif_height
    )

    scale = GAP_SCALE * GAP_SCALE
    return {
        "frames": frames,
        "seam_gap_px": int(max(gap_px - rest_gaps, 0) / scale),
        "seam_gap_ratio": round(max(worst_ratio - fit_ratio, 0.0), 5),
        "rest_gap_ratio": round(fit_ratio, 5),
        "worst_frame": worst_frame,
    }


def triage(checks: dict) -> str:
    """Светофор автотриажа (принцип №3 в DESIGN.md).

    Три разных дефекта — три числа, каждое меряется там, где оно осмысленно:

    - `seam_gap_ratio` — насколько при движении открывается тело, которого
      не закрыл ни один слой. Это швы, лечится нахлёстом.
    - `fit_inside` (с этапа позы) — какая доля капсул скелета лежит внутри
      арта. Это посадка, лечится правкой суставов. Не IoU: уши, клочья
      шерсти и пушистый хвост капсулой не описываются, и IoU на живом арте
      упирается в потолок, не имеющий отношения к качеству рига.
    - `uncovered_ratio` — пиксели арта, не попавшие ни в одну часть.

    Щели в позе покоя (`rest_gap_ratio`) в триаже не участвуют: у живого
    арта силуэт всегда немного не совпадает с капсулами (клочья шерсти,
    одежда), и это не дефект — потому и вычитается из щелей в движении.
    """
    seam = checks.get("seam_gap_ratio", 0.0)
    uncovered = checks.get("uncovered_ratio", 0.0)
    fit = checks.get("fit_inside", 1.0)
    if seam < 0.002 and uncovered < 0.0005 and fit >= 0.95:
        return "green"
    if seam < 0.01 and uncovered < 0.005 and fit >= 0.88:
        return "yellow"
    return "red"
