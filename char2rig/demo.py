"""Процедурный полосатый кот из капсул шаблона.

Нужен, чтобы гонять весь конвейер без моделей и без чужого арта, и сразу
бить в главный риск фазы 0: полосы в суставах. Рисуется теми же капсулами,
по которым потом идёт сегментация, но конвейер об этом не знает — на вход
ему приходит обычный RGBA-PNG.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("нужен opencv: pip install -r requirements.txt") from exc

from .geometry import capsule_axis_param, capsule_field, capsule_side, pixel_grid
from .template import Template, extent

FUR = (214, 150, 86)
STRIPE = (146, 86, 44)
LIGHT = (243, 226, 200)  # живот, морда, «носки», кончик хвоста
EAR_INNER = (226, 158, 150)
EYE = (112, 182, 98)
DARK = (38, 30, 26)
NOSE = (206, 122, 122)

LIGHT_MIX = {  # насколько часть уходит в светлый цвет
    "hand_l": 0.55,
    "hand_r": 0.55,
    "foot_l": 0.55,
    "foot_r": 0.55,
}


def _rgb(color: tuple[int, int, int]) -> np.ndarray:
    return np.array(color, dtype=np.float32)


class _Canvas:
    """RGB + альфа во float, композит «сверху» с готовым сглаживанием."""

    def __init__(self, shape: tuple[int, int]):
        self.rgb = np.zeros((*shape, 3), dtype=np.float32)
        self.alpha = np.zeros(shape, dtype=np.float32)

    def over(self, color: np.ndarray, alpha: np.ndarray) -> None:
        a = alpha[..., None]
        self.rgb = self.rgb * (1.0 - a) + color * a
        self.alpha = self.alpha * (1.0 - alpha) + alpha

    def to_rgba(self) -> np.ndarray:
        # холст копится с premultiplied-цветом (стартует с нуля) — на выходе
        # делим обратно, иначе по краю получится двойное умножение на альфу
        # и тёмная кайма вокруг персонажа
        straight = self.rgb / np.maximum(self.alpha, 1e-3)[..., None]
        rgba = np.zeros((*self.alpha.shape, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(straight, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(self.alpha * 255, 0, 255).astype(np.uint8)
        return rgba


def _aa(field: np.ndarray) -> np.ndarray:
    """Сглаженная альфа из поля расстояний (полпикселя на границе)."""
    return np.clip(0.5 - field, 0.0, 1.0)


def _noise(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random(shape, dtype=np.float32)
    smooth = cv2.GaussianBlur(raw, (0, 0), sigmaX=3.0)
    smooth -= smooth.mean()
    peak = max(float(np.abs(smooth).max()), 1e-6)
    return smooth / peak


def place_template(
    template: Template, size: tuple[int, int], margin: float = 0.04
) -> tuple[dict[str, tuple[float, float]], float]:
    """Вписать шаблон в холст. Вернуть (суставы в пикселях, пикселей на рост)."""
    width, height = size
    ex0, ex1, ey0, ey1 = extent(template, 1.0)
    unit = height * (1 - 2 * margin) / (ey1 - ey0)
    oy = height * margin - ey0 * unit
    ox = width / 2 - (ex0 + ex1) / 2 * unit
    joints = {
        name: (ox + x * unit, oy + y * unit)
        for name, (x, y) in template.joints.items()
    }
    return joints, unit


def draw(template: Template, size: tuple[int, int] = (512, 768)) -> np.ndarray:
    """Нарисовать демо-кота. size = (ширина, высота)."""
    width, height = size
    shape = (height, width)
    grid = pixel_grid(shape)
    joints, unit = place_template(template, size)
    canvas = _Canvas(shape)

    def norm(x: float, y: float) -> tuple[float, float]:
        """Нормализованная точка шаблона → пиксели (через тот же масштаб)."""
        jx, jy = joints["head_top"]
        tx, ty = template.joints["head_top"]
        return (jx + (x - tx) * unit, jy + (y - ty) * unit)

    fur = _rgb(FUR)
    stripe = _rgb(STRIPE)
    light = _rgb(LIGHT)
    grain = _noise(shape, seed=7)

    # --- тело: капсулы с полосами, поперечными оси кости ------------------
    for index, bone in enumerate(sorted(template.bones, key=lambda b: b.z)):
        a, b = joints[bone.a], joints[bone.b]
        field = capsule_field(shape, a, b, bone.ra * unit, bone.rb * unit, grid=grid)
        alpha = _aa(field)
        if alpha.max() <= 0:
            continue

        radius = max((bone.ra + bone.rb) / 2 * unit, 1.0)
        depth = np.clip(-field / radius, 0.0, 1.0)  # 1 в центре, 0 у края
        t = np.clip(capsule_axis_param(shape, a, b, grid=grid), 0.0, 1.0)
        side = capsule_side(shape, a, b, grid=grid) / radius

        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        bands = max(2.0, length / (0.055 * unit))
        wave = np.sin(t * bands * 2 * np.pi + index * 1.7 + 0.7 * np.sin(side * 2.2))
        stripe_mix = np.clip((wave - 0.25) * 2.2, 0.0, 1.0) * (0.35 + 0.65 * depth)

        color = fur * (1 - stripe_mix[..., None]) + stripe * stripe_mix[..., None]

        belly = LIGHT_MIX.get(bone.name, 0.0)
        if bone.name == "torso":
            # светлое брюхо по центральной полосе, шире книзу
            belly_mask = np.clip(1.0 - np.abs(side) / (0.55 + 0.35 * t), 0.0, 1.0)
            belly_mask *= np.clip((t - 0.05) * 1.8, 0.0, 1.0)
            color = color * (1 - belly_mask[..., None] * 0.85) + light * (
                belly_mask[..., None] * 0.85
            )
        elif bone.name == "tail_3":
            tip = np.clip((t - 0.45) * 2.0, 0.0, 1.0)
            color = color * (1 - tip[..., None]) + light * tip[..., None]
        elif belly:
            color = color * (1 - belly) + light * belly

        shade = 0.74 + 0.26 * depth + 0.05 * grain
        canvas.over(color * shade[..., None], alpha)

    # --- уши: торчат из головы, при сегментации достаются ей же -----------
    for sign in (-1, 1):
        base = norm(0.5 + sign * 0.055, 0.075)
        tip = norm(0.5 + sign * 0.085, -0.015)
        outer = capsule_field(shape, base, tip, 0.038 * unit, 0.006 * unit, grid=grid)
        canvas.over(fur * 0.9, _aa(outer))
        inner = capsule_field(
            shape,
            norm(0.5 + sign * 0.055, 0.068),
            norm(0.5 + sign * 0.078, 0.005),
            0.022 * unit,
            0.004 * unit,
            grid=grid,
        )
        canvas.over(_rgb(EAR_INNER), _aa(inner))

    # --- морда -------------------------------------------------------------
    muzzle = capsule_field(
        shape, norm(0.472, 0.140), norm(0.528, 0.140), 0.036 * unit, 0.036 * unit,
        grid=grid,
    )
    canvas.over(light, _aa(muzzle) * 0.95)

    for sign in (-1, 1):
        eye = capsule_field(
            shape, norm(0.5 + sign * 0.038, 0.094), norm(0.5 + sign * 0.038, 0.094),
            0.021 * unit, 0.021 * unit, grid=grid,
        )
        canvas.over(_rgb(EYE), _aa(eye))
        pupil = capsule_field(
            shape, norm(0.5 + sign * 0.038, 0.086), norm(0.5 + sign * 0.038, 0.102),
            0.006 * unit, 0.006 * unit, grid=grid,
        )
        canvas.over(_rgb(DARK), _aa(pupil))
        for row, spread in ((0.138, 0.020), (0.146, 0.004), (0.154, -0.012)):
            whisker = capsule_field(
                shape,
                norm(0.5 + sign * 0.030, 0.144),
                norm(0.5 + sign * 0.150, row + spread),
                0.0035 * unit,
                0.0015 * unit,
                grid=grid,
            )
            canvas.over(_rgb(DARK), _aa(whisker) * 0.7)

    nose = capsule_field(
        shape, norm(0.5, 0.126), norm(0.5, 0.132), 0.013 * unit, 0.009 * unit,
        grid=grid,
    )
    canvas.over(_rgb(NOSE), _aa(nose))

    # --- обводка силуэта: лёгкая тень по краю, как в painterly-арте -------
    solid = (canvas.alpha > 0.5).astype(np.uint8)
    if solid.any():
        inside = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
        rim = np.clip(1.0 - inside / (0.008 * unit), 0.0, 1.0) * canvas.alpha
        canvas.rgb *= (1.0 - 0.38 * rim)[..., None]

    return canvas.to_rgba()
