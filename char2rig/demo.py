"""Процедурный кот из капсул шаблона — тестовый персонаж и генератор данных.

Нужен, чтобы гонять весь конвейер без моделей и без чужого арта, и сразу
бить в главный риск фазы 0: полосы в суставах. Рисуется теми же капсулами,
по которым потом идёт сегментация, но конвейер об этом не знает — на вход
ему приходит обычный RGBA-PNG.

Рисовалка заодно отдаёт карту частей: какой кости принадлежит каждый
пиксель. Это точная разметка (кто нарисован сверху, тот и владелец), а не
догадка по ближайшей капсуле — из неё собирается обучающий набор
(`dataset.py`), которого нет ни в одном публичном датасете: там части
размечены по смыслу, а риг требует разрезов по суставам.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("нужен opencv: pip install -r requirements.txt") from exc

from .geometry import capsule_axis_param, capsule_field, capsule_side, pixel_grid, rot
from .template import Template, extent

Color = tuple[int, int, int]


@dataclass(frozen=True)
class Look:
    """Внешность персонажа: цвета и параметры узора."""

    fur: Color = (214, 150, 86)
    stripe: Color = (146, 86, 44)
    light: Color = (243, 226, 200)  # живот, морда, «носки», кончик хвоста
    ear: Color = (226, 158, 150)
    eye: Color = (112, 182, 98)
    nose: Color = (206, 122, 122)
    dark: Color = (38, 30, 26)
    stripe_density: float = 0.055  # шаг полос в долях роста
    stripe_strength: float = 1.0  # 0 — гладкая шерсть
    sock: float = 0.55  # насколько светлеют кисти и стопы
    belly: float = 0.85
    rim: float = 0.38  # тень по контуру
    grain: float = 0.05  # зернистость
    seed: int = 7
    whiskers: bool = True
    ears: bool = True
    # шерсть по контуру: без неё силуэт — объединение идеальных капсул, и
    # обучать на нём сегментатор для живого арта бессмысленно
    fluff: float = 0.018  # размах клочьев в долях роста
    fluff_scale: float = 0.9  # крупные космы или мелкий ворс
    toes: int = 3  # пальцы на кистях и стопах
    ruff: float = 0.5  # воротник на груди


DEFAULT_LOOK = Look()

_SOCK_BONES = ("hand_l", "hand_r", "foot_l", "foot_r")


def _rgb(color: Color) -> np.ndarray:
    return np.array(color, dtype=np.float32)


class _Canvas:
    """RGB + альфа во float, композит «сверху» с готовым сглаживанием."""

    def __init__(self, shape: tuple[int, int]):
        self.rgb = np.zeros((*shape, 3), dtype=np.float32)
        self.alpha = np.zeros(shape, dtype=np.float32)
        self.labels = np.zeros(shape, dtype=np.uint8)

    def over(
        self, color: np.ndarray, alpha: np.ndarray, label: int | None = None
    ) -> None:
        a = alpha[..., None]
        self.rgb = self.rgb * (1.0 - a) + color * a
        self.alpha = self.alpha * (1.0 - alpha) + alpha
        if label is not None:
            self.labels[alpha > 0.5] = label

    def to_rgba(self) -> np.ndarray:
        # холст копится с premultiplied-цветом (стартует с нуля) — на выходе
        # делим обратно, иначе по краю получится двойное умножение на альфу
        # и тёмная кайма вокруг персонажа
        straight = self.rgb / np.maximum(self.alpha, 1e-3)[..., None]
        rgba = np.zeros((*self.alpha.shape, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(straight, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(self.alpha * 255, 0, 255).astype(np.uint8)
        return rgba


def _aa(field_: np.ndarray) -> np.ndarray:
    """Сглаженная альфа из поля расстояний (полпикселя на границе)."""
    return np.clip(0.5 - field_, 0.0, 1.0)


def _noise(shape: tuple[int, int], seed: int, sigma: float = 3.0) -> np.ndarray:
    """Гладкий шум в диапазоне примерно [-1, 1]."""
    rng = np.random.default_rng(seed)
    raw = rng.random(shape, dtype=np.float32)
    smooth = cv2.GaussianBlur(raw, (0, 0), sigmaX=max(sigma, 0.5))
    smooth -= smooth.mean()
    peak = max(float(np.abs(smooth).max()), 1e-6)
    return smooth / peak


def _toes(
    shape: tuple[int, int],
    bone,
    a: tuple[float, float],
    b: tuple[float, float],
    unit: float,
    look: Look,
    grid: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Пальцы на конце кисти или стопы — веером поперёк оси кости."""
    axis = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    length = float(np.hypot(*axis))
    if length < 1e-3:
        return np.full(shape, 1e6, dtype=np.float32)
    axis /= length
    across = np.array([-axis[1], axis[0]])
    radius = bone.rb * unit
    field_ = np.full(shape, 1e6, dtype=np.float32)
    for i in range(look.toes):
        offset = (i - (look.toes - 1) / 2) / max(look.toes - 1, 1)
        base = np.array(b) - axis * radius * 0.5 + across * offset * radius * 1.1
        tip = base + axis * radius * 0.9 + across * offset * radius * 0.5
        field_ = np.minimum(
            field_,
            capsule_field(
                shape,
                (base[0], base[1]),
                (tip[0], tip[1]),
                radius * 0.42,
                radius * 0.34,
                grid=grid,
            ),
        )
    return field_


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


def fit_to_canvas(
    template: Template,
    joints: dict[str, tuple[float, float]],
    unit: float,
    size: tuple[int, int],
    margin: float = 0.05,
) -> tuple[dict[str, tuple[float, float]], float]:
    """Вписать позу в холст: после поворота суставов конечности вылезают.

    Обрезанный краем персонаж в обучающем наборе — брак, а не разнообразие.
    """
    width, height = size
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for bone in template.bones:
        for joint, radius in ((bone.a, bone.ra), (bone.b, bone.rb)):
            x, y = joints[joint]
            reach = radius * unit
            x0, x1 = min(x0, x - reach), max(x1, x + reach)
            y0, y1 = min(y0, y - reach), max(y1, y + reach)

    room_x, room_y = width * (1 - 2 * margin), height * (1 - 2 * margin)
    scale = min(1.0, room_x / max(x1 - x0, 1e-6), room_y / max(y1 - y0, 1e-6))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    moved = {
        name: (width / 2 + (x - cx) * scale, height / 2 + (y - cy) * scale)
        for name, (x, y) in joints.items()
    }
    return moved, unit * scale


def draw(
    template: Template,
    size: tuple[int, int] = (512, 768),
    joints: dict[str, tuple[float, float]] | None = None,
    corr: dict[str, float] | None = None,
    look: Look = DEFAULT_LOOK,
    unit: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Нарисовать кота. Вернуть (RGBA, карта частей).

    В карте частей 0 — фон, i+1 — индекс кости в `template.bones`.
    `joints` — уже посаженные (и, если надо, повёрнутые) суставы в пикселях;
    `corr` — поправки углов костей, чтобы морда ехала вместе с головой.
    """
    width, height = size
    shape = (height, width)
    grid = pixel_grid(shape)
    placed, placed_unit = place_template(template, size)
    joints = joints or placed
    unit = unit or placed_unit
    corr = corr or {}
    canvas = _Canvas(shape)
    index = {bone.name: i + 1 for i, bone in enumerate(template.bones)}

    head_bone = template.bone("head") if any(
        b.name == "head" for b in template.bones
    ) else None

    def head_point(x: float, y: float) -> tuple[float, float]:
        """Точка морды из нормализованных координат шаблона — в пиксели.

        Едет вместе с головой: то же преобразование, что и у её кости.
        """
        if head_bone is None:
            return (x * unit, y * unit)
        anchor = np.array(template.joints[head_bone.a], dtype=float)
        offset = (np.array([x, y], dtype=float) - anchor) * unit
        base = np.array(joints[head_bone.a], dtype=float)
        moved = base + rot(corr.get(head_bone.name, 0.0)) @ offset
        return (float(moved[0]), float(moved[1]))

    fur, stripe, light = _rgb(look.fur), _rgb(look.stripe), _rgb(look.light)
    grain = _noise(shape, look.seed) * look.grain

    # смещение контура: вычитаем из поля расстояний — граница капсулы
    # начинает гулять на ±fur, и силуэт перестаёт быть склейкой овалов
    tufts = np.zeros(shape, dtype=np.float32)
    if look.fluff:
        sigma = max(look.fluff_scale * unit * 0.02, 0.8)
        tufts = _noise(shape, look.seed + 101, sigma=sigma) * look.fluff * unit
        # шерсть не везде одинаковая: воротник на груди и пушистый хвост
        xs, ys = grid
        gain = np.ones(shape, dtype=np.float32)
        for joint, extra, spread in (("neck", look.ruff, 0.13), ("tail_c", 0.8, 0.10)):
            if joint not in joints or extra <= 0:
                continue
            jx, jy = joints[joint]
            sigma = max(spread * unit, 1.0)
            gain += extra * np.exp(
                -((xs - jx) ** 2 + (ys - jy) ** 2) / (2 * sigma**2)
            )
        tufts *= gain

    # --- тело: капсулы с полосами, поперечными оси кости ------------------
    for order, bone in enumerate(sorted(template.bones, key=lambda b: b.z)):
        a, b = joints[bone.a], joints[bone.b]
        field_ = capsule_field(shape, a, b, bone.ra * unit, bone.rb * unit, grid=grid)
        if bone.name in _SOCK_BONES and look.toes:
            field_ = np.minimum(field_, _toes(shape, bone, a, b, unit, look, grid))
        alpha = _aa(field_ - tufts)
        if alpha.max() <= 0:
            continue

        radius = max((bone.ra + bone.rb) / 2 * unit, 1.0)
        depth = np.clip(-field_ / radius, 0.0, 1.0)  # 1 в центре, 0 у края
        t = np.clip(capsule_axis_param(shape, a, b, grid=grid), 0.0, 1.0)
        side = capsule_side(shape, a, b, grid=grid) / radius

        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        bands = max(2.0, length / max(look.stripe_density * unit, 1e-3))
        wave = np.sin(t * bands * 2 * np.pi + order * 1.7 + 0.7 * np.sin(side * 2.2))
        mix = np.clip((wave - 0.25) * 2.2, 0.0, 1.0) * (0.35 + 0.65 * depth)
        mix *= look.stripe_strength
        color = fur * (1 - mix[..., None]) + stripe * mix[..., None]

        if bone.name == "torso":
            # светлое брюхо по центральной полосе, шире книзу
            belly = np.clip(1.0 - np.abs(side) / (0.55 + 0.35 * t), 0.0, 1.0)
            belly *= np.clip((t - 0.05) * 1.8, 0.0, 1.0) * look.belly
            color = color * (1 - belly[..., None]) + light * belly[..., None]
        elif bone.name == "tail_3":
            tip = np.clip((t - 0.45) * 2.0, 0.0, 1.0)
            color = color * (1 - tip[..., None]) + light * tip[..., None]
        elif bone.name in _SOCK_BONES and look.sock:
            color = color * (1 - look.sock) + light * look.sock

        shade = 0.74 + 0.26 * depth + grain
        canvas.over(color * shade[..., None], alpha, index[bone.name])

    head_label = index.get("head")

    # --- уши: торчат из головы, при сегментации достаются ей же -----------
    if look.ears and head_bone is not None:
        for sign in (-1, 1):
            outer = capsule_field(
                shape,
                head_point(0.5 + sign * 0.055, 0.075),
                head_point(0.5 + sign * 0.085, -0.015),
                0.038 * unit,
                0.006 * unit,
                grid=grid,
            )
            canvas.over(fur * 0.9, _aa(outer - tufts * 0.7), head_label)
            inner = capsule_field(
                shape,
                head_point(0.5 + sign * 0.055, 0.068),
                head_point(0.5 + sign * 0.078, 0.005),
                0.022 * unit,
                0.004 * unit,
                grid=grid,
            )
            canvas.over(_rgb(look.ear), _aa(inner), head_label)

    if head_bone is not None:
        muzzle = capsule_field(
            shape,
            head_point(0.472, 0.140),
            head_point(0.528, 0.140),
            0.036 * unit,
            0.036 * unit,
            grid=grid,
        )
        canvas.over(light, _aa(muzzle) * 0.95, head_label)

        for sign in (-1, 1):
            eye = capsule_field(
                shape,
                head_point(0.5 + sign * 0.038, 0.094),
                head_point(0.5 + sign * 0.038, 0.094),
                0.021 * unit,
                0.021 * unit,
                grid=grid,
            )
            canvas.over(_rgb(look.eye), _aa(eye), head_label)
            pupil = capsule_field(
                shape,
                head_point(0.5 + sign * 0.038, 0.086),
                head_point(0.5 + sign * 0.038, 0.102),
                0.006 * unit,
                0.006 * unit,
                grid=grid,
            )
            canvas.over(_rgb(look.dark), _aa(pupil), head_label)
            if not look.whiskers:
                continue
            for row, spread in ((0.138, 0.020), (0.146, 0.004), (0.154, -0.012)):
                whisker = capsule_field(
                    shape,
                    head_point(0.5 + sign * 0.030, 0.144),
                    head_point(0.5 + sign * 0.150, row + spread),
                    0.0035 * unit,
                    0.0015 * unit,
                    grid=grid,
                )
                canvas.over(_rgb(look.dark), _aa(whisker) * 0.7, head_label)

        nose = capsule_field(
            shape,
            head_point(0.5, 0.126),
            head_point(0.5, 0.132),
            0.013 * unit,
            0.009 * unit,
            grid=grid,
        )
        canvas.over(_rgb(look.nose), _aa(nose), head_label)

    # --- обводка силуэта: лёгкая тень по краю, как в painterly-арте -------
    solid = (canvas.alpha > 0.5).astype(np.uint8)
    if solid.any() and look.rim:
        inside = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
        edge = np.clip(1.0 - inside / (0.008 * unit), 0.0, 1.0) * canvas.alpha
        canvas.rgb *= (1.0 - look.rim * edge)[..., None]

    labels = np.where(canvas.alpha > 0.5, canvas.labels, 0).astype(np.uint8)
    return canvas.to_rgba(), labels
