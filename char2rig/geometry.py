"""Геометрия конвейера: повороты, капсулы, поля расстояний.

Система координат везде пиксельная: x вправо, y вниз (как в изображении).
Угол считается через ``atan2(dy, dx)``, поэтому положительный угол выглядит
на экране как поворот по часовой стрелке. Одна и та же матрица поворота
используется и в FK, и в рендере — смешивать конвенции нельзя.
"""

from __future__ import annotations

import numpy as np

Point = tuple[float, float]


def rot(angle: float) -> np.ndarray:
    """Матрица поворота 2x2 для векторов в пиксельных координатах."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def angle_of(a: Point, b: Point) -> float:
    """Абсолютный угол направления a → b."""
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def length_of(a: Point, b: Point) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def solve_chain(
    order: list[str],
    parent_of: dict[str, str | None],
    rest_start: dict[str, np.ndarray],
    deltas: dict[str, float],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """FK по цепочке костей: (поправка угла, положение начального сустава).

    Для каждой кости хранится не мировая матрица, а «поправка» — насколько
    её угол отличается от покоя::

        corr[кость] = corr[родитель] + delta[кость]
        pos[кость]  = pos[родитель] + rot(corr[родитель]) · (rest - rest_род)

    `order` должен идти родителями вперёд. Кость без родителя (или с
    родителем, которого нет в наборе) считается корневой.
    """
    corr: dict[str, float] = {}
    pos: dict[str, np.ndarray] = {}
    for name in order:
        parent = parent_of.get(name)
        delta = float(deltas.get(name, 0.0))
        if parent is None or parent not in pos:
            corr[name], pos[name] = delta, rest_start[name]
            continue
        corr[name] = corr[parent] + delta
        pos[name] = pos[parent] + rot(corr[parent]) @ (
            rest_start[name] - rest_start[parent]
        )
    return corr, pos


def pixel_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Сетка координат (xs, ys) для изображения shape=(h, w)."""
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    return xs.astype(np.float32), ys.astype(np.float32)


def capsule_field(
    shape: tuple[int, int],
    a: Point,
    b: Point,
    ra: float,
    rb: float,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Поле расстояний до капсулы a→b с радиусами ra→rb: <0 внутри, >0 снаружи.

    Радиус интерполируется линейно вдоль оси — это приближение усечённого
    конуса, точности за глаза хватает и для масок, и для сглаживания края.
    """
    xs, ys = grid if grid is not None else pixel_grid(shape)
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    denom = max(dx * dx + dy * dy, 1e-6)
    t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / denom, 0.0, 1.0)
    d = np.hypot(xs - (ax + t * dx), ys - (ay + t * dy))
    return d - (ra + (rb - ra) * t)


def capsule_axis_param(
    shape: tuple[int, int],
    a: Point,
    b: Point,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Параметр проекции пикселя на ось капсулы, 0 в a и 1 в b (без клипа)."""
    xs, ys = grid if grid is not None else pixel_grid(shape)
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    denom = max(dx * dx + dy * dy, 1e-6)
    return ((xs - ax) * dx + (ys - ay) * dy) / denom


def capsule_side(
    shape: tuple[int, int],
    a: Point,
    b: Point,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Знаковое расстояние пикселя от оси капсулы поперёк неё (в пикселях)."""
    xs, ys = grid if grid is not None else pixel_grid(shape)
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    length = max(np.hypot(dx, dy), 1e-6)
    nx, ny = -dy / length, dx / length
    return (xs - ax) * nx + (ys - ay) * ny
