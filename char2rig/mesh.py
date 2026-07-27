"""Деформация сеткой: персонаж гнётся, а не перекладывается досками.

Нарезка на жёсткие части — тупик по качеству. Сколько ни расширяй нахлёст,
кусок остаётся плоской доской: при повороте у него вылезают углы из-под
соседа, а на сгибе видно, что это две отдельные картинки, а не одно тело.
Ровно поэтому AnimatedDrawings у Meta не режет персонажа вовсе, а натягивает
на него треугольную сетку и гнёт её.

Здесь то же самое:

1. по силуэту строится сетка — контур плюс сетка внутренних точек;
2. каждая вершина получает веса костей: не «эта вершина принадлежит бедру», а
   «на 70% бедру, на 30% торсу». Веса берутся из тех же капсульных полей, по
   которым идёт нарезка, и плавно перетекают у сустава — это и есть морфинг
   на стыке;
3. в кадре вершина едет как взвешенная сумма своих костей, а текстура
   натягивается на треугольники аффинно.

Швов нет по построению: тело одно, разрезов в нём нет. Нарезка на слои при
этом никуда не девается — она нужна для экспорта в движки, которые умеют
только cutout, — но превью и проверка качества идут по сетке.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import rot

WEIGHT_SOFTNESS = 0.55  # доля радиуса кости, на которой веса перетекают
WEIGHT_BONES = 4  # сколько костей влияет на вершину


def _bone_distance(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, ra: float, rb: float
) -> np.ndarray:
    """Расстояние от точек до капсулы кости (внутри — ноль)."""
    axis = b - a
    length2 = float(axis @ axis)
    if length2 < 1e-6:
        return np.linalg.norm(points - a, axis=1) - ra
    t = np.clip((points - a) @ axis / length2, 0.0, 1.0)
    nearest = a + t[:, None] * axis
    return np.linalg.norm(points - nearest, axis=1) - (ra + (rb - ra) * t)


def skin_weights(points: np.ndarray, rig: dict) -> np.ndarray:
    """Веса костей для каждой вершины: (вершины, кости), сумма по строке = 1.

    Мягкость привязки задаётся толщиной кости: у тонкой кисти переход узкий,
    у торса широкий. Без этого сустав либо ломается углом, либо размазывается.
    """
    bones = rig["bones"]
    distances = np.empty((len(points), len(bones)), dtype=np.float64)
    softness = np.empty(len(bones), dtype=np.float64)
    for index, bone in enumerate(bones):
        a = np.array(bone["rest_a"], dtype=float)
        b = np.array(bone["rest_b"], dtype=float)
        ra, rb = float(bone["radius_a"]), float(bone["radius_b"])
        distances[:, index] = np.maximum(_bone_distance(points, a, b, ra, rb), 0.0)
        softness[index] = max(WEIGHT_SOFTNESS * min(ra, rb), 2.0)

    weights = np.exp(-distances / softness)
    # оставляем несколько ближайших костей: дальние дают только шум
    if len(bones) > WEIGHT_BONES:
        cut = np.partition(weights, -WEIGHT_BONES, axis=1)[:, -WEIGHT_BONES]
        weights[weights < cut[:, None]] = 0.0
    total = weights.sum(axis=1, keepdims=True)
    # вершина, до которой не дотянулась ни одна кость, цепляется к ближайшей
    orphan = (total < 1e-12).ravel()
    if orphan.any():
        weights[orphan] = 0.0
        weights[orphan, np.argmin(distances[orphan], axis=1)] = 1.0
        total = weights.sum(axis=1, keepdims=True)
    return weights / total


def build(alpha: np.ndarray, unit: float, step: float = 0.032) -> tuple:
    """Сетка по силуэту: (вершины, треугольники).

    Шаг сетки — доля роста персонажа. Мельче не нужно: деформация гладкая,
    и лишние треугольники только замедляют рендер.
    """
    solid = (alpha > 16).astype(np.uint8)
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("силуэт пустой — сетку строить не из чего")
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.0015 * cv2.arcLength(contour, True)
    outline = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)

    spacing = max(step * unit, 4.0)
    height, width = alpha.shape
    ys, xs = np.mgrid[0 : height : spacing, 0 : width : spacing]
    inside = solid[ys.astype(int), xs.astype(int)] > 0
    grid = np.stack([xs[inside], ys[inside]], axis=1).astype(float)

    points = np.vstack([outline, grid]) if len(grid) else outline
    # точки, слипшиеся в комок, дают вырожденные треугольники
    keep = np.ones(len(points), dtype=bool)
    for index in range(len(outline), len(points)):
        if np.min(np.linalg.norm(outline - points[index], axis=1)) < spacing * 0.45:
            keep[index] = False
    points = points[keep]

    # вершины сажаем на целые пиксели: Subdiv2D отдаёт их обратно во float32,
    # и по исходным дробным координатам они уже не находятся
    points = np.unique(np.round(points).astype(np.int32), axis=0).astype(float)

    subdiv = cv2.Subdiv2D((0, 0, width + 1, height + 1))
    for x, y in points:
        subdiv.insert((float(x), float(y)))
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(points)}

    triangles = []
    for triangle in subdiv.getTriangleList():
        corners = triangle.reshape(3, 2)
        indices = [lookup.get((int(round(x)), int(round(y)))) for x, y in corners]
        if any(i is None for i in indices):
            continue  # вершины вспомогательного супертреугольника
        centre = corners.mean(axis=0)
        cx, cy = int(centre[0]), int(centre[1])
        if not (0 <= cy < height and 0 <= cx < width and solid[cy, cx]):
            continue  # треугольник снаружи силуэта или в вырезе
        triangles.append(indices)
    return points, np.array(triangles, dtype=np.int32)


def deform(
    points: np.ndarray, weights: np.ndarray, rig: dict, pose: dict
) -> np.ndarray:
    """Сдвинуть вершины по позе: взвешенная сумма преобразований костей."""
    moved = np.zeros_like(points)
    for index, bone in enumerate(rig["bones"]):
        corr, position = pose[bone["name"]]
        pivot = np.array(bone["rest_a"], dtype=float)
        placed = (points - pivot) @ rot(corr).T + position
        moved += weights[:, index : index + 1] * placed
    return moved


def render(
    texture: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    moved: np.ndarray,
    canvas: tuple[int, int],
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Натянуть текстуру на деформированную сетку. Вернуть RGBA."""
    height, width = canvas
    result = np.zeros((height, width, 4), dtype=np.uint8)
    shifted = moved + np.array(offset, dtype=float)

    for triangle in triangles:
        source = points[triangle].astype(np.float32)
        target = shifted[triangle].astype(np.float32)
        # чуть раздуваем треугольник от центра: иначе между соседними
        # остаётся пиксельная щель, и сетка проступает решёткой
        centre = target.mean(axis=0)
        target = centre + (target - centre) * 1.06

        x, y, w, h = cv2.boundingRect(target.astype(np.int32))
        if w <= 0 or h <= 0:
            continue
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, width), min(y + h, height)
        if x1 <= x0 or y1 <= y0:
            continue

        local = target - np.array([x0, y0], dtype=np.float32)
        matrix = cv2.getAffineTransform(source, local.astype(np.float32))
        patch = cv2.warpAffine(
            texture, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR
        )
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local.astype(np.int32), 255, cv2.LINE_AA)
        region = result[y0:y1, x0:x1]
        np.copyto(region, patch, where=(mask > 0)[..., None])
    return result
