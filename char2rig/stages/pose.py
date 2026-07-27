"""Этап 2: посадка шаблонного скелета на силуэт.

База — подгонка по bbox силуэта (работает всегда, моделей не требует).
Если установлен rtmlib с DWPose и детектор уверен — скелет дополнительно
подтягивается к найденным ключевым точкам подобием (масштаб + сдвиг).
Поворот и попарное уточнение суставов — фаза 1: на антропоморфах детектор
врёт достаточно часто, чтобы не давать ему ломать позу целиком.
"""

from __future__ import annotations

import numpy as np

from . import note_fallback
from ..geometry import capsule_field, pixel_grid
from ..template import Template, extent, pixels_per_unit

# rtmlib отдаёт COCO-17; «left» — сторона персонажа, во фронтальном арте
# это правая половина картинки, как и `_l` в шаблоне.
_COCO_TO_TEMPLATE = {
    5: "shoulder_l",
    6: "shoulder_r",
    7: "elbow_l",
    8: "elbow_r",
    9: "wrist_l",
    10: "wrist_r",
    11: "hip_l",
    12: "hip_r",
    13: "knee_l",
    14: "knee_r",
}

X_SCALE_LIMITS = (0.75, 1.35)

# порядок «родитель раньше ребёнка»: сустав тянет за собой свою цепочку,
# и плечо должно встать на место до того, как поедет локоть
SNAP_ORDER: tuple[tuple[int, str], ...] = (
    (5, "shoulder_l"),
    (6, "shoulder_r"),
    (11, "hip_l"),
    (12, "hip_r"),
    (7, "elbow_l"),
    (8, "elbow_r"),
    (13, "knee_l"),
    (14, "knee_r"),
    (9, "wrist_l"),
    (10, "wrist_r"),
    (15, "hock_l"),  # у COCO лодыжка, у digitigrade-ноги это скакательный сустав
    (16, "hock_r"),
)
SNAP_MIN_SCORE = 0.4


def _solve_x_scale(template: Template, target_width: float) -> float:
    """Растяжение шаблона по ширине под относительную ширину силуэта.

    target_width — ширина bbox силуэта в долях его высоты, приведённая к
    масштабу шаблона. Радиусы капсул при этом не растягиваются: персонаж
    может быть шире в размахе рук, но не толще в кости.
    """
    lo_x, lo_r, hi_x, hi_r = None, 0.0, None, 0.0
    for bone in template.bones:
        for joint, radius in ((bone.a, bone.ra), (bone.b, bone.rb)):
            x = template.joints[joint][0]
            if lo_x is None or x - radius < lo_x - lo_r:
                lo_x, lo_r = x, radius
            if hi_x is None or x + radius > hi_x + hi_r:
                hi_x, hi_r = x, radius
    span = max(hi_x - lo_x, 1e-6)
    scale = (target_width - lo_r - hi_r) / span
    return float(np.clip(scale, *X_SCALE_LIMITS))


def bbox_of(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) непрозрачной области."""
    ys, xs = np.nonzero(alpha > 16)
    if len(xs) == 0:
        raise ValueError("силуэт пустой — нечего рижить")
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(
        ys.max() - ys.min() + 1
    )


def fit_bbox(template: Template, alpha: np.ndarray) -> tuple[dict, dict]:
    """Посадить шаблон в bbox силуэта. Вернуть (суставы в пикселях, параметры)."""
    bx, by, bw, bh = bbox_of(alpha)
    _, _, ey0, ey1 = extent(template, 1.0)
    scale = bh / max(ey1 - ey0, 1e-6)
    x_scale = _solve_x_scale(template, bw / scale)
    ex0, _, ey0, _ = extent(template, x_scale)
    ox = bx - ex0 * scale
    oy = by - ey0 * scale

    joints = {
        name: (ox + (0.5 + (x - 0.5) * x_scale) * scale, oy + y * scale)
        for name, (x, y) in template.joints.items()
    }
    params = {
        "bbox": [bx, by, bw, bh],
        "scale": round(scale, 3),
        "x_scale": round(x_scale, 3),
    }
    return joints, params


def _try_drawn_pose(
    rgb: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Оценщик позы Meta, дообученный на рисунках. bbox берём из силуэта.

    Свой детектор персонажа у них тоже есть, но нам он не нужен: силуэт уже
    посчитан, и его рамка точнее любого детектора.
    """
    from ..drawnpose import estimate

    try:
        result = estimate(rgb, bbox_of(alpha))
    except Exception as exc:
        note_fallback("скелет", f"drawn_pose упал: {type(exc).__name__}: {exc}")
        return None
    if result is None:
        note_fallback("скелет", "нет весов models/drawn_pose_resnet50.pth")
    return result


_POSE_MODEL = None


def _try_dwpose(
    rgb: np.ndarray, alpha: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """(keypoints[17,2], scores[17]) от RTMPose или None, если недоступно.

    Берём `Body`, а не `Wholebody`: лицо и пальцы нам не нужны, а модель
    вдвое легче. Первые 17 точек в обоих — один и тот же COCO-порядок.
    Модель весит недёшево и грузится один раз на процесс.
    """
    global _POSE_MODEL
    try:
        from rtmlib import Body
    except ImportError:
        note_fallback("скелет", "rtmlib не установлен")
        return None
    try:
        if _POSE_MODEL is None:
            _POSE_MODEL = Body(mode="balanced", backend="onnxruntime", device="cpu")
        keypoints, scores = _POSE_MODEL(np.ascontiguousarray(rgb[..., ::-1]))
    except Exception as exc:
        note_fallback("скелет", f"rtmlib упал: {type(exc).__name__}: {exc}")
        return None
    if keypoints is None or len(keypoints) == 0:
        note_fallback("скелет", "детектор никого не нашёл")
        return None
    return np.asarray(keypoints[0], dtype=float), np.asarray(scores[0], dtype=float)


def _similarity_to(
    joints: dict[str, tuple[float, float]],
    keypoints: np.ndarray,
    scores: np.ndarray,
    min_score: float = 0.5,
) -> tuple[float, tuple[float, float]] | None:
    """Масштаб+сдвиг, приводящие шаблонные суставы к точкам детектора."""
    pairs = [
        (joints[name], keypoints[idx])
        for idx, name in _COCO_TO_TEMPLATE.items()
        if idx < len(scores) and scores[idx] >= min_score and name in joints
    ]
    if len(pairs) < 6:
        return None
    src = np.array([p[0] for p in pairs], dtype=float)
    dst = np.array([p[1] for p in pairs], dtype=float)
    src_c, dst_c = src.mean(axis=0), dst.mean(axis=0)
    src_r = np.linalg.norm(src - src_c, axis=1).mean()
    dst_r = np.linalg.norm(dst - dst_c, axis=1).mean()
    if src_r < 1e-6:
        return None
    scale = float(np.clip(dst_r / src_r, 0.6, 1.6))
    shift = (
        float(dst_c[0] - src_c[0] * scale),
        float(dst_c[1] - src_c[1] * scale),
    )
    return scale, shift


FIT_SCALE = 0.4  # совпадение с силуэтом считаем в уменьшенном разрешении


def _subtree(template: Template, joint: str) -> set[str]:
    """Суставы, которые едут вместе с этим: он сам и вся цепочка ниже."""
    moving = {joint}
    growing = True
    while growing:
        growing = False
        for bone in template.bones:
            if bone.a in moving and bone.b not in moving:
                moving.add(bone.b)
                growing = True
    return moving


def _inside(point: tuple[float, float], alpha: np.ndarray) -> bool:
    height, width = alpha.shape
    x, y = int(round(point[0])), int(round(point[1]))
    return 0 <= y < height and 0 <= x < width and alpha[y, x] > 16


def _snap_joints(
    template: Template,
    joints: dict[str, tuple[float, float]],
    keypoints: np.ndarray,
    scores: np.ndarray,
    alpha: np.ndarray | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Посадить найденные суставы точно, утащив за каждым его цепочку.

    Подгонка подобием двигает скелет целиком и потому бессильна, когда у
    персонажа другие пропорции — а именно их детектор и меряет. Здесь
    каждый найденный сустав встаёт на своё место, а его потомки едут следом,
    так что длины костей подстраиваются под конкретного персонажа.
    """
    moved = dict(joints)
    snapped = 0
    for index, name in SNAP_ORDER:
        if index >= len(scores) or scores[index] < SNAP_MIN_SCORE:
            continue
        if name not in moved:
            continue
        target = keypoints[index]
        # детектор промахивается мимо персонажа — на реальном коте он вынес
        # лодыжку наружу, и вся нога уехала за силуэт. Такую точку не берём:
        # общий IoU может даже вырасти за счёт других, и промах не заметят
        if alpha is not None and not _inside((target[0], target[1]), alpha):
            continue
        dx = float(target[0]) - moved[name][0]
        dy = float(target[1]) - moved[name][1]
        for joint in _subtree(template, name):
            x, y = moved[joint]
            moved[joint] = (x + dx, y + dy)
        snapped += 1
    return moved if snapped >= 4 else None


def _agreement(
    template: Template,
    joints: dict[str, tuple[float, float]],
    alpha: np.ndarray,
    radii: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Насколько «тело по скелету» согласуется с силуэтом: (IoU, доля внутри).

    Считаем по капсулам, а не по попаданию суставов внутрь силуэта: сустав
    может лежать в силуэте, а капсула вокруг него — торчать наружу, и потом
    ровно это вылезает щелями в стресс-тесте.

    Две меры нужны разные. **IoU** годится, чтобы сравнить кандидатов между
    собой — радиусы у них одни и те же, значит сравнение честное. А вот
    оценивать им качество посадки нельзя: уши, клочья шерсти и пушистый
    хвост капсулой не описываются в принципе, и на живом арте IoU упирается
    в потолок, который не имеет отношения к качеству рига. Дефект — это
    когда скелет считает телом то, где арта нет; его и меряет **доля
    капсул внутри силуэта**.
    """
    import cv2

    small = cv2.resize(
        alpha, None, fx=FIT_SCALE, fy=FIT_SCALE, interpolation=cv2.INTER_AREA
    )
    shape = small.shape
    grid = pixel_grid(shape)
    scaled = {k: (x * FIT_SCALE, y * FIT_SCALE) for k, (x, y) in joints.items()}
    unit = pixels_per_unit(scaled, template)
    body = np.zeros(shape, dtype=bool)
    for bone in template.bones:
        ra, rb = (
            (bone.ra * unit, bone.rb * unit)
            if radii is None
            else (
                radii[bone.name][0] * FIT_SCALE,
                radii[bone.name][1] * FIT_SCALE,
            )
        )
        body |= (
            capsule_field(shape, scaled[bone.a], scaled[bone.b], ra, rb, grid=grid) < 0
        )
    silhouette = small > 16
    union = int((body | silhouette).sum())
    inside = int((body & silhouette).sum())
    return (inside / max(union, 1), inside / max(int(body.sum()), 1))


CHAIN_SCALE = 0.25  # поиск отростка идёт в уменьшенном разрешении
CHAIN_MIN_AREA = 0.004  # мельче этой доли силуэта — не отросток, а шум
CHAIN_MIN_DEPTH = 0.03  # и торчать должен хотя бы на эту долю высоты
CHAIN_MAX_REACH = 0.4  # отросток ищем не дальше этого от корня цепочки


def _capsule_body(
    template: Template,
    joints: dict[str, tuple[float, float]],
    shape: tuple[int, int],
    skip: set[str],
) -> np.ndarray:
    """Маска «где скелет считает тело», без указанных костей."""
    grid = pixel_grid(shape)
    unit = pixels_per_unit(joints, template)
    body = np.zeros(shape, dtype=bool)
    for bone in template.bones:
        if bone.name in skip:
            continue
        body |= (
            capsule_field(
                shape,
                joints[bone.a],
                joints[bone.b],
                bone.ra * unit,
                bone.rb * unit,
                grid=grid,
            )
            < 0
        )
    return body


def _geodesic_path(
    mask: np.ndarray, start: tuple[int, int], thickness: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """Путь от начальной точки до самой дальней внутри маски.

    Волной от старта, не по прямой: хвост загибается, и прямая через него
    вышла бы наружу. Считается по маске, а не по скелетизации, — так проще
    и не зависит от scipy, которого в ядре нет.

    При возврате из дальней точки из равных по шагу соседей берётся самый
    толстый — иначе путь липнет к краю отростка, и померенная по нему
    толщина выходит втрое меньше настоящей.
    """
    import cv2

    frontier = np.zeros(mask.shape, dtype=bool)
    frontier[start[1], start[0]] = True
    if not mask[start[1], start[0]]:
        return []
    steps = np.full(mask.shape, -1, dtype=np.int32)
    steps[frontier] = 0
    kernel = np.ones((3, 3), np.uint8)
    step = 0
    while True:
        grown = (
            cv2.dilate(frontier.astype(np.uint8), kernel).astype(bool) & mask
        )
        fresh = grown & (steps < 0)
        if not fresh.any():
            break
        step += 1
        steps[fresh] = step
        frontier = fresh

    far = np.argmax(np.where(steps >= 0, steps, -1))
    y, x = np.unravel_index(far, steps.shape)
    path = [(int(x), int(y))]
    for back in range(int(steps[y, x]) - 1, -1, -1):
        top, left = max(y - 1, 0), max(x - 1, 0)
        window = steps[top : y + 2, left : x + 2]
        found = np.argwhere(window == back)
        if len(found) == 0:
            break
        if thickness is not None:
            fat = np.argmax(
                [thickness[top + int(dy), left + int(dx)] for dy, dx in found]
            )
            dy, dx = found[int(fat)]
        else:
            dy, dx = found[0]
        y, x = top + int(dy), left + int(dx)
        path.append((int(x), int(y)))
    path.reverse()
    return path


def fit_chain(
    template: Template,
    chain: str,
    joints: dict[str, tuple[float, float]],
    alpha: np.ndarray,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]] | None:
    """Посадить опциональную цепочку (хвост) на непокрытый отросток силуэта.

    Шаблон кладёт хвост туда, где он был у эталона, — а у живого кота он
    уходит куда угодно. Детекторы поз про хвосты не знают вовсе (в COCO их
    нет), зато силуэт знает: хвост — это ровно тот кусок арта, до которого
    не дотянулась ни одна капсула. Ищем такой кусок рядом с корнем цепочки
    и раскладываем по нему кости.

    Заодно меряем толщину прямо по найденному отростку и возвращаем радиусы:
    общий замер поперёк кости на хвосте врёт, потому что кость — это хорда, а
    хвост загнут, и луч уходит наружу раньше времени. Тонкая капсула хвоста
    потом отдаёт его пиксели жирной капсуле бедра, и хвост улетает с ногой.
    """
    import cv2

    bones = [template.bone(name) for name in template.chains.get(chain, ())]
    # цепочка должна быть именно цепочкой: кость за костью. «digitigrade-ноги»
    # из шаблона — это две отдельные стопы, их так сажать нельзя
    if not bones or any(
        child.parent != parent.name for parent, child in zip(bones, bones[1:])
    ):
        return None
    root = bones[0].a

    small = cv2.resize(
        alpha, None, fx=CHAIN_SCALE, fy=CHAIN_SCALE, interpolation=cv2.INTER_AREA
    )
    shape = small.shape
    scaled = {k: (x * CHAIN_SCALE, y * CHAIN_SCALE) for k, (x, y) in joints.items()}
    body = _capsule_body(template, scaled, shape, skip={b.name for b in bones})
    unclaimed = (small > 16) & ~body

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        unclaimed.astype(np.uint8), 8
    )
    if count <= 1:
        return None
    silhouette_area = max(int((small > 16).sum()), 1)
    # насколько глубоко кусок торчит из тела: у хвоста это десятки пикселей,
    # у кромки вдоль руки — единицы. Именно это отличает отросток от каймы,
    # а близость к корню — нет: кайма проходит вплотную к нему
    depth = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 3)
    rx, ry = scaled[root]
    best, best_depth = None, CHAIN_MIN_DEPTH * shape[0]
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < CHAIN_MIN_AREA * silhouette_area:
            continue
        piece = labels == index
        ys, xs = np.nonzero(piece)
        if float(np.min(np.hypot(xs - rx, ys - ry))) > CHAIN_MAX_REACH * shape[0]:
            continue
        reach = float(depth[piece].max())
        if reach > best_depth:
            best, best_depth = piece, reach
    if best is None:
        return None

    thickness = cv2.distanceTransform((small > 16).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(best)
    start_index = int(np.argmin(np.hypot(xs - rx, ys - ry)))
    path = _geodesic_path(
        best, (int(xs[start_index]), int(ys[start_index])), thickness
    )
    if len(path) < len(bones) * 2:
        return None

    moved = dict(joints)
    positions = [path[0]] + [
        path[min(len(path) - 1, round((i + 1) * (len(path) - 1) / len(bones)))]
        for i in range(len(bones))
    ]
    for bone, point in zip(bones, positions[1:]):
        moved[bone.b] = (point[0] / CHAIN_SCALE, point[1] / CHAIN_SCALE)
    moved[root] = (positions[0][0] / CHAIN_SCALE, positions[0][1] / CHAIN_SCALE)

    # толщину берём не в самой точке, а по окрестности вдоль пути: концы
    # пути лежат на границе отростка, там расстояние до фона почти нулевое
    window = max(len(path) // (2 * len(bones)), 2)
    indices = [0] + [
        min(len(path) - 1, round((i + 1) * (len(path) - 1) / len(bones)))
        for i in range(len(bones))
    ]
    at = []
    for index in indices:
        near = path[max(index - window, 0) : index + window + 1]
        at.append(
            max(float(thickness[p[1], p[0]]) for p in near) / CHAIN_SCALE
            if near
            else 4.0
        )
    radii = {
        bone.name: (max(at[i], 2.0), max(at[i + 1], 2.0))
        for i, bone in enumerate(bones)
    }
    return moved, radii


RADIUS_LIMITS = (0.55, 1.9)  # во сколько раз кость может отличаться от шаблона


def measure_radii(
    template: Template,
    joints: dict[str, tuple[float, float]],
    alpha: np.ndarray,
) -> dict[str, tuple[float, float]]:
    """Померить толщину персонажа вдоль каждой кости.

    Шаблон задаёт пропорции эталона, а живой кот бывает тощим или пузатым, и
    капсула из шаблона либо не покрывает арт, либо вылезает наружу — и то и
    другое портит и разбиение на части, и проверки. Расстояние до фона в
    точке оси — это ровно половина местной толщины, её и берём, зажимая
    относительно шаблона: рядом с плечом внутри торса замер завышен.
    """
    height, width = alpha.shape
    unit = pixels_per_unit(joints, template)
    solid = alpha > 16
    reach = max(int(0.35 * unit), 4)
    steps = np.arange(1, reach + 1, dtype=np.float32)

    def half_width(x: float, y: float, nx: float, ny: float) -> float:
        """Докуда добивает луч поперёк кости, прежде чем выйти из силуэта."""
        xs = np.clip(np.round(x + nx * steps).astype(int), 0, width - 1)
        ys = np.clip(np.round(y + ny * steps).astype(int), 0, height - 1)
        outside = np.flatnonzero(~solid[ys, xs])
        return float(steps[outside[0]]) if len(outside) else float(reach)

    def measure(bone, position: float, fallback: float) -> float:
        ax, ay = joints[bone.a]
        bx, by = joints[bone.b]
        dx, dy = bx - ax, by - ay
        length = float(np.hypot(dx, dy))
        if length < 1e-3:
            return fallback
        nx, ny = -dy / length, dx / length
        samples = []
        for shift in (-0.08, 0.0, 0.08):
            t = min(max(position + shift, 0.0), 1.0)
            x, y = ax + dx * t, ay + dy * t
            if not (0 <= int(y) < height and 0 <= int(x) < width and solid[int(y), int(x)]):
                continue
            # меньшая из сторон: рука, прижатая к телу, наружу выходит сразу,
            # а внутрь луч уйдёт через весь торс и намерит его ширину
            samples.append(
                min(half_width(x, y, nx, ny), half_width(x, y, -nx, -ny))
            )
        if not samples:
            return fallback
        measured = float(np.median(samples))
        low, high = RADIUS_LIMITS
        return float(np.clip(measured, fallback * low, fallback * high))

    return {
        bone.name: (
            measure(bone, 0.2, bone.ra * unit),
            measure(bone, 0.8, bone.rb * unit),
        )
        for bone in template.bones
    }


def run(
    template: Template,
    rgba: np.ndarray,
    alpha: np.ndarray,
    use_ml: bool = True,
    overrides: dict | None = None,
) -> tuple[dict, dict, str, bool, dict]:
    """Вернуть (суставы, радиусы костей в px, метод, фоллбек ли, параметры)."""
    joints, params = fit_bbox(template, alpha)
    method, fallback = "bbox_fit", True

    if use_ml:
        # Детекторы обучены на других доменах — на антропоморфе любой из них
        # может уехать. Поэтому не верим ни одному на слово: каждый даёт
        # кандидата, а выбирает силуэт (торчащая наружу капсула потом
        # вылезает щелями). Посадка по bbox участвует наравне с моделями.
        best = _agreement(template, joints, alpha)[0]
        scores = {"bbox_fit": round(best, 3)}
        for name, detector in (
            ("dwpose", _try_dwpose),
            ("drawn_pose", _try_drawn_pose),
        ):
            detected = detector(rgba[..., :3], alpha)
            if detected is None:
                continue
            variants: dict[str, dict[str, tuple[float, float]]] = {}
            fitted = _similarity_to(joints, *detected)
            if fitted is not None:
                scale, (dx, dy) = fitted
                variants[f"{name}_similarity"] = {
                    key: (x * scale + dx, y * scale + dy)
                    for key, (x, y) in joints.items()
                }
            snapped = _snap_joints(template, joints, *detected, alpha=alpha)
            if snapped is not None:
                variants[f"{name}_snap"] = snapped
            if not variants:
                note_fallback("скелет", f"{name}: мало уверенных точек")
                continue
            for label, moved in variants.items():
                score = _agreement(template, moved, alpha)[0]
                scores[label] = round(score, 3)
                if score > best:
                    joints, best = moved, score
                    method, fallback = f"bbox_fit+{label}", False
                    params["pose_conf"] = round(float(np.mean(detected[1])), 3)
        params["candidates"] = scores

    # опциональные цепочки садятся последними: они цепляются за корень,
    # который до этого мог переехать вслед за детектором
    chain_radii: dict[str, tuple[float, float]] = {}
    for chain in template.chains:
        result = fit_chain(template, chain, joints, alpha)
        if result is None:
            continue
        fitted, measured = result
        if _agreement(template, fitted, alpha)[0] > _agreement(template, joints, alpha)[0]:
            joints = fitted
            chain_radii.update(measured)
            params.setdefault("chains", []).append(chain)

    # принцип №1: ручные правки — оверлеи поверх автоматики
    for name, delta in (overrides or {}).get("joints", {}).items():
        if name in joints:
            joints[name] = (joints[name][0] + delta[0], joints[name][1] + delta[1])

    # толщину меряем в самом конце, когда суставы уже на своих местах;
    # у посаженных цепочек она уже померена по самому отростку
    radii = measure_radii(template, joints, alpha)
    radii.update(chain_radii)
    # толщину тоже можно поправить руками: замер по силуэту врёт там, где
    # части соприкасаются, и человек видит это быстрее любой эвристики
    for name, factor in (overrides or {}).get("radius_scale", {}).items():
        if name in radii:
            radii[name] = (radii[name][0] * factor, radii[name][1] * factor)
    # итоговое совпадение с силуэтом считаем всегда: это и есть оценка
    # качества посадки, по которой персонажа красит триаж
    fit_iou, fit_inside = _agreement(template, joints, alpha, radii)
    params["fit_iou"] = round(fit_iou, 3)
    params["fit_inside"] = round(fit_inside, 3)
    return joints, radii, method, fallback, params
