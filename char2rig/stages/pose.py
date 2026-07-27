"""Этап 2: посадка шаблонного скелета на силуэт.

База — подгонка по bbox силуэта (работает всегда, моделей не требует).
Если установлен rtmlib с DWPose и детектор уверен — скелет дополнительно
подтягивается к найденным ключевым точкам подобием (масштаб + сдвиг).
Поворот и попарное уточнение суставов — фаза 1: на антропоморфах детектор
врёт достаточно часто, чтобы не давать ему ломать позу целиком.
"""

from __future__ import annotations

import numpy as np

from ..template import Template, extent

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


def _try_dwpose(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(keypoints[17,2], scores[17]) от DWPose или None, если недоступно."""
    try:
        from rtmlib import Wholebody
    except ImportError:
        return None
    try:
        model = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
        keypoints, scores = model(rgb[..., ::-1])  # rtmlib ждёт BGR
    except Exception:
        return None
    if keypoints is None or len(keypoints) == 0:
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


def run(
    template: Template,
    rgba: np.ndarray,
    alpha: np.ndarray,
    use_ml: bool = True,
    overrides: dict | None = None,
) -> tuple[dict, str, bool, dict]:
    """Вернуть (суставы, метод, фоллбек ли, параметры посадки)."""
    joints, params = fit_bbox(template, alpha)
    method, fallback = "bbox_fit", True

    if use_ml:
        detected = _try_dwpose(rgba[..., :3])
        if detected is not None:
            fitted = _similarity_to(joints, *detected)
            if fitted is not None:
                scale, (dx, dy) = fitted
                joints = {
                    name: (x * scale + dx, y * scale + dy)
                    for name, (x, y) in joints.items()
                }
                params["dwpose_scale"] = round(scale, 3)
                params["dwpose_conf"] = round(float(np.mean(detected[1])), 3)
                method, fallback = "bbox_fit+dwpose", False

    # принцип №1: ручные правки — оверлеи поверх автоматики
    for name, delta in (overrides or {}).get("joints", {}).items():
        if name in joints:
            joints[name] = (joints[name][0] + delta[0], joints[name][1] + delta[1])

    return joints, method, fallback, params
