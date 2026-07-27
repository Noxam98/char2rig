"""Оценщик позы, дообученный на рисунках (Meta AnimatedDrawings).

Зачем ещё один. DWPose обучен на фотографиях людей, и на нашем коте он
проиграл тупой посадке по bbox. У Meta для проекта AnimatedDrawings есть
модель, дообученная на 178 тысячах любительских рисунков — домен куда
ближе к нашему, чем фотографии.

Модель отдаётся в виде TorchServe-архива с весами mmpose, но сама сеть —
классический SimpleBaseline: ResNet-50, три деконв-слоя, голова 1x1 на 17
точек COCO, вход 192x256, тепловая карта 48x64. Всё это поднимается на
голом torch; ставить ради него стек mmcv/mmpose (который на Python 3.13 не
собирается) незачем.

Веса:

    curl -L -o models/drawn_humanoid_pose_estimator.mar \\
      https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_pose_estimator.mar

`.mar` — обычный zip, внутри `best_AP_epoch_72.pth`; распаковать в
`models/drawn_pose_resnet50.pth`. Лицензия MIT.
"""

from __future__ import annotations

import zipfile

import cv2
import numpy as np

from . import models_dir

WEIGHTS = models_dir() / "drawn_pose_resnet50.pth"
ARCHIVE = models_dir() / "drawn_humanoid_pose_estimator.mar"

INPUT_SIZE = (192, 256)  # ширина, высота
HEATMAP_SIZE = (48, 64)
PADDING = 1.25  # запас вокруг bbox, как в обучении mmpose
PIXEL_STD = 200.0  # мера масштаба в mmpose: scale = размер / 200

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL = None


def _build(joints: int = 17):
    """SimpleBaseline: ResNet-50 + три деконв-слоя + голова 1x1."""
    import torch
    from torch import nn
    from torchvision.models import resnet50

    class SimpleBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = resnet50(weights=None)
            layers: list[nn.Module] = []
            channels = 2048
            for _ in range(3):
                layers += [
                    nn.ConvTranspose2d(channels, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ]
                channels = 256
            self.deconv_layers = nn.Sequential(*layers)
            self.final_layer = nn.Conv2d(256, joints, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            b = self.backbone
            x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
            x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
            return self.final_layer(self.deconv_layers(x))

    return SimpleBaseline()


def _unpack_weights() -> bool:
    """Достать .pth из .mar, если рядом лежит только архив."""
    if WEIGHTS.exists():
        return True
    if not ARCHIVE.exists():
        return False
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = [n for n in archive.namelist() if n.endswith(".pth")]
        if not names:
            return False
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(names[0]) as src, open(WEIGHTS, "wb") as dst:
            dst.write(src.read())
    return True


def load():
    """Загрузить модель один раз на процесс. None, если весов нет."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not _unpack_weights():
        return None
    import torch

    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model = _build()
    renamed = {
        key.replace("keypoint_head.", ""): value for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    # у torchvision-resnet есть fc, которого в чекпойнте нет — это нормально
    if [k for k in missing if not k.startswith("backbone.fc")]:
        raise RuntimeError(f"веса не подошли: не хватает {missing[:4]}")
    if unexpected:
        raise RuntimeError(f"веса не подошли: лишние {unexpected[:4]}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = model.eval().to(device)
    return _MODEL


def _box_to_center_scale(
    bbox: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """bbox → центр и масштаб в мере mmpose, с приведением к пропорции входа."""
    x, y, w, h = bbox
    center = np.array([x + w / 2, y + h / 2], dtype=np.float32)
    aspect = INPUT_SIZE[0] / INPUT_SIZE[1]
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], dtype=np.float32) / PIXEL_STD * PADDING


def estimate(
    rgb: np.ndarray, bbox: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray] | None:
    """(точки[17,2] в пикселях исходника, уверенности[17]) или None."""
    model = load()
    if model is None:
        return None
    import torch

    center, scale = _box_to_center_scale(bbox)
    span = scale * PIXEL_STD  # ширина и высота вырезаемой области
    sx, sy = INPUT_SIZE[0] / span[0], INPUT_SIZE[1] / span[1]
    matrix = np.array(
        [
            [sx, 0.0, INPUT_SIZE[0] / 2 - center[0] * sx],
            [0.0, sy, INPUT_SIZE[1] / 2 - center[1] * sy],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(rgb, matrix, INPUT_SIZE, flags=cv2.INTER_LINEAR)
    tensor = ((crop.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)

    device = next(model.parameters()).device
    with torch.inference_mode():
        heatmaps = model(torch.from_numpy(tensor)[None].to(device))
    heatmaps = heatmaps[0].float().cpu().numpy()

    joints, height, width = heatmaps.shape
    flat = heatmaps.reshape(joints, -1)
    best = flat.argmax(axis=1)
    scores = flat.max(axis=1)
    coords = np.stack([best % width, best // width], axis=1).astype(np.float32)

    # полупиксельная поправка по соседям — стандартный трюк top-down моделей
    for j in range(joints):
        px, py = int(coords[j, 0]), int(coords[j, 1])
        if 0 < px < width - 1:
            step = heatmaps[j, py, px + 1] - heatmaps[j, py, px - 1]
            coords[j, 0] += 0.25 * np.sign(step)
        if 0 < py < height - 1:
            step = heatmaps[j, py + 1, px] - heatmaps[j, py - 1, px]
            coords[j, 1] += 0.25 * np.sign(step)

    coords[:, 0] = coords[:, 0] * span[0] / width + center[0] - span[0] / 2
    coords[:, 1] = coords[:, 1] * span[1] / height + center[1] - span[1] / 2
    return coords, scores
