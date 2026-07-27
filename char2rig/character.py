"""Папка персонажа: пути, чтение/запись артефактов, status.json.

Раскладка (см. DESIGN.md → «Хранилище»)::

    characters/<name>/
      source.png              исходный арт
      silhouette.png          альфа после удаления фона
      skeleton.json           авто-скелет
      skeleton.overrides.json ручные дельты суставов (фаза 1)
      masks/<part>.png        маска на часть
      masks.overrides.png     ручной перенос пикселей между частями
      masks.overrides.json    легенда к нему: номер → имя кости, хеш арта
      layers/<part>.png       RGBA-слой части
      rig.json                риг
      preview.gif             стресс-тест
      preview_strip.png       та же анимация лентой (видно прямо в git)
      status.json             какие этапы прошли, чем и с каким результатом
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CHARACTERS_DIR = Path("characters")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Character:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @classmethod
    def open(cls, name: str, base: Path | str = CHARACTERS_DIR) -> "Character":
        char = cls(Path(base) / name)
        char.ensure_dirs()
        return char

    # --- пути ------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.root.name

    @property
    def source(self) -> Path:
        return self.root / "source.png"

    @property
    def silhouette(self) -> Path:
        return self.root / "silhouette.png"

    @property
    def silhouette_overrides(self) -> Path:
        """Мазки кистью по силуэту: 255 — добавить, 128 — убрать, 0 — не трогать."""
        return self.root / "silhouette.overrides.png"

    @property
    def contour_overrides(self) -> Path:
        """Поправленный руками контур силуэта: многоугольник плюс хеш арта."""
        return self.root / "silhouette.contour.json"

    def source_digest(self) -> str:
        """Хеш исходника: по нему видно, что арт перегенерили и правки устарели."""
        import hashlib

        if not self.source.exists():
            return ""
        return hashlib.sha256(self.source.read_bytes()).hexdigest()[:16]

    @property
    def skeleton(self) -> Path:
        return self.root / "skeleton.json"

    @property
    def skeleton_overrides(self) -> Path:
        return self.root / "skeleton.overrides.json"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def parts_overrides(self) -> Path:
        """Ручная карта частей: значение = номер кости + 1, 0 — не трогать."""
        return self.root / "masks.overrides.png"

    @property
    def parts_legend(self) -> Path:
        """К карте частей: какому имени соответствует номер, плюс хеш арта."""
        return self.root / "masks.overrides.json"

    @property
    def layers_dir(self) -> Path:
        return self.root / "layers"

    @property
    def rig(self) -> Path:
        return self.root / "rig.json"

    @property
    def preview(self) -> Path:
        return self.root / "preview.gif"

    @property
    def preview_strip(self) -> Path:
        return self.root / "preview_strip.png"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    def ensure_dirs(self) -> None:
        for path in (self.root, self.masks_dir, self.layers_dir):
            path.mkdir(parents=True, exist_ok=True)

    # --- изображения -----------------------------------------------------
    def read_rgba(self, path: Path) -> np.ndarray:
        """RGBA uint8 (h, w, 4)."""
        return np.array(Image.open(path).convert("RGBA"))

    def write_rgba(self, path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.astype(np.uint8), mode="RGBA").save(path)

    def read_mask(self, path: Path) -> np.ndarray:
        """Одноканальная маска uint8."""
        return np.array(Image.open(path).convert("L"))

    def write_mask(self, path: Path, mask: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if mask.dtype != np.uint8:
            mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(mask, mode="L").save(path)

    # --- json ------------------------------------------------------------
    def read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)
        path.write_text(text + "\n", encoding="utf-8")

    # --- status.json -----------------------------------------------------
    def status(self) -> dict[str, Any]:
        if self.status_path.exists():
            return self.read_json(self.status_path)
        return {"name": self.name, "created": _now(), "stages": {}, "checks": {}}

    def record_stage(
        self,
        stage: str,
        method: str,
        fallback: bool,
        **extra: Any,
    ) -> None:
        """Отметить прохождение этапа.

        method — чем реально сделано ('sam2' / 'capsules' / ...), fallback —
        деградировал ли этап до варианта без моделей. По этим полям потом
        видно, «настоящий» ли это прогон.
        """
        status = self.status()
        status["stages"][stage] = {
            "method": method,
            "fallback": fallback,
            "at": _now(),
            **extra,
        }
        status["updated"] = _now()
        self.write_json(self.status_path, status)

    def record_checks(self, checks: dict[str, Any], triage: str) -> None:
        status = self.status()
        status["checks"] = checks  # результат одного стресс-теста, не накопление
        status["triage"] = triage
        status["updated"] = _now()
        self.write_json(self.status_path, status)

    def used_fallbacks(self) -> list[str]:
        return [
            name
            for name, info in self.status().get("stages", {}).items()
            if info.get("fallback")
        ]
