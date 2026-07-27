"""char2rig — конвейер «картинка персонажа → анимируемый 2D-персонаж»."""

import os
from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    """Куда складываются веса моделей.

    От корня пакета, а не от текущей папки: иначе конвейер, запущенный из
    соседнего каталога, молча не находит модель и уходит в фоллбек.
    Переопределяется переменной окружения CHAR2RIG_MODELS.
    """
    override = os.environ.get("CHAR2RIG_MODELS")
    return Path(override) if override else ROOT / "models"
