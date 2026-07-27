"""Этапы конвейера. Каждый умеет работать без нейросетей (фоллбек)."""

from __future__ import annotations


_SAID: set[tuple[str, str]] = set()


def note_fallback(stage: str, reason: str) -> None:
    """Сказать вслух, почему этап ушёл в фоллбек — по разу на причину.

    Молчаливый `except: return None` превращает «модель не скачалась» и
    «модели тут не должно быть» в одно и то же — а это разные новости.
    """
    if (stage, reason) in _SAID:
        return
    _SAID.add((stage, reason))
    print(f"  [{stage}] без модели: {reason}", flush=True)
