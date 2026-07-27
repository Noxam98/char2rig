"""Правка результатов конвейера мышкой: локальный сервер плюс страница.

Почему так, а не окно cv2: opencv в ядре стоит headless-сборкой, окон у неё
нет, а тянуть ради редактора Qt — перебор. Страница же открывается на любой
машине, в том числе когда конвейер крутится на другом компьютере.

Инструменты идут в порядке конвейера — силуэт, скелет, части, — потому что
правят там, где ошибка возникла, а не там, где она вылезла. Сама страница
лежит рядом в `editor.html`; здесь только отдача данных и приём правок.

Правки сохраняются оверлеями (принцип №1 из DESIGN.md): дельты суставов в
`skeleton.overrides.json`, мазки по силуэту в `silhouette.overrides.png`,
карта частей в `masks.overrides.png`. Каждая живёт отдельно от результата и
переприменяется поверх нового прогона.
"""

from __future__ import annotations

import colorsys
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import template as templates
from .character import Character

PAGE_FILE = Path(__file__).with_name("editor.html")

ALLOWED_FILES = {
    "source.png",
    "silhouette.png",
    "silhouette.overrides.png",
    "skeleton.json",
    "skeleton.overrides.json",
    "preview.gif",
    "preview_strip.png",
}

CONTENT_TYPES = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".json": "application/json; charset=utf-8",
}

BINARY = "application/octet-stream"


def palette(count: int) -> list[list[int]]:
    """Цвета частей: шаг по кругу оттенков золотым углом.

    Соседние по списку кости получают заведомо далёкие оттенки — на карте
    частей важно видеть границу между плечом и предплечьем, а не то, что они
    родня. Яркость чередуется, чтобы цвета не сливались и через круг.
    """
    colors = []
    for index in range(count):
        hue = (index * 0.618033988749895) % 1.0
        value = 0.96 if index % 2 == 0 else 0.76
        r, g, b = colorsys.hsv_to_rgb(hue, 0.62, value)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return colors


def _template_of(character: Character) -> templates.Template:
    return templates.get(character.read_json(character.skeleton)["template"])


def owner_map(character: Character, template: templates.Template) -> np.ndarray:
    """Карта «пиксель → кость» из готовых масок: значение = номер кости + 1.

    Ноль — «части тут нет»; за силуэтом кисти делать нечего, и браузер по
    этому же нулю понимает, куда красить нельзя.
    """
    names = [bone.name for bone in template.bones]
    owner: np.ndarray | None = None
    for index, name in enumerate(names):
        path = character.masks_dir / f"{name}.png"
        if not path.exists():
            continue
        mask = character.read_mask(path) > 127
        if owner is None:
            owner = np.zeros(mask.shape, dtype=np.uint8)
        if mask.shape != owner.shape:
            continue
        owner[mask] = index + 1
    return owner if owner is not None else np.zeros((0, 0), dtype=np.uint8)


def _save_strokes(character: Character, data_url: str | None) -> int:
    """Сохранить мазки кистью по силуэту. Вернуть число закрашенных пикселей."""
    painted = _decode(data_url)
    if painted is None:
        return 0
    # холст приходит RGBA: белое — «это тело», серое — «это фон», прозрачное —
    # не трогать. Сводим к одному каналу, чтобы этап фона читал его маской
    grey = painted[..., 0].astype(np.uint8)
    grey[painted[..., 3] < 128] = 0
    if not (grey > 64).any():
        character.silhouette_overrides.unlink(missing_ok=True)
        return 0
    character.write_mask(character.silhouette_overrides, grey)
    return int((grey > 64).sum())


def _save_parts(
    character: Character, template: templates.Template, data_url: str | None
) -> int:
    """Сохранить ручную карту частей. Вернуть число перенесённых пикселей."""
    painted = _decode(data_url)
    if painted is None:
        return 0
    # номер части лежит в красном канале, «не тронуто» — в прозрачности:
    # непрозрачные пиксели PNG отдаёт байт в байт, карта доезжает целой
    index = painted[..., 0].astype(np.uint8)
    index[painted[..., 3] < 128] = 0
    index[index > len(template.bones)] = 0
    if not index.any():
        character.parts_overrides.unlink(missing_ok=True)
        character.parts_legend.unlink(missing_ok=True)
        return 0
    character.write_mask(character.parts_overrides, index)
    character.write_json(
        character.parts_legend,
        {
            # хеш арта: перегенерят картинку — и правка окажется не про неё
            "source": character.source_digest(),
            "parts": [bone.name for bone in template.bones],
        },
    )
    return int((index > 0).sum())


def _decode(data_url: str | None) -> np.ndarray | None:
    """data:image/png;base64,… → RGBA-массив."""
    if not data_url or "," not in data_url:
        return None
    import base64
    import io

    from PIL import Image

    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))


class _Handler(BaseHTTPRequestHandler):
    character: Character

    def log_message(self, *args) -> None:  # тише в консоли
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(200, body, CONTENT_TYPES[".json"])

    # --- GET -------------------------------------------------------------
    def _get_contour(self) -> None:
        character = self.character
        saved = character.contour_overrides
        if saved.exists() and "auto=1" not in self.path:
            points = character.read_json(saved).get("points", [])
        else:
            from .stages.background import extract_contour

            points = extract_contour(character.read_mask(character.silhouette))
        self._json({"points": points})

    def _get_bones(self) -> None:
        template = _template_of(self.character)
        self._json([{"name": b.name, "a": b.a, "b": b.b} for b in template.bones])

    def _get_parts(self) -> None:
        template = _template_of(self.character)
        owner = owner_map(self.character, template)
        height, width = owner.shape if owner.size else (0, 0)
        colors = palette(len(template.bones))
        self._json(
            {
                "size": [int(width), int(height)],
                "parts": [
                    {"name": bone.name, "z": bone.z, "color": colors[index]}
                    for index, bone in enumerate(template.bones)
                ],
            }
        )

    def _get_parts_bin(self) -> None:
        owner = owner_map(self.character, _template_of(self.character))
        self._send(200, owner.tobytes(), BINARY)

    def _get_parts_overrides(self) -> None:
        """Сохранённая карта правок в номерах текущего шаблона."""
        from .cli import load_parts_overrides

        character = self.character
        template = _template_of(character)
        shape = owner_map(character, template).shape
        hand, _stale = load_parts_overrides(character, template)
        if hand is None or hand.shape != shape:
            hand = np.zeros(shape, dtype=np.uint8)
        self._send(200, hand.astype(np.uint8).tobytes(), BINARY)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].lstrip("/")
        routes = {
            "contour.json": self._get_contour,
            "bones.json": self._get_bones,
            "parts.json": self._get_parts,
            "parts.bin": self._get_parts_bin,
            "parts.overrides.bin": self._get_parts_overrides,
        }
        if path in ("", "index.html"):
            page = PAGE_FILE.read_text(encoding="utf-8").replace(
                "__CHARACTER__", self.character.name
            )
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if path in routes:
            return routes[path]()
        if path in ALLOWED_FILES:
            target = self.character.root / path
            if not target.exists():
                return self._send(
                    404, "нет файла".encode("utf-8"), "text/plain; charset=utf-8"
                )
            kind = CONTENT_TYPES.get(Path(path).suffix, BINARY)
            return self._send(200, target.read_bytes(), kind)
        self._send(404, b"not found", "text/plain")

    # --- POST ------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        action = self.path.split("?")[0].lstrip("/")
        if action not in ("save", "rebuild"):
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            overrides = {
                "joints": {
                    str(name): [float(value[0]), float(value[1])]
                    for name, value in dict(payload.get("joints", {})).items()
                },
                "radius_scale": {
                    str(name): float(value)
                    for name, value in dict(payload.get("radius_scale", {})).items()
                },
            }
        except (ValueError, TypeError, IndexError) as exc:
            body = f"кривой JSON: {exc}".encode("utf-8")
            return self._send(400, body, "text/plain; charset=utf-8")

        template = _template_of(self.character)
        self.character.write_json(self.character.skeleton_overrides, overrides)
        painted = _save_strokes(self.character, payload.get("strokes"))
        moved = _save_parts(self.character, template, payload.get("parts"))
        points = payload.get("contour")
        if isinstance(points, list) and len(points) >= 3:
            # вместе с контуром пишем хеш арта: если картинку перегенерят,
            # правка окажется не про неё, и конвейер это заметит
            self.character.write_json(
                self.character.contour_overrides,
                {
                    "source": self.character.source_digest(),
                    "points": [[float(x), float(y)] for x, y in points],
                },
            )
        print(
            f"  сохранено: суставов {len(overrides['joints'])}, "
            f"толщин {len(overrides['radius_scale'])}"
            + (f", мазков {painted} px" if painted else "")
            + (f", частей {moved} px" if moved else ""),
            flush=True,
        )
        if action == "save":
            return self._send(200, b'{"ok":true}', CONTENT_TYPES[".json"])

        # пересчёт прямо из редактора: без него правка вслепую — сохранил,
        # ушёл в консоль, вернулся, и так по кругу
        from .cli import STAGES, run_pipeline

        start = str(payload.get("start", "pose"))
        if start not in STAGES:
            start = "pose"
        preview = "parts" if payload.get("preview") == "parts" else "mesh"
        try:
            checks = run_pipeline(
                self.character, template.name, start=start, preview=preview
            )
        except Exception as exc:  # noqa: BLE001 — пользователю нужен текст, не трейс
            body = f"пересчёт упал: {type(exc).__name__}: {exc}".encode("utf-8")
            return self._send(500, body, "text/plain; charset=utf-8")

        status = self.character.status()
        self._json(
            {
                "triage": status.get("triage", "?"),
                "fit_inside": checks.get("fit_inside", 0),
            }
        )


def serve(character: Character, port: int = 8765) -> ThreadingHTTPServer:
    """Поднять редактор. Останавливается `server.shutdown()`."""
    handler = type("Handler", (_Handler,), {"character": character})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/"
