"""Правка суставов мышкой: локальный сервер плюс страница в браузере.

Почему так, а не окно cv2: opencv в ядре стоит headless-сборкой, окон у неё
нет, а тянуть ради редактора Qt — перебор. Страница же открывается на любой
машине, в том числе когда конвейер крутится на другом компьютере.

Правки сохраняются в `skeleton.overrides.json` дельтами к автоматическим
координатам — принцип №1 из DESIGN.md: правка живёт отдельно от результата и
переприменяется поверх нового прогона. Дальше::

    python -m char2rig recut <имя> --stage pose
"""

from __future__ import annotations

import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import template as templates
from .character import Character

ALLOWED_FILES = {
    "source.png",
    "silhouette.png",
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

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>char2rig — {name}</title>
<style>
 body {{ margin:0; background:#1d1f23; color:#d8dbe0;
        font:14px/1.5 system-ui, sans-serif; }}
 header {{ display:flex; gap:16px; align-items:center; padding:10px 16px;
           background:#25282e; border-bottom:1px solid #33373f; }}
 button {{ background:#3a6ea5; color:#fff; border:0; border-radius:4px;
           padding:6px 14px; font:inherit; cursor:pointer; }}
 button.ghost {{ background:#33373f; }}
 button:disabled {{ opacity:.45; cursor:default; }}
 #note {{ margin-left:auto; color:#9aa1ad; }}
 canvas {{ display:block; margin:16px auto; cursor:grab;
           background:#111318 url() center/contain no-repeat; }}
 code {{ background:#2b2f36; padding:1px 5px; border-radius:3px; }}
 #moved {{ padding:0 16px 16px; color:#9aa1ad; }}
</style>
<header>
  <strong>{name}</strong>
  <button id="save">Сохранить правки</button>
  <button id="reset" class="ghost">Сбросить</button>
  <span id="note">тяни суставы мышкой</span>
</header>
<canvas id="c"></canvas>
<div id="moved"></div>
<script>
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
let joints = {{}}, base = {{}}, bones = [], image = new Image();
let scale = 1, drag = null;

async function boot() {{
  const skeleton = await (await fetch('skeleton.json')).json();
  bones = await (await fetch('bones.json')).json();
  base = skeleton.joints;
  joints = JSON.parse(JSON.stringify(base));
  try {{
    const saved = await (await fetch('skeleton.overrides.json')).json();
    for (const [name, d] of Object.entries(saved.joints || {{}}))
      if (joints[name]) {{ joints[name][0] += d[0]; joints[name][1] += d[1]; }}
  }} catch (e) {{ /* правок ещё нет */ }}
  image.onload = () => {{
    scale = Math.min(1, (window.innerHeight - 160) / image.height);
    canvas.width = image.width * scale;
    canvas.height = image.height * scale;
    draw();
  }};
  image.src = 'source.png?' + Date.now();
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(60,140,220,.9)';
  for (const b of bones) {{
    const a = joints[b.a], z = joints[b.b];
    if (!a || !z) continue;
    ctx.beginPath();
    ctx.moveTo(a[0] * scale, a[1] * scale);
    ctx.lineTo(z[0] * scale, z[1] * scale);
    ctx.stroke();
  }}
  for (const [name, p] of Object.entries(joints)) {{
    const shifted = base[name][0] !== p[0] || base[name][1] !== p[1];
    ctx.fillStyle = shifted ? '#e8b33a' : '#e04a4a';
    ctx.beginPath();
    ctx.arc(p[0] * scale, p[1] * scale, 6, 0, 7);
    ctx.fill();
  }}
  const moved = Object.keys(joints).filter(
    n => base[n][0] !== joints[n][0] || base[n][1] !== joints[n][1]);
  document.getElementById('moved').textContent = moved.length
    ? 'сдвинуто: ' + moved.join(', ')
    : 'ничего не сдвинуто';
}}

function pick(event) {{
  const box = canvas.getBoundingClientRect();
  const x = (event.clientX - box.left) / scale, y = (event.clientY - box.top) / scale;
  let best = null, bestDistance = 14 / scale;
  for (const [name, p] of Object.entries(joints)) {{
    const d = Math.hypot(p[0] - x, p[1] - y);
    if (d < bestDistance) {{ best = name; bestDistance = d; }}
  }}
  return {{ name: best, x, y }};
}}

canvas.onmousedown = e => {{ const hit = pick(e); if (hit.name) drag = hit.name; }};
canvas.onmousemove = e => {{
  if (!drag) return;
  const hit = pick(e);
  joints[drag] = [hit.x, hit.y];
  draw();
}};
window.onmouseup = () => {{ drag = null; }};

document.getElementById('reset').onclick = () => {{
  joints = JSON.parse(JSON.stringify(base)); draw();
}};

document.getElementById('save').onclick = async () => {{
  const deltas = {{}};
  for (const [name, p] of Object.entries(joints)) {{
    const dx = p[0] - base[name][0], dy = p[1] - base[name][1];
    if (dx || dy) deltas[name] = [Math.round(dx * 100) / 100,
                                  Math.round(dy * 100) / 100];
  }}
  const response = await fetch('save', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ joints: deltas }}),
  }});
  const note = document.getElementById('note');
  note.textContent = response.ok
    ? 'сохранено — теперь: python -m char2rig recut {name} --stage pose'
    : 'не сохранилось: ' + response.status;
}};

boot();
</script>
"""


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

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].lstrip("/")
        if path in ("", "index.html"):
            page = PAGE.format(name=self.character.name)
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if path == "bones.json":
            skeleton = self.character.read_json(self.character.skeleton)
            template = templates.get(skeleton["template"])
            bones = [{"name": b.name, "a": b.a, "b": b.b} for b in template.bones]
            body = json.dumps(bones, ensure_ascii=False).encode("utf-8")
            return self._send(200, body, CONTENT_TYPES[".json"])
        if path in ALLOWED_FILES:
            target = self.character.root / path
            if not target.exists():
                return self._send(
                    404, "нет файла".encode("utf-8"), "text/plain; charset=utf-8"
                )
            kind = CONTENT_TYPES.get(Path(path).suffix, "application/octet-stream")
            return self._send(200, target.read_bytes(), kind)
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0].lstrip("/") != "save":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            deltas = {
                str(name): [float(value[0]), float(value[1])]
                for name, value in dict(payload.get("joints", {})).items()
            }
        except (ValueError, TypeError, IndexError) as exc:
            body = f"кривой JSON: {exc}".encode("utf-8")
            return self._send(400, body, "text/plain; charset=utf-8")

        self.character.write_json(
            self.character.skeleton_overrides, {"joints": deltas}
        )
        print(
            f"  сохранено {len(deltas)} правок → "
            f"{self.character.skeleton_overrides}",
            flush=True,
        )
        self._send(200, b'{"ok":true}', CONTENT_TYPES[".json"])


def serve(character: Character, port: int = 8765) -> ThreadingHTTPServer:
    """Поднять редактор. Останавливается `server.shutdown()`."""
    handler = type("Handler", (_Handler,), {"character": character})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/"
