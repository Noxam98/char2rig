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
 header {{ display:flex; gap:10px; align-items:center; padding:10px 16px;
           background:#25282e; border-bottom:1px solid #33373f;
           position:sticky; top:0; z-index:2; }}
 button {{ background:#3a6ea5; color:#fff; border:0; border-radius:4px;
           padding:6px 14px; font:inherit; cursor:pointer; }}
 button.ghost {{ background:#33373f; }}
 button:disabled {{ opacity:.45; cursor:default; }}
 #note {{ margin-left:auto; color:#9aa1ad; }}
 #stage {{ display:flex; gap:20px; justify-content:center; padding:16px; }}
 canvas {{ cursor:crosshair; background:#111318; }}
 #shot {{ max-height:78vh; border-radius:4px; background:#111318; }}
 #hint {{ padding:0 16px 16px; color:#9aa1ad; }}
 b.k {{ color:#e8b33a; }}
</style>
<header>
  <strong>{name}</strong>
  <button id="save">Сохранить</button>
  <button id="rebuild">Сохранить и пересчитать</button>
  <button id="reset" class="ghost">Сбросить</button>
  <span id="note">тяни суставы; колесо над костью — толщина</span>
</header>
<div id="stage"><canvas id="c"></canvas><img id="shot" hidden></div>
<div id="hint"></div>
<script>
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
const shot = document.getElementById('shot'), note = document.getElementById('note');
let joints = {{}}, base = {{}}, bones = [], radii = {{}}, baseRadii = {{}};
let image = new Image(), scale = 1, drag = null, hover = null;

async function boot() {{
  const skeleton = await (await fetch('skeleton.json')).json();
  bones = await (await fetch('bones.json')).json();
  base = skeleton.joints;
  baseRadii = skeleton.radii || {{}};
  joints = JSON.parse(JSON.stringify(base));
  radii = {{}};
  for (const b of bones) radii[b.name] = 1;
  try {{
    const saved = await (await fetch('skeleton.overrides.json')).json();
    for (const [name, d] of Object.entries(saved.joints || {{}}))
      if (joints[name]) {{ joints[name][0] += d[0]; joints[name][1] += d[1]; }}
    for (const [name, k] of Object.entries(saved.radius_scale || {{}}))
      radii[name] = k;
  }} catch (e) {{ /* правок ещё нет */ }}
  image.onload = () => {{
    scale = Math.min(1, (window.innerHeight - 190) / image.height);
    canvas.width = image.width * scale;
    canvas.height = image.height * scale;
    draw();
  }};
  image.src = 'source.png?' + Date.now();
}}

function boneRadius(bone, end) {{
  const r = baseRadii[bone.name];
  return (r ? r[end] : 12) * (radii[bone.name] || 1);
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  for (const b of bones) {{
    const a = joints[b.a], z = joints[b.b];
    if (!a || !z) continue;
    // капсула — то, что риг считает телом этой кости
    const on = hover === b.name;
    ctx.strokeStyle = on ? 'rgba(232,179,58,.95)' : 'rgba(90,170,240,.45)';
    ctx.lineWidth = on ? 3 : 2;
    ctx.beginPath();
    ctx.arc(a[0] * scale, a[1] * scale, boneRadius(b, 0) * scale, 0, 7);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(z[0] * scale, z[1] * scale, boneRadius(b, 1) * scale, 0, 7);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(a[0] * scale, a[1] * scale);
    ctx.lineTo(z[0] * scale, z[1] * scale);
    ctx.stroke();
  }}
  for (const [name, p] of Object.entries(joints)) {{
    const shifted = base[name][0] !== p[0] || base[name][1] !== p[1];
    ctx.fillStyle = shifted ? '#e8b33a' : '#e04a4a';
    ctx.beginPath();
    ctx.arc(p[0] * scale, p[1] * scale, 5, 0, 7);
    ctx.fill();
  }}
  const movedJoints = Object.keys(joints).filter(
    n => base[n][0] !== joints[n][0] || base[n][1] !== joints[n][1]);
  const thick = Object.entries(radii).filter(([, k]) => k !== 1)
    .map(([n, k]) => n + ' ×' + k.toFixed(2));
  document.getElementById('hint').innerHTML =
    (movedJoints.length ? 'суставы: <b class="k">' + movedJoints.join(', ') + '</b>' : 'суставы не тронуты')
    + ' &nbsp;·&nbsp; ' +
    (thick.length ? 'толщина: <b class="k">' + thick.join(', ') + '</b>' : 'толщина не тронута');
}}

function at(event) {{
  const box = canvas.getBoundingClientRect();
  return {{ x: (event.clientX - box.left) / scale,
           y: (event.clientY - box.top) / scale }};
}}

function nearestJoint(p) {{
  let best = null, bestDistance = 14 / scale;
  for (const [name, q] of Object.entries(joints)) {{
    const d = Math.hypot(q[0] - p.x, q[1] - p.y);
    if (d < bestDistance) {{ best = name; bestDistance = d; }}
  }}
  return best;
}}

function nearestBone(p) {{
  let best = null, bestDistance = Infinity;
  for (const b of bones) {{
    const a = joints[b.a], z = joints[b.b];
    if (!a || !z) continue;
    const vx = z[0] - a[0], vy = z[1] - a[1];
    const len2 = vx * vx + vy * vy || 1;
    let t = ((p.x - a[0]) * vx + (p.y - a[1]) * vy) / len2;
    t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(a[0] + vx * t - p.x, a[1] + vy * t - p.y);
    if (d < bestDistance) {{ best = b.name; bestDistance = d; }}
  }}
  return bestDistance < 60 ? best : null;
}}

canvas.onmousedown = e => {{ drag = nearestJoint(at(e)); }};
canvas.onmousemove = e => {{
  const p = at(e);
  if (drag) {{ joints[drag] = [p.x, p.y]; }}
  else {{ hover = nearestBone(p); }}
  draw();
}};
window.onmouseup = () => {{ drag = null; }};
canvas.onwheel = e => {{
  const name = hover || nearestBone(at(e));
  if (!name) return;
  e.preventDefault();
  radii[name] = Math.max(0.3, Math.min(3,
    (radii[name] || 1) * (e.deltaY < 0 ? 1.06 : 1 / 1.06)));
  draw();
}};

document.getElementById('reset').onclick = () => {{
  joints = JSON.parse(JSON.stringify(base));
  for (const b of bones) radii[b.name] = 1;
  draw();
}};

function payload() {{
  const deltas = {{}}, scales = {{}};
  for (const [name, p] of Object.entries(joints)) {{
    const dx = p[0] - base[name][0], dy = p[1] - base[name][1];
    if (dx || dy) deltas[name] = [Math.round(dx * 100) / 100,
                                  Math.round(dy * 100) / 100];
  }}
  for (const [name, k] of Object.entries(radii))
    if (k !== 1) scales[name] = Math.round(k * 1000) / 1000;
  return {{ joints: deltas, radius_scale: scales }};
}}

async function post(path) {{
  const response = await fetch(path, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(payload()),
  }});
  return response;
}}

document.getElementById('save').onclick = async () => {{
  const response = await post('save');
  note.textContent = response.ok
    ? 'сохранено — дальше: python -m char2rig recut {name} --stage pose'
    : 'не сохранилось: ' + response.status;
}};

document.getElementById('rebuild').onclick = async e => {{
  e.target.disabled = true;
  note.textContent = 'считаю…';
  const response = await post('rebuild');
  const result = response.ok ? await response.json() : null;
  note.textContent = result
    ? 'пересчитано: ' + result.triage + ', скелет внутри арта ' + result.fit_inside
    : 'не сошлось: ' + response.status;
  if (result) {{
    shot.hidden = false;
    shot.src = 'preview.gif?' + Date.now();
    await boot();
  }}
  e.target.disabled = false;
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

        self.character.write_json(self.character.skeleton_overrides, overrides)
        print(
            f"  сохранено: суставов {len(overrides['joints'])}, "
            f"толщин {len(overrides['radius_scale'])}",
            flush=True,
        )
        if action == "save":
            return self._send(200, b'{"ok":true}', CONTENT_TYPES[".json"])

        # пересчёт прямо из редактора: без него правка вслепую — сохранил,
        # ушёл в консоль, вернулся, и так по кругу
        from .cli import run_pipeline

        skeleton = self.character.read_json(self.character.skeleton)
        try:
            checks = run_pipeline(
                self.character, skeleton["template"], start="pose"
            )
        except Exception as exc:  # noqa: BLE001 — пользователю нужен текст, не трейс
            body = f"пересчёт упал: {type(exc).__name__}: {exc}".encode("utf-8")
            return self._send(500, body, "text/plain; charset=utf-8")

        status = self.character.status()
        answer = json.dumps(
            {
                "triage": status.get("triage", "?"),
                "fit_inside": checks.get("fit_inside", 0),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._send(200, answer, CONTENT_TYPES[".json"])


def serve(character: Character, port: int = 8765) -> ThreadingHTTPServer:
    """Поднять редактор. Останавливается `server.shutdown()`."""
    handler = type("Handler", (_Handler,), {"character": character})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    return server


def url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}/"
