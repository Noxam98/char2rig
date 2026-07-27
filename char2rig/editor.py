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
  <button id="tab-contour" class="ghost">1 · Контур</button>
  <button id="tab-silhouette" class="ghost">1б · Кисть</button>
  <button id="tab-skeleton">2 · Скелет</button>
  <button id="rebuild">Пересчитать</button>
  <button id="reset" class="ghost">Сбросить этап</button>
  <button id="fit" class="ghost">Вписать</button>
  <span id="note">этапы правятся сверху вниз: сначала силуэт, потом скелет</span>
</header>
<div id="stage"><canvas id="c"></canvas><img id="shot" hidden></div>
<div id="hint"></div>
<script>
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
const shot = document.getElementById('shot'), note = document.getElementById('note');
let joints = {{}}, base = {{}}, bones = [], radii = {{}}, baseRadii = {{}};
let image = new Image(), scale = 1, drag = null, hover = null;
// вид: масштаб и сдвиг. Ctrl+колесо — зум под курсором, средняя кнопка — тащить
let baseScale = 1, zoom = 1, panX = 0, panY = 0, panning = null;
let mode = 'skeleton', brush = 22, painting = 0;
let strokes = null, strokeCtx = null, silhouette = new Image();
let contour = [], contourNode = null, contourHover = -1;

const ZOOM = ' · колесо — зум, средняя кнопка — тащить, 0 — вписать';
const NOTES = {{
  contour: 'узлы тянутся; двойной клик по линии — добавить, правый клик — удалить'
           + ZOOM,
  silhouette: 'левой — дорисовать тело, правой — стереть; колесо — кисть,'
              + ' Ctrl+колесо — зум',
  skeleton: 'тяни суставы; колесо над костью — толщина, Ctrl+колесо — зум',
}};

function setMode(next) {{
  mode = next;
  for (const tab of ['contour', 'silhouette', 'skeleton'])
    document.getElementById('tab-' + tab).className = tab === next ? '' : 'ghost';
  note.textContent = NOTES[next];
  draw();
}}

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
    canvas.height = Math.max(320, window.innerHeight - 190);
    canvas.width = Math.max(
      320, Math.min(image.width * canvas.height / image.height, window.innerWidth - 380));
    fitView();
    if (!strokes) {{
      strokes = document.createElement('canvas');
      strokes.width = image.width;
      strokes.height = image.height;
      strokeCtx = strokes.getContext('2d', {{ willReadFrequently: true }});
      const old = new Image();
      old.onload = () => {{ strokeCtx.drawImage(old, 0, 0); draw(); }};
      old.onerror = () => draw();
      old.src = 'silhouette.overrides.png?' + Date.now();
    }}
    draw();
  }};
  image.src = 'source.png?' + Date.now();
  silhouette.onload = draw;
  silhouette.src = 'silhouette.png?' + Date.now();

  const fetched = await (await fetch('contour.json?' + Date.now())).json();
  contour = fetched.points || [];
  draw();
}}

function fitView() {{
  baseScale = Math.min(canvas.width / image.width, canvas.height / image.height);
  zoom = 1;
  scale = baseScale;
  panX = (canvas.width - image.width * scale) / 2;
  panY = (canvas.height - image.height * scale) / 2;
  draw();
}}

function boneRadius(bone, end) {{
  const r = baseRadii[bone.name];
  return (r ? r[end] : 12) * (radii[bone.name] || 1);
}}

function draw() {{
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // сдвиг вида делаем трансформом, а масштаб остаётся в `scale`: так все
  // размеры узлов и линий остаются постоянными на экране при любом зуме
  ctx.setTransform(1, 0, 0, 1, panX, panY);
  ctx.drawImage(image, 0, 0, image.width * scale, image.height * scale);

  if (mode === 'contour') {{
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(90,210,140,.95)';
    ctx.beginPath();
    contour.forEach((p, i) => {{
      const x = p[0] * scale, y = p[1] * scale;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }});
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = 'rgba(90,210,140,.12)';
    ctx.fill();
    contour.forEach((p, i) => {{
      ctx.fillStyle = i === contourHover ? '#e8b33a' : '#3ad48a';
      ctx.beginPath();
      ctx.arc(p[0] * scale, p[1] * scale, i === contourHover ? 6 : 4, 0, 7);
      ctx.fill();
    }});
    document.getElementById('hint').textContent =
      'узлов: ' + contour.length + ' · контур заменяет автоматику целиком';
    return;
  }}

  if (mode === 'silhouette') {{
    // что конвейер считает телом — поверх арта, чтобы промахи было видно
    ctx.globalAlpha = 0.45;
    if (silhouette.complete && silhouette.naturalWidth)
      ctx.drawImage(silhouette, 0, 0, image.width * scale, image.height * scale);
    ctx.globalAlpha = 0.7;
    if (strokes)
      ctx.drawImage(strokes, 0, 0, image.width * scale, image.height * scale);
    ctx.globalAlpha = 1;
    document.getElementById('hint').textContent =
      'полупрозрачным показан текущий силуэт; кисть ' + Math.round(brush) + ' px';
    return;
  }}

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
  return {{ x: (event.clientX - box.left - panX) / scale,
           y: (event.clientY - box.top - panY) / scale }};
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

function paintAt(p) {{
  // 255 — «это тело», 128 — «это фон»; так правка остаётся отдельным слоем
  strokeCtx.globalCompositeOperation = 'source-over';
  strokeCtx.fillStyle = painting === 1 ? 'rgb(255,255,255)' : 'rgb(128,128,128)';
  strokeCtx.beginPath();
  strokeCtx.arc(p.x, p.y, brush / 2, 0, 7);
  strokeCtx.fill();
}}

function nearestNode(p) {{
  let best = -1, bestDistance = 12 / scale;
  contour.forEach((q, i) => {{
    const d = Math.hypot(q[0] - p.x, q[1] - p.y);
    if (d < bestDistance) {{ best = i; bestDistance = d; }}
  }});
  return best;
}}

function nearestEdge(p) {{
  let best = -1, bestDistance = 14 / scale;
  for (let i = 0; i < contour.length; i++) {{
    const a = contour[i], b = contour[(i + 1) % contour.length];
    const vx = b[0] - a[0], vy = b[1] - a[1];
    const len2 = vx * vx + vy * vy || 1;
    let t = ((p.x - a[0]) * vx + (p.y - a[1]) * vy) / len2;
    t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(a[0] + vx * t - p.x, a[1] + vy * t - p.y);
    if (d < bestDistance) {{ best = i; bestDistance = d; }}
  }}
  return best;
}}

canvas.oncontextmenu = e => e.preventDefault();

canvas.ondblclick = e => {{
  if (mode !== 'contour') return;
  const p = at(e);
  const edge = nearestEdge(p);
  if (edge >= 0) {{
    contour.splice(edge + 1, 0, [p.x, p.y]);
    contourHover = edge + 1;
    draw();
  }}
}};

canvas.onmousedown = e => {{
  const p = at(e);
  if (e.button === 1) {{  // средняя кнопка — тащить холст
    e.preventDefault();
    panning = {{ x: e.clientX - panX, y: e.clientY - panY }};
    return;
  }}
  if (mode === 'contour') {{
    const node = nearestNode(p);
    if (e.button === 2) {{
      if (node >= 0 && contour.length > 3) {{ contour.splice(node, 1); draw(); }}
      return;
    }}
    contourNode = node;
    return;
  }}
  if (mode === 'silhouette') {{
    painting = e.button === 2 ? 2 : 1;
    paintAt(p);
    draw();
  }} else {{
    drag = nearestJoint(p);
  }}
}};
canvas.onmousemove = e => {{
  if (panning) {{
    panX = e.clientX - panning.x;
    panY = e.clientY - panning.y;
    draw();
    return;
  }}
  const p = at(e);
  if (mode === 'contour') {{
    if (contourNode !== null && contourNode >= 0) contour[contourNode] = [p.x, p.y];
    else contourHover = nearestNode(p);
    draw();
    return;
  }}
  if (mode === 'silhouette') {{
    if (painting) {{ paintAt(p); draw(); }}
    return;
  }}
  if (drag) {{ joints[drag] = [p.x, p.y]; }}
  else {{ hover = nearestBone(p); }}
  draw();
}};
window.onmouseup = () => {{
  drag = null; painting = 0; contourNode = null; panning = null;
}};

canvas.onwheel = e => {{
  e.preventDefault();
  if (e.ctrlKey || e.metaKey || mode === 'contour') {{
    // зум под курсором: точка арта под мышкой остаётся на месте
    const box = canvas.getBoundingClientRect();
    const mx = e.clientX - box.left, my = e.clientY - box.top;
    const before = {{ x: (mx - panX) / scale, y: (my - panY) / scale }};
    zoom = Math.max(0.2, Math.min(12, zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    scale = baseScale * zoom;
    panX = mx - before.x * scale;
    panY = my - before.y * scale;
    draw();
    return;
  }}
  if (mode === 'silhouette') {{
    brush = Math.max(4, Math.min(200, brush * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    draw();
    return;
  }}
  const name = hover || nearestBone(at(e));
  if (!name) return;
  radii[name] = Math.max(0.3, Math.min(3,
    (radii[name] || 1) * (e.deltaY < 0 ? 1.06 : 1 / 1.06)));
  draw();
}};

document.getElementById('fit').onclick = () => fitView();
window.addEventListener('keydown', e => {{
  if (e.key === '0') fitView();
}});
document.getElementById('tab-contour').onclick = () => setMode('contour');
document.getElementById('tab-silhouette').onclick = () => setMode('silhouette');
document.getElementById('tab-skeleton').onclick = () => setMode('skeleton');

document.getElementById('reset').onclick = async () => {{
  if (mode === 'contour') {{
    const fetched = await (await fetch('contour.json?auto=1&' + Date.now())).json();
    contour = fetched.points || [];
  }} else if (mode === 'silhouette') {{
    strokeCtx.clearRect(0, 0, strokes.width, strokes.height);
  }} else {{
    joints = JSON.parse(JSON.stringify(base));
    for (const b of bones) radii[b.name] = 1;
  }}
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
  return {{
    joints: deltas,
    radius_scale: scales,
    strokes: strokes ? strokes.toDataURL('image/png') : null,
    contour: mode === 'contour' ? contour.map(p => [
      Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]) : null,
    // правка силуэта тянет за собой весь конвейер, правка скелета — часть
    start: mode === 'skeleton' ? 'pose' : 'background',
  }};
}}

async function post(path) {{
  const response = await fetch(path, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(payload()),
  }});
  return response;
}}

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


def _save_strokes(character: Character, data_url: str | None) -> int:
    """Сохранить мазки кистью по силуэту. Вернуть число закрашенных пикселей."""
    if not data_url or "," not in data_url:
        return 0
    import base64
    import io

    import numpy as np
    from PIL import Image

    raw = base64.b64decode(data_url.split(",", 1)[1])
    # холст приходит RGBA: белое — «это тело», серое — «это фон», прозрачное —
    # не трогать. Сводим к одному каналу, чтобы этап фона читал его маской
    painted = np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
    grey = painted[..., 0].astype(np.uint8)
    grey[painted[..., 3] < 128] = 0
    if not (grey > 64).any():
        character.silhouette_overrides.unlink(missing_ok=True)
        return 0
    character.write_mask(character.silhouette_overrides, grey)
    return int((grey > 64).sum())


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
        if path == "contour.json":
            character = self.character
            saved = character.contour_overrides
            fresh = "auto=1" in self.path
            if saved.exists() and not fresh:
                points = character.read_json(saved).get("points", [])
            else:
                from .stages.background import extract_contour

                points = extract_contour(character.read_mask(character.silhouette))
            body = json.dumps({"points": points}, ensure_ascii=False).encode("utf-8")
            return self._send(200, body, CONTENT_TYPES[".json"])
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
        painted = _save_strokes(self.character, payload.get("strokes"))
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
            + (f", мазков {painted} px" if painted else ""),
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
        skeleton = self.character.read_json(self.character.skeleton)
        try:
            checks = run_pipeline(
                self.character, skeleton["template"], start=start
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
