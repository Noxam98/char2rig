"""CLI фазы 0: `python -m char2rig <команда>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import template as templates
from .character import CHARACTERS_DIR, Character
from .stages import background, layers, pose, render, rig, segment

STAGES = ("background", "pose", "segment", "layers", "rig", "render")


def say(message: str) -> None:
    print(message, flush=True)


def _from(start: str, stage: str) -> bool:
    """Нужно ли считать `stage`, если конвейер запущен с `start`."""
    return STAGES.index(stage) >= STAGES.index(start)


def _texture(rgba: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Исходник с альфой силуэта: фон не должен ехать вместе с телом."""
    texture = rgba.copy()
    texture[..., 3] = alpha
    return texture


def run_pipeline(
    char: Character,
    template_name: str,
    use_ml: bool = True,
    start: str = "background",
    preview: str = "mesh",
) -> dict:
    template = templates.get(template_name)
    rgba = char.read_rgba(char.source)
    height, width = rgba.shape[:2]

    if _from(start, "background"):
        alpha, method, fallback = background.run(rgba, use_ml)
        char.write_mask(char.silhouette, alpha)
        char.record_stage("background", method, fallback)
        say(f"  фон:      {method}{' (фоллбек)' if fallback else ''}")
    else:
        alpha = char.read_mask(char.silhouette)

    if _from(start, "pose"):
        overrides = (
            char.read_json(char.skeleton_overrides)
            if char.skeleton_overrides.exists()
            else None
        )
        joints, radii, method, fallback, params = pose.run(
            template, rgba, alpha, use_ml, overrides
        )
        char.write_json(
            char.skeleton,
            {
                "template": template.name,
                "size": [width, height],
                "fit": params,
                "joints": {
                    k: [round(v[0], 2), round(v[1], 2)] for k, v in joints.items()
                },
                "radii": {
                    k: [round(v[0], 2), round(v[1], 2)] for k, v in radii.items()
                },
            },
        )
        char.record_stage("pose", method, fallback, **params)
        say(f"  скелет:   {method}{' (фоллбек)' if fallback else ''}")
    else:
        saved = char.read_json(char.skeleton)
        joints = {k: (v[0], v[1]) for k, v in saved["joints"].items()}
        radii = {k: (v[0], v[1]) for k, v in saved.get("radii", {}).items()} or None

    unit = segment.pixels_per_unit(joints, template)

    if _from(start, "segment"):
        masks, method, fallback, stats = segment.run(
            template, rgba, alpha, joints, radii, use_ml
        )
        for name, mask in masks.items():
            char.write_mask(char.masks_dir / f"{name}.png", mask)
        char.record_stage("segment", method, fallback, **stats)
        say(f"  части:    {method} — {stats['parts']} шт")
    else:
        masks = {
            path.stem: char.read_mask(path)
            for path in sorted(char.masks_dir.glob("*.png"))
        }

    if _from(start, "layers"):
        cut, method, fallback, stats = layers.run(
            template, rgba, alpha, joints, masks, unit, radii, use_ml
        )
        char.record_stage("layers", method, fallback, **stats)
        say(
            f"  слои:     {stats['layers']} шт ({method}), непокрытых пикселей "
            f"{stats['uncovered_px']}, скрытых {stats['hidden_px']}"
        )
        rig_data = rig.run(
            char, template, joints, cut, (width, height), unit, radii
        )
        char.record_stage("rig", "joint_pivots", False, bones=len(rig_data["bones"]))
    else:
        rig_data = char.read_json(char.rig)
        stats = {}

    if preview == "mesh":
        checks = render.swing_mesh(char, rig_data, _texture(rgba, alpha), alpha)
    else:
        images = rig.load_images(char, rig_data)
        checks = render.swing(char, rig_data, images)
    checks.update({k: v for k, v in stats.items() if k.startswith("uncovered")})
    pose_stage = char.status()["stages"].get("pose", {})
    checks["fit_inside"] = pose_stage.get("fit_inside", 1.0)
    checks["fit_iou"] = pose_stage.get("fit_iou", 1.0)
    verdict = render.triage(checks)
    char.record_checks(checks, verdict)
    if preview == "mesh":
        say(
            f"  свинг:    {checks['frames']} кадров сеткой "
            f"({checks['mesh_triangles']} треугольников), скелет внутри арта "
            f"{checks['fit_inside']:.2f} → {verdict}"
        )
    else:
        say(
            f"  свинг:    {checks['frames']} кадров, швы "
            f"{checks['seam_gap_ratio']:.3%} (худший кадр {checks['worst_frame']}), "
            f"скелет внутри арта {checks['fit_inside']:.2f} → {verdict}"
        )
    say(f"  готово:   {char.preview}")
    return checks


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import draw

    width, height = (int(v) for v in args.size.lower().split("x"))
    template = templates.get(args.template)
    char = Character.open(args.name, args.dir)
    say(f"демо-кот {args.name} ({width}x{height}, шаблон {template.name})")
    rgba, _ = draw(template, (width, height))
    char.write_rgba(char.source, rgba)
    run_pipeline(
        char, template.name, use_ml=not args.no_ml, preview=args.preview
    )
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from .dataset import generate

    out = Path(args.out)
    say(f"набор {args.count} персонажей → {out} (seed {args.seed})")
    summary = generate(
        out,
        count=args.count,
        seed=args.seed,
        template_name=args.template,
        preview=args.preview_count,
        on_progress=lambda done, total: say(f"  {done}/{total}"),
    )
    say(
        f"готово: {summary['count']} образцов, {summary['labels']} меток "
        f"(0 — фон), превью {out / 'preview.png'}"
    )
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    source = Path(args.image)
    if not source.exists():
        say(f"нет файла {source}")
        return 1
    name = args.name or source.stem
    char = Character.open(name, args.dir)
    say(f"персонаж {name} ← {source}")
    char.write_rgba(char.source, char.read_rgba(source))
    run_pipeline(
        char, args.template, use_ml=not args.no_ml, preview=args.preview
    )
    return 0


def _require(char: Character, path: Path, hint: str) -> bool:
    if path.exists():
        return True
    say(f"нет {path} — сначала {hint}")
    return False


def cmd_recut(args: argparse.Namespace) -> int:
    char = Character.open(args.name, args.dir)
    if not _require(char, char.skeleton, f"`char2rig process <png> --name {args.name}`"):
        return 1
    saved = char.read_json(char.skeleton)
    say(f"пересборка {args.name} с этапа {args.stage}")
    run_pipeline(
        char,
        saved["template"],
        use_ml=not args.no_ml,
        start=args.stage,
        preview=args.preview,
    )
    return 0


def cmd_swing(args: argparse.Namespace) -> int:
    char = Character.open(args.name, args.dir)
    if not _require(char, char.rig, f"`char2rig process <png> --name {args.name}`"):
        return 1
    rig_data = char.read_json(char.rig)
    if args.preview == "mesh":
        alpha = char.read_mask(char.silhouette)
        texture = _texture(char.read_rgba(char.source), alpha)
        checks = render.swing_mesh(char, rig_data, texture, alpha, frames=args.frames)
    else:
        images = rig.load_images(char, rig_data)
        checks = render.swing(char, rig_data, images, frames=args.frames)
    status = char.status()
    pose_stage = status["stages"].get("pose", {})
    checks["fit_inside"] = pose_stage.get("fit_inside", 1.0)
    checks["uncovered_ratio"] = (
        status["stages"].get("layers", {}).get("uncovered_ratio", 0.0)
    )
    verdict = render.triage(checks)
    char.record_checks(checks, verdict)
    seams = (
        f"швы {checks['seam_gap_ratio']:.3%}, "
        if "seam_gap_ratio" in checks
        else ""
    )
    say(
        f"свинг {args.name}: {seams}"
        f"скелет внутри арта {checks['fit_inside']:.2f} → {verdict}"
    )
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    import webbrowser

    from .editor import serve, url

    char = Character.open(args.name, args.dir)
    if not _require(char, char.skeleton, f"`char2rig process <png> --name {args.name}`"):
        return 1

    server = serve(char, args.port)
    address = url(server)
    say(f"редактор суставов: {address}")
    say("тяни суставы мышкой, жми «Сохранить правки», потом:")
    say(f"  python -m char2rig recut {args.name} --stage pose")
    say("остановить — Ctrl+C")
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        say("редактор остановлен")
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    # общие флаги дублируются в каждую подкоманду: argparse иначе принимает
    # их только до её имени, а писать `char2rig --no-ml demo` неудобно
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dir", default=str(CHARACTERS_DIR), help="папка с персонажами"
    )
    common.add_argument(
        "--no-ml", action="store_true", help="не трогать нейросети даже если стоят"
    )
    common.add_argument(
        "--preview",
        choices=("mesh", "parts"),
        default="mesh",
        help="чем рисовать превью: деформацией сетки или жёсткими частями",
    )

    parser = argparse.ArgumentParser(
        prog="char2rig",
        parents=[common],
        description="картинка персонажа → риг → анимация",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "demo", parents=[common], help="нарисовать демо-кота и прогнать конвейер"
    )
    demo.add_argument("--name", default="demo_cat")
    demo.add_argument("--size", default="512x768")
    demo.add_argument("--template", default=templates.DEFAULT_TEMPLATE)
    demo.set_defaults(func=cmd_demo)

    dataset = sub.add_parser(
        "dataset", parents=[common], help="сгенерировать обучающий набор"
    )
    dataset.add_argument("--count", type=int, default=200)
    dataset.add_argument("--out", default="dataset")
    dataset.add_argument("--seed", type=int, default=1)
    dataset.add_argument("--preview-count", type=int, default=24)
    dataset.add_argument("--template", default=templates.DEFAULT_TEMPLATE)
    dataset.set_defaults(func=cmd_dataset)

    process = sub.add_parser(
        "process", parents=[common], help="прогнать конвейер на PNG"
    )
    process.add_argument("image")
    process.add_argument("--name")
    process.add_argument("--template", default=templates.DEFAULT_TEMPLATE)
    process.set_defaults(func=cmd_process)

    recut = sub.add_parser(
        "recut", parents=[common], help="пересчитать с указанного этапа"
    )
    recut.add_argument("name")
    recut.add_argument("--stage", default="segment", choices=STAGES)
    recut.set_defaults(func=cmd_recut)

    swing = sub.add_parser(
        "swing", parents=[common], help="пересобрать превью из готового рига"
    )
    swing.add_argument("name")
    swing.add_argument("--frames", type=int, default=render.FRAMES)
    swing.set_defaults(func=cmd_swing)

    edit = sub.add_parser(
        "edit", parents=[common], help="правка суставов мышкой в браузере"
    )
    edit.add_argument("name")
    edit.add_argument("--port", type=int, default=8765)
    edit.add_argument("--no-browser", action="store_true")
    edit.set_defaults(func=cmd_edit)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # чтобы русский текст не падал на cp866-консоли Windows
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    np.seterr(all="ignore")
    return args.func(args)
