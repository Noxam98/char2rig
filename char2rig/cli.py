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


def run_pipeline(
    char: Character,
    template_name: str,
    use_ml: bool = True,
    start: str = "background",
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
        joints, method, fallback, params = pose.run(
            template, rgba, alpha, use_ml, overrides
        )
        char.write_json(
            char.skeleton,
            {
                "template": template.name,
                "size": [width, height],
                "fit": params,
                "joints": {k: [round(v[0], 2), round(v[1], 2)] for k, v in joints.items()},
            },
        )
        char.record_stage("pose", method, fallback, **params)
        say(f"  скелет:   {method}{' (фоллбек)' if fallback else ''}")
    else:
        saved = char.read_json(char.skeleton)
        joints = {k: (v[0], v[1]) for k, v in saved["joints"].items()}

    unit = segment.pixels_per_unit(joints, template)

    if _from(start, "segment"):
        masks, method, fallback, stats = segment.run(
            template, rgba, alpha, joints, use_ml
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
            template, rgba, alpha, joints, masks, unit
        )
        char.record_stage("layers", method, fallback, **stats)
        say(
            f"  слои:     {stats['layers']} шт, непокрытых пикселей "
            f"{stats['uncovered_px']}, достроено {stats['inpainted_px']}"
        )
        rig_data = rig.run(char, template, joints, cut, (width, height), unit)
        char.record_stage("rig", "joint_pivots", False, bones=len(rig_data["bones"]))
    else:
        rig_data = char.read_json(char.rig)
        stats = {}

    images = rig.load_images(char, rig_data)
    checks = render.swing(char, rig_data, images)
    checks.update({k: v for k, v in stats.items() if k.startswith("uncovered")})
    verdict = render.triage(checks)
    char.record_checks(checks, verdict)
    say(
        f"  свинг:    {checks['frames']} кадров, щели {checks['gap_px']} px "
        f"({checks['gap_ratio']:.3%}, худший кадр {checks['worst_frame']}) → {verdict}"
    )
    say(f"  готово:   {char.preview}")
    return checks


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import draw

    width, height = (int(v) for v in args.size.lower().split("x"))
    template = templates.get(args.template)
    char = Character.open(args.name, args.dir)
    say(f"демо-кот {args.name} ({width}x{height}, шаблон {template.name})")
    char.write_rgba(char.source, draw(template, (width, height)))
    run_pipeline(char, template.name, use_ml=not args.no_ml)
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
    run_pipeline(char, args.template, use_ml=not args.no_ml)
    return 0


def cmd_recut(args: argparse.Namespace) -> int:
    char = Character.open(args.name, args.dir)
    saved = char.read_json(char.skeleton)
    say(f"пересборка {args.name} с этапа {args.stage}")
    run_pipeline(char, saved["template"], use_ml=not args.no_ml, start=args.stage)
    return 0


def cmd_swing(args: argparse.Namespace) -> int:
    char = Character.open(args.name, args.dir)
    rig_data = char.read_json(char.rig)
    images = rig.load_images(char, rig_data)
    checks = render.swing(char, rig_data, images, frames=args.frames)
    verdict = render.triage(checks)
    char.record_checks(checks, verdict)
    say(
        f"свинг {args.name}: щели {checks['gap_px']} px "
        f"({checks['gap_ratio']:.3%}, худший кадр {checks['worst_frame']}) → {verdict}"
    )
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    say(
        "редактор суставов ещё не написан (PLAN.md: появится после первого "
        "прогона). Пока правки — руками в skeleton.overrides.json:\n"
        '  {"joints": {"elbow_l": [12, -4]}}\n'
        "затем `python -m char2rig recut <имя> --stage pose`."
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="char2rig", description="картинка персонажа → риг → анимация"
    )
    parser.add_argument(
        "--dir", default=str(CHARACTERS_DIR), help="папка с персонажами"
    )
    parser.add_argument(
        "--no-ml", action="store_true", help="не трогать нейросети даже если стоят"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="нарисовать демо-кота и прогнать конвейер")
    demo.add_argument("--name", default="demo_cat")
    demo.add_argument("--size", default="512x768")
    demo.add_argument("--template", default=templates.DEFAULT_TEMPLATE)
    demo.set_defaults(func=cmd_demo)

    process = sub.add_parser("process", help="прогнать конвейер на PNG")
    process.add_argument("image")
    process.add_argument("--name")
    process.add_argument("--template", default=templates.DEFAULT_TEMPLATE)
    process.set_defaults(func=cmd_process)

    recut = sub.add_parser("recut", help="пересчитать с указанного этапа")
    recut.add_argument("name")
    recut.add_argument("--stage", default="segment", choices=STAGES)
    recut.set_defaults(func=cmd_recut)

    swing = sub.add_parser("swing", help="пересобрать превью из готового рига")
    swing.add_argument("name")
    swing.add_argument("--frames", type=int, default=render.FRAMES)
    swing.set_defaults(func=cmd_swing)

    edit = sub.add_parser("edit", help="правка суставов (пока вручную)")
    edit.add_argument("name")
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
