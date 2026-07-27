"""Шаблоны скелетов в нормализованных координатах A-позы.

Нормировка: единица — рост персонажа. y=0 — макушка, y=1 — низ стоп,
x=0.5 — центральная ось. Масштаб x тот же, что и y (аспект сохраняется),
поэтому шаблон занимает по ширине примерно 0.6 своего роста.

Каждая кость даёт ровно одну часть тела — часть жёстко привязана к кости,
пивот части = начальный сустав кости. Иерархия костей = иерархия рига.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Bone:
    """Кость шаблона.

    ra/rb — радиусы капсулы в начале и конце (в долях роста); капсула служит
    и формой для процедурного демо-персонажа, и фоллбек-маской части.
    z — порядок глубины (больше = ближе к зрителю).
    overlap — множитель нахлёста части на соседей (торсу нужен больший,
    иначе при отведении конечности вылезает дырка).
    """

    name: str
    parent: str | None
    a: str
    b: str
    ra: float
    rb: float
    z: int
    overlap: float = 1.0


@dataclass(frozen=True)
class Template:
    name: str
    joints: dict[str, tuple[float, float]]
    bones: tuple[Bone, ...]
    chains: dict[str, tuple[str, ...]]

    @property
    def root(self) -> Bone:
        return next(b for b in self.bones if b.parent is None)

    def bone(self, name: str) -> Bone:
        return next(b for b in self.bones if b.name == name)

    def ordered_bones(self) -> list[Bone]:
        """Кости в порядке «родитель раньше ребёнка»."""
        by_parent: dict[str | None, list[Bone]] = {}
        for b in self.bones:
            by_parent.setdefault(b.parent, []).append(b)
        out: list[Bone] = []
        stack = list(by_parent.get(None, []))
        while stack:
            bone = stack.pop(0)
            out.append(bone)
            stack.extend(by_parent.get(bone.name, []))
        return out

    def drop_chain(self, chain: str, name: str | None = None) -> "Template":
        """Шаблон без опциональной цепочки (например, без хвоста)."""
        names = set(self.chains.get(chain, ()))
        bones = tuple(b for b in self.bones if b.name not in names)
        used = {b.a for b in bones} | {b.b for b in bones}
        joints = {k: v for k, v in self.joints.items() if k in used}
        chains = {k: v for k, v in self.chains.items() if k != chain}
        return replace(
            self,
            name=name or f"{self.name}_no_{chain}",
            joints=joints,
            bones=bones,
            chains=chains,
        )


# --- biped_tail: антропоморф с хвостом и digitigrade-ногами --------------
#
# Поза A: руки опущены и разведены ~45°, ноги слегка расставлены, хвост
# уходит назад-вниз и загибается вверх. Всё, что смотрит вбок, задаётся
# зеркально: _l — левая сторона персонажа (правая для зрителя).

_JOINTS: dict[str, tuple[float, float]] = {
    "head_top": (0.500, 0.030),
    "neck": (0.500, 0.215),
    "pelvis": (0.500, 0.520),
    # руки
    "shoulder_l": (0.585, 0.265),
    "elbow_l": (0.680, 0.400),
    "wrist_l": (0.755, 0.520),
    "hand_l": (0.795, 0.585),
    "shoulder_r": (0.415, 0.265),
    "elbow_r": (0.320, 0.400),
    "wrist_r": (0.245, 0.520),
    "hand_r": (0.205, 0.585),
    # ноги (digitigrade: бедро → голень → плюсна+стопа)
    "hip_l": (0.560, 0.530),
    "knee_l": (0.600, 0.665),
    "hock_l": (0.565, 0.800),
    "toe_l": (0.600, 0.965),
    "hip_r": (0.440, 0.530),
    "knee_r": (0.400, 0.665),
    "hock_r": (0.435, 0.800),
    "toe_r": (0.400, 0.965),
    # хвост
    # хвост уходит назад-вниз за ноги: во фронтальной A-позе задранный вбок
    # хвост пересекает руку и читается как третья конечность
    "tail_a": (0.452, 0.545),
    "tail_b": (0.330, 0.635),
    "tail_c": (0.268, 0.760),
    "tail_d": (0.238, 0.880),
}

_BONES: tuple[Bone, ...] = (
    Bone("torso", None, "pelvis", "neck", 0.105, 0.088, 20, overlap=1.8),
    Bone("head", "torso", "neck", "head_top", 0.072, 0.090, 40, overlap=1.2),
    # правая сторона персонажа (для зрителя левая) — чуть дальше от камеры
    Bone("arm_upper_r", "torso", "shoulder_r", "elbow_r", 0.048, 0.040, 22),
    Bone("arm_fore_r", "arm_upper_r", "elbow_r", "wrist_r", 0.040, 0.032, 23),
    Bone("hand_r", "arm_fore_r", "wrist_r", "hand_r", 0.036, 0.030, 24),
    Bone("arm_upper_l", "torso", "shoulder_l", "elbow_l", 0.048, 0.040, 26),
    Bone("arm_fore_l", "arm_upper_l", "elbow_l", "wrist_l", 0.040, 0.032, 27),
    Bone("hand_l", "arm_fore_l", "wrist_l", "hand_l", 0.036, 0.030, 28),
    Bone("thigh_r", "torso", "hip_r", "knee_r", 0.062, 0.050, 8, overlap=1.4),
    Bone("shin_r", "thigh_r", "knee_r", "hock_r", 0.050, 0.034, 9),
    Bone("foot_r", "shin_r", "hock_r", "toe_r", 0.034, 0.030, 10),
    Bone("thigh_l", "torso", "hip_l", "knee_l", 0.062, 0.050, 12, overlap=1.4),
    Bone("shin_l", "thigh_l", "knee_l", "hock_l", 0.050, 0.034, 13),
    Bone("foot_l", "shin_l", "hock_l", "toe_l", 0.034, 0.030, 14),
    Bone("tail_1", "torso", "tail_a", "tail_b", 0.036, 0.028, -2),
    Bone("tail_2", "tail_1", "tail_b", "tail_c", 0.028, 0.020, -3),
    Bone("tail_3", "tail_2", "tail_c", "tail_d", 0.020, 0.010, -4),
)

BIPED_TAIL = Template(
    name="biped_tail",
    joints=_JOINTS,
    bones=_BONES,
    chains={
        "tail": ("tail_1", "tail_2", "tail_3"),
        "leg_digitigrade": ("foot_l", "foot_r"),
    },
)

BIPED = BIPED_TAIL.drop_chain("tail", name="biped")

TEMPLATES: dict[str, Template] = {t.name: t for t in (BIPED_TAIL, BIPED)}

DEFAULT_TEMPLATE = "biped_tail"


def get(name: str) -> Template:
    if name not in TEMPLATES:
        known = ", ".join(sorted(TEMPLATES))
        raise KeyError(f"неизвестный шаблон {name!r}; есть: {known}")
    return TEMPLATES[name]


def extent(
    template: Template, x_scale: float = 1.0
) -> tuple[float, float, float, float]:
    """Габариты шаблона (x0, x1, y0, y1) с учётом радиусов капсул.

    Радиусы при растяжении по ширине не масштабируются: персонаж бывает
    шире в размахе рук, но не толще в кости.
    """
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for bone in template.bones:
        for joint, radius in ((bone.a, bone.ra), (bone.b, bone.rb)):
            x, y = template.joints[joint]
            xs = 0.5 + (x - 0.5) * x_scale
            x0, x1 = min(x0, xs - radius), max(x1, xs + radius)
            y0, y1 = min(y0, y - radius), max(y1, y + radius)
    return x0, x1, y0, y1


def place(
    template: Template, scale: float, origin: tuple[float, float], x_scale: float = 1.0
) -> dict[str, tuple[float, float]]:
    """Перевод нормализованных суставов в пиксели.

    scale — рост персонажа в пикселях, origin — левый верхний угол его bbox,
    x_scale — дополнительное растяжение по ширине относительно центральной оси
    (шаблон один, а персонажи бывают шире или уже).
    """
    ox, oy = origin
    return {
        name: (ox + (0.5 + (x - 0.5) * x_scale) * scale, oy + y * scale)
        for name, (x, y) in template.joints.items()
    }
