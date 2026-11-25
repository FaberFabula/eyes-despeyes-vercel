"""Item specs and equipped or loose gear."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, ceil, cos, sin
from typing import TYPE_CHECKING

from .physics import Body, PointHit, Poly, Segment, Shape, Space, Vec2d, add_item_body

if TYPE_CHECKING:
    from .actor import Actor


@dataclass(frozen=True)
class AttackProfile:
    reach: float
    damage: int
    recovery: float
    kind: str = "blunt"

    def damage_against(self, organic: bool, force: float = 1.0) -> int:
        organic_factor, inorganic_factor = {
            "blunt": (0.8, 1.4),
            "piercing": (1.35, 0.55),
            "slashing": (1.15, 0.75),
            "arcane": (1.0, 1.0),
            "heat": (1.2, 0.6),
        }[self.kind]
        factor = organic_factor if organic else inorganic_factor
        return max(1, round(self.damage * factor * force)) if self.damage > 0 else 0


@dataclass(frozen=True)
class Equipment:
    name: str
    attack: AttackProfile | None = None
    mode: str = "melee"
    length: float = 34
    mass: float = 0.8
    durability: int | None = 52
    effect: str | None = None
    projectile_speed: float = 510
    width: float = 8
    sharp_grip_from: float | None = None  # Fraction along the local axis, from -1 to +1.
    draw_time: float = 0.0


BLADE = Equipment("Sword", AttackProfile(62, 4, 0.55, "slashing"), length=42,
                  effect="sword", sharp_grip_from=-0.25)
HAMMER = Equipment("Hammer", AttackProfile(58, 4, 0.76), length=36, mass=1.35,
                   width=13, effect="hammer")
SLINGSHOT = Equipment("Slingshot", AttackProfile(250, 4, 0.70, "arcane"), "ranged", length=24,
                     width=13, durability=38, projectile_speed=625, effect="sling", draw_time=0.46)
FLAMETHROWER = Equipment("Flamethrower", AttackProfile(130, 2, 0.075, "heat"),
                         "ranged", length=45, mass=1.65, width=14, durability=66, effect="flame")
GRAVITY_GUN = Equipment("Gravity Gun", AttackProfile(320, 0, 1.55, "arcane"),
                        "magic", length=35, mass=1.35, width=15, durability=54,
                        effect="gravity", projectile_speed=430)
SHIELD = Equipment("Shield", AttackProfile(66, 3, 0.8), mode="shield", length=13, width=37, mass=1.55,
                   durability=140, effect="shield")
BOW = Equipment("Bow", AttackProfile(345, 12, 1.15, "piercing"), "ranged",
                length=25, width=28, effect="bow", draw_time=1.06, projectile_speed=950)
WAND = Equipment("Wand", AttackProfile(255, 7, 1.03, "arcane"), "magic",
                 length=26, mass=0.5, projectile_speed=472, effect="wand")
LASER = Equipment("Laser", AttackProfile(455, 8, 1.33, "piercing"), "ranged",
                  length=29, effect="beam")
TORCH = Equipment("Torch", mode="utility", length=13, width=27, mass=0.55,
                  durability=18, effect="torch")
RADAR = Equipment("Radar", mode="utility", length=18, width=14, mass=0.6, durability=24, effect="radar")
HEAL_PACK = Equipment("Healing Pack", mode="utility", length=18, width=12, mass=0.5, durability=8, effect="heal")
GATE_KEY = Equipment("Gate Key", mode="utility", length=20, mass=0.45, durability=None, effect="key")
MIRROR = Equipment("Exit Mirror", mode="utility", length=42, width=14, mass=4.8, durability=None, effect="mirror")
DEBUG_TOOL = Equipment("De Bug", mode="utility", length=32, width=16, mass=0.45,
                        durability=None, effect="debug")
SPEAR = Equipment("Spear", AttackProfile(93, 4, 0.64, "piercing"), length=81,
                  mass=1.35, durability=58, effect="spear", width=9, sharp_grip_from=0.48)
MAN_CATCHER = Equipment("Man Catcher", AttackProfile(82, 0, 0.7), length=70,
                        mass=1.48, durability=60, effect="catcher", width=28, sharp_grip_from=0.67)
PULSE_GUN = Equipment("Pulse Gun", AttackProfile(345, 5, 1.14, "arcane"), "ranged",
                      length=36, mass=1.35, effect="pulse", projectile_speed=632, width=17)
GRIMOIRE = Equipment("Chaotic Grimoire", AttackProfile(245, 7, 0.14, "arcane"), "magic",
                     length=26, width=23, mass=0.85, durability=None, effect="grimoire")
SEALING = Equipment("Sealing Equipment", AttackProfile(230, 2, 0.20, "arcane"), "magic",
                    length=24, width=20, mass=1.0, durability=None, effect="seal")
WHIP = Equipment("Cat-o'-nine-tails", AttackProfile(117, 3, 0.46, "piercing"),
                 length=24, width=7, mass=0.65, durability=None, effect="whip")
ALL_LAUNCHER = Equipment("All-launcher", AttackProfile(172, 8, 1.05), "ranged",
                         length=40, width=19, mass=1.65, durability=None, effect="launcher")
WEAPONS = (BLADE, HAMMER, BOW, WAND, LASER, SLINGSHOT, FLAMETHROWER, GRAVITY_GUN,
           SPEAR, MAN_CATCHER, PULSE_GUN)
LOOT = (*WEAPONS, SHIELD, RADAR)
EPIC_ITEMS = (GRIMOIRE, SEALING, WHIP, ALL_LAUNCHER)
SCOUT_SCRATCH = AttackProfile(39, 2, 0.48, "slashing")
SCOUT_BITE = AttackProfile(27, 3, 0.64, "piercing")
CLAW_ATTACK = AttackProfile(45, 3, 0.54, "slashing")


def healing_pack_hp(max_hp: int) -> int:
    return 5 + ceil(max_hp / 10)


@dataclass(eq=False)
class Gear:
    spec: Equipment
    body: Body
    shape: Poly
    durability: int | None
    holder: Actor | None = None
    contender: Actor | None = None
    physics_safe_until: float = 0.0
    light_on: bool = False
    activated_at: float = float('-inf')
    special_ready: float = 0.0
    impact_times: dict = field(default_factory=dict)
    jaws: tuple[Segment, ...] = ()
    jaw_closure: float = 0.0
    temporary: bool = False
    flame_pulses: int = 0

    @classmethod
    def spawn(cls, space: Space, spec: Equipment, position: Vec2d, angle: float = 0) -> "Gear":
        body, shape, jaws = add_item_body(space, spec, position, angle)
        return cls(spec, body, shape, spec.durability, jaws=jaws,
                   light_on=spec.effect == "torch")

    @property
    def shapes(self) -> tuple[Shape, ...]:
        return (self.shape, *self.jaws)

    def point_query(self, position: Vec2d) -> PointHit:
        closest = self.shape.point_query(position)
        for jaw in self.jaws:
            query = jaw.point_query(position)
            if query.distance < closest.distance:
                closest = query
        return closest

    @property
    def position(self) -> Vec2d:
        return self.body.position

    @property
    def axis(self) -> Vec2d:
        return Vec2d(1, 0).rotated(self.body.angle)

    def point_on_axis(self, fraction: float) -> Vec2d:
        return self.body.local_to_world((self.spec.length * 0.5 * fraction, 0))

    def sharp_at(self, point: Vec2d) -> bool:
        start = self.spec.sharp_grip_from
        return start is not None and self.body.world_to_local(point).x >= self.spec.length * 0.5 * start

    def steer(self, angle: float, grip: Vec2d, *, manual: bool = False) -> None:
        difference = angle - self.body.angle
        error = atan2(sin(difference), cos(difference))
        gain, damping = (170, 24) if manual else (88, 16)
        drive = gain * error - damping * self.body.angular_velocity
        inertia = self.body.moment + self.body.mass * grip.length_squared
        self.body.torque += inertia * max(-310, min(310, drive))
