"""Timed light sources and transient feedback driven by the simulation."""

from dataclasses import dataclass


from .physics import Shape, Vec2d
from .actor import Actor


@dataclass
class Fire:
    position: Vec2d
    owner: Actor
    expires: float
    radius: float = 68
    power: float = 1.0
    weapon: str = "Fire"


@dataclass
class Burn:
    owner: Actor
    expires: float
    next_tick: float
    power: float
    weapon: str = "Fire"


@dataclass
class GravityWell:
    position: Vec2d
    owner: Actor
    expires: float
    radius: float = 156
    strength: float = 1.0
    weapon: str = "Gravity Gun"


@dataclass(frozen=True)
class WeaponFlash:
    position: Vec2d
    radius: float
    color: tuple[int, int, int]
    started: float
    direction: Vec2d | None = None
    half_angle: float = 0.0


@dataclass(frozen=True)
class Beam:
    start: Vec2d
    end: Vec2d
    expires: float
    width: int = 4
    slashing: bool = False


DEATH_ANIMATION_SECONDS = 2.0


@dataclass
class Death:
    weapon: str
    delivery: str
    attacker: str | None
    position: Vec2d
    velocity: Vec2d
    direction: Vec2d
    attacker_position: Vec2d | None = None
    elapsed: float = 0.0
    quote: str = ""


@dataclass(frozen=True)
class Hit:
    position: Vec2d
    shape: Shape
    started: float
    kind: str
    damage: int


@dataclass(frozen=True)
class Cue:
    name: str
    volume: float
    pan: float
