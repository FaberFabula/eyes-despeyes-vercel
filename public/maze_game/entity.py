"""World objects with positions."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .item import EPIC_ITEMS, HEAL_PACK, LOOT, Equipment
from .layout import Cell, Room, TILE, center
from .physics import Physics, Poly, Shape, Vec2d

if TYPE_CHECKING:
    from .actor import Actor

DOOR_HP = 15


class Entity:
    position: Vec2d

    def distance_to(self, point: Vec2d) -> float:
        return self.position.get_distance(point)


@dataclass(eq=False)
class RoomEye(Entity):
    cell: Cell
    room: Room
    shape: Shape
    hp: int

    @property
    def position(self) -> Vec2d:
        return center(self.cell)

    @property
    def alive(self) -> bool:
        return self.hp > 0


def solid_tile(physics: Physics, cell: Cell, other: Cell | None = None) -> Poly:
    if other is None:
        other = cell
    left, top = min(cell[0], other[0]) * TILE, min(cell[1], other[1]) * TILE
    right, bottom = (max(cell[0], other[0]) + 1) * TILE, (max(cell[1], other[1]) + 1) * TILE
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    return physics.add_wall(corners)


@dataclass(eq=False)
class Chest(Entity):
    cell: Cell
    epic: bool = False
    featured: Equipment | None = None
    opened: bool = False
    rewards: list[Equipment | int] = field(default_factory=list)
    next_reward: float = 0.0

    @property
    def position(self) -> Vec2d:
        return center(self.cell)

    def open(self, now: float, rng) -> None:
        self.opened = True
        self.next_reward = now
        if self.epic:
            self.rewards = [2, 2, 2, HEAL_PACK, HEAL_PACK, self.featured or rng.choice(EPIC_ITEMS)]
        else:
            self.rewards = [1, 1, 1, HEAL_PACK, *rng.choices((*LOOT, HEAL_PACK), k=2)]


@dataclass(eq=False)
class XPOrb:
    position: Vec2d
    amount: int
    velocity: Vec2d
    collect_at: float


@dataclass(eq=False)
class Door(Entity):
    cell: Cell
    shape: Poly
    is_open: bool = False
    locked: bool = False
    jamb: Cell | None = None
    wide: bool = False
    breach: float = 0.0
    broken: bool = False
    hp: int = DOOR_HP
    opened_at: float = 0.0

    @property
    def position(self) -> Vec2d:
        if self.wide and self.jamb is not None:
            return (center(self.cell) + center(self.jamb)) / 2
        return center(self.cell)

    def toggle(self, physics: Physics, actors: list["Actor"]) -> bool:
        if self.broken:
            return False
        if self.is_open:
            for actor in actors:
                if actor.active and self.shape.point_query(actor.position).distance < actor.radius:
                    return False
            physics.add_shape(self.shape)
        else:
            physics.remove_shape(self.shape)
        self.is_open = not self.is_open
        return True
