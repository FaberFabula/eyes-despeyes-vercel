from dataclasses import dataclass
from heapq import heappop, heappush
from typing import TYPE_CHECKING

from .physics import Vec2d

from .layout import Cell, tile_at

if TYPE_CHECKING:
    from .world import Maze

NOTICE_TIME = 0.65
HOSTILE_TIME = 0.32
SIGHT_LOSS_TIME = 0.55
SEARCH_TIME = 10.0


@dataclass
class Alert:
    yellow: float = 0.0
    red: float = 0.0
    aware: bool = False
    hostile: bool = False
    engaged: bool = False
    visible: bool = False
    memory: float = 0.0
    unnoticed_for: float = 0.0
    damage: int = 0
    pressure_seconds: float = 0.0
    role: str | None = None
    can_be_hostile: bool = True

    @property
    def score(self) -> float:
        return self.damage + self.pressure_seconds / 3

    def become_hostile(self) -> None:
        if not self.can_be_hostile:
            return
        self.yellow = self.red = 1.0
        self.aware = self.hostile = self.engaged = True
        self.role = None
        self.unnoticed_for = 0.0

    def update(self, visible: bool, dt: float, *, tracked: bool = False) -> None:
        if not self.can_be_hostile:
            return
        self.visible = visible
        if visible:
            self.memory = 4.0
            self.unnoticed_for = 0.0
            if self.aware:
                self.red = min(1.0, self.red + dt / HOSTILE_TIME)
                if self.red >= 1.0:
                    self.become_hostile()
            else:
                self.yellow = min(1.0, self.yellow + dt / NOTICE_TIME)
                self.aware = self.yellow >= 1.0
            return

        if tracked:
            self.memory = 4.0
            self.unnoticed_for = 0.0
            if not self.hostile:
                self.red = 0.0
            return
        if self.hostile:
            self.unnoticed_for += dt
            if self.unnoticed_for < SIGHT_LOSS_TIME:
                return
            self.hostile = False
            self.memory = 4.0
        self.red = 0.0
        if self.role is not None:
            return
        self.memory = max(0.0, self.memory - dt)
        if self.memory == 0:
            self.aware = False
            self.yellow = max(0.0, self.yellow - dt)

    def hear(self) -> None:
        if not self.can_be_hostile:
            return
        if not self.aware:
            self.yellow = max(self.yellow, 0.24)
        self.memory = max(self.memory, 2.0)

    def assign(self, role: str) -> None:
        if not self.can_be_hostile:
            return
        self.role = role
        self.aware = True
        self.hostile = False
        self.yellow = 1.0
        self.red = 0.0

    def stand_down(self) -> None:
        self.engaged = False
        self.role = None
        self.red = 0.0
        self.memory = 4.0
        self.damage = 0
        self.pressure_seconds = 0.0


def sound_distances(maze: "Maze", origin: Vec2d, radius: int) -> dict[Cell, int]:
    """Flood the walkable tiles. Closed doors attenuate sound by two extra tiles."""
    start = tile_at(origin)
    if maze.current_tile(start) == "#":
        return {}
    costs = {start: 0}
    frontier = [(0, start)]
    while frontier:
        cost, cell = heappop(frontier)
        if cost != costs[cell]:
            continue
        x, y = cell
        for neighbor in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            if maze.current_tile(neighbor) == "#":
                continue
            door = maze.doors.get(neighbor)
            next_cost = cost + (3 if door is not None and not door.is_open else 1)
            if next_cost <= radius and next_cost < costs.get(neighbor, radius + 1):
                costs[neighbor] = next_cost
                heappush(frontier, (next_cost, neighbor))
    return costs


@dataclass
class Noise:
    origin: Vec2d
    radius: int
    remaining: float = 0.33
