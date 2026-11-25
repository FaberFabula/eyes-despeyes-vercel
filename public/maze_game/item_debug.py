"""De Bug's sandbox menus and summons, using the campaign's entity constructors."""

from typing import TYPE_CHECKING


from .merchant import spawn_merchant
from .physics import Vec2d
from .item import DEBUG_TOOL, EPIC_ITEMS, GATE_KEY, HEAL_PACK, LOOT, MIRROR, TORCH, Equipment
from .actor import Actor
from .layout import COLS, ROWS, TILE, Cell, center, tile_at
from .item_debug_effects import DebugEffects, EFFECT_OPTIONS

if TYPE_CHECKING:
    from .world import Maze

GOLD = (240, 201, 91)
SUMMON_ITEMS = (*LOOT, TORCH, HEAL_PACK, GATE_KEY, MIRROR, *EPIC_ITEMS, DEBUG_TOOL)
SUMMON_MOBS = (
    ("Eye Scout", "scout"), ("Claw Eye", "claw"), ("Room Eye Guard", "guard"),
    ("Armored Eye", "armored"), ("Eye Elite", "elite"), ("Eye Scribe", "scribe"),
    ("Mimic", "mimic"), ("Merchant", "merchant"), ("Eye Elite Boss", "elite_boss"),
    ("Eye Scribe Boss", "scribe_boss"), ("Target Dummy", "dummy"),
)
MOB_NAMES = {kind: label for label, kind in SUMMON_MOBS}
EFFECT_PAGE_SIZE = 20


class SandboxTools:
    def __init__(self, maze: "Maze"):
        self.maze = maze
        self.page: str | None = None
        self.hostility_chance = 0.0
        self.dragging_slider = False
        self.mouse_buttons: set[int] = set()
        self.effect_page = 0
        self.effects = DebugEffects(maze)
        self.origin = Vec2d(0, 0)
        self.message = ""

    @property
    def is_open(self) -> bool:
        return self.page is not None

    def open(self, page: str, cursor: Vec2d | None = None) -> bool:
        maze = self.maze
        held = maze.item_actions.held_by(maze.player)
        if (not maze.sandbox or page not in ("items", "mobs", "effects") or held is None
                or held.item.spec is not DEBUG_TOOL):
            return False
        if not self.is_open:
            toward = cursor - maze.player.position if cursor is not None else maze.player.aim
            direction = toward.normalized() if toward.length else maze.player.aim
            distance = min(4 * TILE, max(2 * TILE, toward.length))
            self.origin = maze.player.position + direction * distance
        self.page = page
        self.message = ""
        maze.item_actions.clear_input(maze.player)
        return True

    def close(self) -> None:
        self.page = None
        self.dragging_slider = False

    def set_hostility_chance(self, value: float) -> None:
        # Only spawn_mob reads this value. Existing actors keep their own permission.
        self.hostility_chance = round(max(0.0, min(1.0, value)) * 20) / 20

    def spawn_cell(self, spec: Equipment | None = None) -> Cell | None:
        maze = self.maze
        x, y = tile_at(self.origin)
        cells = [(tx, ty) for tx in range(max(1, x - 5), min(COLS - 1, x + 6))
                 for ty in range(max(1, y - 5), min(ROWS - 1, y + 6))]
        cells.sort(key=lambda cell: center(cell).get_dist_sqrd(self.origin))
        gap = max(26, spec.length * 0.7) if spec is not None else 52
        for cell in cells:
            position = center(cell)
            if not maze.spawn_clear(cell, gap) or not maze.player_sees(position):
                continue
            if spec is not None:
                if maze.gear_angle(spec, cell) is not None:
                    return cell
            elif not maze.physics.near_solid(position, 24):
                return cell
        return None

    def spawn_item(self, spec: Equipment) -> bool:
        if not self.maze.sandbox or self.page != "items" or spec not in SUMMON_ITEMS:
            return False
        cell = self.spawn_cell(spec)
        if cell is None:
            self.message = "No clear space. Close the menu and move to a room."
            return False
        item = self.maze.place_gear(spec, cell)
        if item is None:
            return False
        self.maze.gear_budget = max(self.maze.gear_budget, len(self.maze.wielding.items) + 2)
        self.message = f"Summoned {spec.name}"
        return True

    def spawn_mob(self, kind: str) -> Actor | None:
        maze = self.maze
        if not maze.sandbox or self.page != "mobs" or kind not in MOB_NAMES:
            return None
        cell = self.spawn_cell()
        if cell is None:
            self.message = "No clear space. Close the menu and move to a room."
            return None
        # Roll exactly once for this actor. A later menu change cannot alter it.
        permitted = maze.rng.random() < self.hostility_chance
        if kind == "mimic":
            actor = maze.spawn_mimic(cell, can_be_hostile=permitted)
        elif kind == "merchant":
            actor = spawn_merchant(maze, cell, can_be_hostile=permitted).actor
        elif kind == "scribe":
            actor = maze.add_scribe(cell, (cell,), can_be_hostile=permitted)
        elif kind == "dummy":
            actor = Actor(maze.physics, cell, speed=0, hp=80, organic=False, radius=13, can_be_hostile=False)
            maze.targets.append(actor)
            maze.add_actor(actor)
        else:
            boss = kind.endswith("_boss")
            actor = maze.spawn_guard(kind.removesuffix("_boss"), cell, boss=boss, can_be_hostile=permitted)
        name = MOB_NAMES[kind]
        disposition = "can become hostile" if actor.can_be_hostile else "passive"
        self.message = f"Summoned {name} · {disposition}"
        return actor

    def spawn_effect(self, kind: str) -> bool:
        if not self.maze.sandbox or self.page != "effects" or not any(key == kind for _, key in EFFECT_OPTIONS):
            return False
        if kind == "seal" and self.maze.item_actions.seals:
            self.message = "Close De Bug and wait for the current seal to disappear."
            return False
        position = self.origin
        hit = self.maze.hit_solid(self.maze.player.position, position)
        if hit is not None:
            position = hit.point + hit.normal * 25
        if not self.effects.trigger(kind, position):
            self.message = "Aim into clear space or toward a target, then reopen De Bug."
            return False
        self.close()
        return True
