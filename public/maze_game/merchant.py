"""Merchant stock and shop state."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .actor import Actor
from .enemy import Enemy
from .item import EPIC_ITEMS, HEAL_PACK, RADAR, WEAPONS, healing_pack_hp
from .layout import Cell, center, tile_at
from .progression import MERCHANT_PRICES

if TYPE_CHECKING:
    from .world import Maze

MERCHANT_EPIC_CHANCE = 0.08
from .entity import Entity
from .item import Equipment
from .physics import Vec2d


@dataclass(eq=False)
class Merchant(Entity):
    actor: Actor
    stock: list[Equipment]
    next_help: float = 0.0
    next_gear: float = 0.0

    @property
    def position(self) -> Vec2d:
        return self.actor.position


def spawn_merchant(maze: "Maze", cell: Cell, *, can_be_hostile: bool = True) -> Merchant:
    stock = [HEAL_PACK, RADAR, *maze.rng.sample(WEAPONS, 3)]
    if maze.rng.random() < MERCHANT_EPIC_CHANCE:
        stock[-1] = maze.rng.choice(EPIC_ITEMS)
    actor = Enemy(maze.physics, cell, speed=96, hp=16, kind="merchant",
                  patrol=(cell,), can_be_hostile=can_be_hostile)
    merchant = Merchant(actor, stock)
    maze.add_actor(actor)
    maze.merchants.append(merchant)
    maze.merchants_by_actor[actor] = merchant
    return merchant


def merchant_drop_cell(maze: "Maze", merchant: Merchant, spec: Equipment) -> Cell | None:
    origin = tile_at(merchant.position)
    ox, oy = origin
    cells = [(ox + dx, oy + dy) for dx, dy in
             ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1))]
    return next((cell for cell in cells if maze.spawn_clear(cell)
                 and maze.clear_path(merchant.position, center(cell))
                 and maze.gear_angle(spec, cell) is not None), None)


def buy_from_merchant(maze: "Maze", index: int) -> bool:
    merchant = maze.shop
    if merchant is None or not 0 <= index < len(merchant.stock):
        return False
    spec = merchant.stock[index]
    cell = merchant_drop_cell(maze, merchant, spec)
    price = MERCHANT_PRICES[spec]
    if cell is None or maze.progress.xp < price or maze.place_gear(spec, cell) is None:
        return False
    maze.progress.xp -= price
    merchant.stock.pop(index)
    return True


def provoke_merchant(maze: "Maze", merchant: Merchant, aggressor: Actor) -> None:
    actor = merchant.actor
    if not actor.can_be_hostile:
        return
    if maze.shop is merchant:
        maze.shop = None
    guard = maze.guard_for(actor)
    if guard is None:
        guard = actor
        maze.guards.append(guard)
        guard.patrol = (tile_at(guard.position),)
    if aggressor is maze.player:
        maze.wake_guard(guard)
        guard.sense(maze, 0)
    arm_merchant(maze, merchant, guard)


def arm_merchant(maze: "Maze", merchant: Merchant, guard: Enemy) -> None:
    actor = merchant.actor
    now = maze.wielding.time
    if now < merchant.next_gear or not merchant.stock or actor.stamina <= 0:
        return
    wants_heal = actor.max_hp - actor.hp >= healing_pack_hp(actor.max_hp) and HEAL_PACK in merchant.stock
    if actor in maze.wielding.held and not wants_heal:
        return
    if wants_heal:
        choices = [HEAL_PACK]
    elif guard.rival is None and not guard.knows_player and RADAR in merchant.stock and not guard.alert.visible:
        choices = [RADAR, *(spec for spec in merchant.stock if spec.attack is not None)]
    else:
        choices = [spec for spec in merchant.stock if spec.attack is not None]
    for spec in choices:
        cell = merchant_drop_cell(maze, merchant, spec)
        if cell is None:
            continue
        item = maze.place_gear(spec, cell)
        if item is None:
            continue
        maze.wielding.release(actor)
        if not maze.wielding.grab(actor, item, item.point_query(actor.position).point):
            maze.wielding.remove(item)
            return
        merchant.stock.remove(spec)
        merchant.next_gear = now + 0.65
        if spec in (HEAL_PACK, RADAR):
            maze.item_actions.attack_held(actor, maze.wielding.held[actor], charge=1.0)
            maze.wielding.release(actor)
        return
