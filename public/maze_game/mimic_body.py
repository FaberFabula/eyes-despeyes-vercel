"""Mimic size, swallowed cargo, and escape after a split."""

from dataclasses import dataclass
from math import pi, sqrt
from typing import TYPE_CHECKING

from .actor import Actor
from .effects import WeaponFlash
from .item import DEBUG_TOOL, GATE_KEY, MIRROR, AttackProfile, Gear
from .layout import TILE, Cell, center, find_path, tile_at
from .physics import ACTOR, ShapeFilter, Vec2d
from .progression import MimicAppearance

if TYPE_CHECKING:
    from .enemy import Enemy
    from .world import Maze

MIMIC_BASE_RADIUS = 21.0
MIMIC_FOLDED_RADIUS = 10.0
MIMIC_MAX_RADIUS = 44.0
MIMIC_MIN_SPLIT_RADIUS = 12.0
MIMIC_GEAR_GROWTH = 0.9
MIMIC_BURST_DELAY = 3.0
MIMIC_BURST_GRACE = 1.4
MIMIC_BURST = AttackProfile(0, 7, 0)
STOP = Vec2d(0, 0)


@dataclass
class SwallowedMob:
    actor: Actor
    collisions: ShapeFilter


@dataclass
class MimicCorpse:
    position: Vec2d
    radius: float
    bursts_at: float
    cargo: list[Gear | SwallowedMob]
    appearance: MimicAppearance
    ruptured: bool = False


def resize_mimic(enemy: "Enemy", maze: "Maze", radius: float) -> None:
    if enemy.radius != radius:
        enemy.radius = radius
        maze.physics.resize_actor(enemy.shape, radius)
        enemy.route_goal = None


def fold_mimic(enemy: "Enemy", maze: "Maze") -> None:
    resize_mimic(enemy, maze, min(enemy.full_radius, MIMIC_FOLDED_RADIUS))


def fit_mimic(enemy: "Enemy", maze: "Maze") -> None:
    """Expand only into free space."""
    if enemy.radius >= enemy.full_radius:
        return
    limit = enemy.full_radius + 1
    clearance = maze.physics.solid_clearance(enemy.position, limit)
    radius = enemy.full_radius if clearance >= limit else int(clearance) - 1
    if radius > enemy.radius:
        resize_mimic(enemy, maze, radius)


def set_mimic_size(enemy: "Enemy", maze: "Maze", radius: float, *, refill: bool = False) -> None:
    missing_hp = enemy.max_hp - enemy.hp
    stamina = enemy.stamina
    enemy.mimic_memory.radius = enemy.full_radius = radius
    size = radius / MIMIC_BASE_RADIUS
    enemy.speed = 175 + 24 * max(0, 1 - size)
    enemy.max_hp = round(40 * size ** 1.35)
    enemy.max_stamina = 100.0
    enemy.grip_multiplier = 0.75 + 0.55 * size
    enemy.strength_multiplier = 0.64 + 0.51 * size
    enemy.dexterity_multiplier = enemy.intelligence_multiplier = 1.0
    maze.scale_mob(enemy)
    if not refill:
        enemy.hp = max(1, enemy.max_hp - missing_hp)
        enemy.stamina = min(stamina, enemy.max_stamina)
    if enemy.radius > radius:
        resize_mimic(enemy, maze, radius)
    maze.physics.resize_actor(enemy.shape, enemy.radius, mass_radius=radius)


def mob_growth(actor: Actor) -> float:
    return 1/4 * min(6.0, 2.5 + actor.max_hp * 0.06)


def feed_growth(enemy: "Enemy", maze: "Maze", growth: float) -> None:
    enemy.nourishment += growth
    enemy.feed_at = maze.wielding.time + 1.0
    set_mimic_size(enemy, maze, min(MIMIC_MAX_RADIUS, enemy.full_radius + growth))
    maze.play_nearby_cue("mimic_feed", enemy.position, 5, 0.78)


def edible_gear(maze: "Maze", item: Gear) -> bool:
    return (item.spec not in (MIRROR, GATE_KEY, DEBUG_TOOL) and not item.temporary
            and item not in maze.item_actions.loads
            and not any(item in seal.captives for seal in maze.item_actions.seals))


def swallow_gear(enemy: "Enemy", maze: "Maze", item: Gear) -> None:
    maze.wielding.remove(item)
    maze.swallowed_gear.add(item)
    enemy.cargo.append(item)
    enemy.mimic_memory.appearance.fragments.append(item.spec)
    feed_growth(enemy, maze, MIMIC_GEAR_GROWTH)


def feed_on_actor(enemy: "Enemy", maze: "Maze", actor: Actor) -> None:
    enemy.mimic_memory.appearance.extra_eyes += 1
    feed_growth(enemy, maze, mob_growth(actor))


def swallow_mob(enemy: "Enemy", maze: "Maze", actor: Actor) -> None:
    actor.hp = max(1, (actor.max_hp + 1) // 2)
    actor.captured = True
    actor.sprinting = False
    actor.desired_velocity = actor.external_velocity = actor.body.velocity = STOP
    filter_before = maze.physics.suspend_collisions(actor.shape, categories=ACTOR)
    enemy.cargo.append(SwallowedMob(actor, filter_before))
    feed_on_actor(enemy, maze, actor)


def travel_mimic(enemy: "Enemy", maze: "Maze", goal: Vec2d, dt: float) -> None:
    offset = goal - enemy.position
    step = offset.normalized() * min(offset.length, TILE)
    ahead = enemy.position + step
    clearance = enemy.full_radius + 1
    if (enemy.state == "spider" and maze.clear_path(enemy.position, ahead, clearance)
            and not any(maze.physics.near_solid(enemy.position + step * fraction, clearance)
                        for fraction in (0.5, 1))):
        fit_mimic(enemy, maze)
    else:
        fold_mimic(enemy, maze)
    if maze.clear_path(enemy.position, goal, enemy.radius):
        enemy.move(offset if offset.length > 5 else STOP)
        if offset.length > 5:
            enemy.aim = offset.normalized()
    else:
        fold_mimic(enemy, maze)
        enemy.follow_route(maze, tile_at(goal), dt)


def refuge(enemy: "Enemy", maze: "Maze", away_from: Vec2d) -> Cell | None:
    start = tile_at(enemy.position)
    rooms = sorted(maze.layout.rooms,
                   key=lambda room: -center(room.middle).get_distance(away_from))
    fallback = None
    danger = min(2 * TILE, enemy.distance_to(away_from)) - TILE / 2
    for room in rooms:
        if room is maze.layout.gate_room or room.contains(start):
            continue
        route = find_path(start, room.middle, maze.current_tile)
        if not route:
            continue
        if fallback is None:
            fallback = room.middle
        if all(center(cell).get_distance(away_from) >= danger for cell in route[:3]):
            return room.middle
    return fallback


def split_mimic(enemy: "Enemy", maze: "Maze", source: Actor | None) -> bool:
    """Leave a corpse, then return whether a smaller form got away."""
    now = maze.wielding.time
    radius = enemy.full_radius
    survives = radius >= MIMIC_MIN_SPLIT_RADIUS
    survivor = max(8.0, radius * 0.49) if survives else 0.0
    corpse_radius = sqrt(radius ** 2 - survivor ** 2)
    memory = enemy.mimic_memory
    maze.mimic_corpses.append(MimicCorpse(enemy.position, corpse_radius, now + MIMIC_BURST_DELAY,
                                         enemy.cargo, memory.appearance))
    enemy.cargo = []
    memory.appearance = MimicAppearance()
    enemy.attack = enemy.prey = None
    maze.play_nearby_cue("mimic_split", enemy.position, 8)
    if not survives:
        return False

    away_from = source.position if source is not None and source is not enemy else maze.player.position
    enemy.state = "fleeing"
    enemy.flee_from = source if source is not enemy else None
    enemy.flee_origin = away_from
    enemy.destination = refuge(enemy, maze, away_from)
    enemy.flee_until = now + 3.0
    enemy.split_safe_until = now + 0.75
    enemy.physics_safe_until = now + MIMIC_BURST_GRACE
    enemy.attack_ready = enemy.flee_until
    enemy.wall_cells = ()
    enemy.hidden_for = 0.0
    enemy.dodge_time = 0.0
    enemy.desired_velocity = enemy.external_velocity = enemy.body.velocity = STOP
    enemy.route_goal = None
    set_mimic_size(enemy, maze, survivor, refill=True)
    fold_mimic(enemy, maze)
    return True


def flee_mimic(enemy: "Enemy", maze: "Maze", dt: float) -> bool:
    """Follow a walkable route away from the attacker before hunting again."""
    if enemy.state != "fleeing":
        return False
    source = enemy.flee_from
    threat = source.position if source is not None and source.active else enemy.flee_origin
    now = maze.wielding.time
    reached = enemy.destination is not None and enemy.distance_to(center(enemy.destination)) < 15
    far = enemy.distance_to(threat) > 8 * TILE and not maze.clear_path(threat, enemy.position)
    if now >= enemy.flee_until and (far or reached or enemy.destination is None or now >= enemy.flee_until + 7.0):
        enemy.state = "spider"
        enemy.flee_from = None
        enemy.destination = None
        enemy.hunt_at = now
        return False
    if enemy.destination is not None:
        fold_mimic(enemy, maze)
        enemy.follow_route(maze, enemy.destination, dt)
        enemy.move(enemy.desired_velocity, sprint=True)
    else:
        away = (enemy.position - threat).normalized()
        for turn in (0, pi / 4, -pi / 4, pi / 2, -pi / 2):
            direction = away.rotated(turn)
            if maze.clear_path(enemy.position, enemy.position + direction * TILE, enemy.radius):
                enemy.move(direction, sprint=True)
                enemy.aim = direction
                break
    enemy.facing = enemy.aim
    return True


def ejection_point(maze: "Maze", origin: Vec2d, direction: Vec2d, radius: float,
                   item: Gear | None) -> tuple[Vec2d, float]:
    size = (item.spec.length, item.spec.width) if item is not None else None
    angle = direction.angle
    for turn in (0, pi / 4, -pi / 4, pi / 2, -pi / 2, pi):
        axis = direction.rotated(turn)
        fits = maze.physics.placement_probe(radius, size, angle=axis.angle)
        for distance in (radius + 7, radius + 22, radius + 44, radius + 70):
            point = origin + axis * distance
            if maze.clear_path(origin, point) and fits(point):
                return point, axis.angle
    if item is not None:
        cell = tile_at(origin)
        if (aligned := maze.gear_angle(item.spec, cell)) is not None:
            return center(cell), aligned
    return origin, angle


def update_mimic_corpses(maze: "Maze") -> None:
    now = maze.wielding.time
    for corpse in tuple(maze.mimic_corpses):
        if corpse.ruptured or now + 1e-9 < corpse.bursts_at:
            continue
        corpse.ruptured = True
        blast_mimic_corpse(maze, corpse)
        angle = maze.rng.uniform(-pi, pi)
        for index, swallowed in enumerate(corpse.cargo):
            direction = Vec2d(1, 0).rotated(angle + index * 2.4)
            if isinstance(swallowed, SwallowedMob):
                actor = swallowed.actor
                actor.body.position, turn = ejection_point(maze, corpse.position, direction, actor.radius, None)
                actor.captured = False
                actor.physics_safe_until = now + MIMIC_BURST_GRACE
                maze.physics.restore_collisions(actor.shape, swallowed.collisions)
                actor.push(Vec2d(530, 0).rotated(turn))
            else:
                item = swallowed
                radius = max(item.spec.length, item.spec.width) / 2
                item.body.position, item.body.angle = ejection_point(maze, corpse.position, direction, radius, item)
                item.body.velocity = Vec2d(560, 0).rotated(item.body.angle)
                item.body.angular_velocity = maze.rng.uniform(-9, 9)
                item.physics_safe_until = now + MIMIC_BURST_GRACE
                maze.swallowed_gear.remove(item)
                maze.wielding.restore(item)
        corpse.cargo.clear()


def blast_mimic_corpse(maze: "Maze", corpse: MimicCorpse) -> None:
    reach = max(60, corpse.radius * 3.6)
    maze.make_cue("mimic_burst", corpse.position, 9)
    maze.weapon_flashes.append(WeaponFlash(corpse.position, reach, (215, 113, 132), maze.wielding.time))
    for actor in maze.actors:
        offset = actor.position - corpse.position
        if (not actor.active or actor.physics_safe_until > maze.wielding.time or offset.length >= reach
                or not maze.clear_path(corpse.position, actor.position)):
            continue
        force = 1 - 0.5 * offset.length / reach
        direction = offset.normalized() if offset.length else actor.aim
        maze.apply_hit(None, actor.shape, MIMIC_BURST, force, position=corpse.position,
                       shove=direction * (860 * force), weapon="Mimic burst")
