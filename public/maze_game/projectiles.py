"""Move shots, resolve swept contacts, and apply projectile effects."""

from typing import TYPE_CHECKING

from .physics import SweepHit, Vec2d
from .effects import GravityWell, WeaponFlash
from .physics import ACTOR, SOLID
from .item import AttackProfile
from .item_physics import Projectile
from .physics import GEAR_CATEGORY, SPIKE_CATEGORY

if TYPE_CHECKING:
    from .world import Maze

PROJECTILE_MASK = SOLID | ACTOR | GEAR_CATEGORY | SPIKE_CATEGORY
FIRE_ATTACK = AttackProfile(0, 1, 0, "heat")


def move_projectiles(maze: "Maze", dt: float) -> None:
    active = []
    for shot in maze.projectiles:
        if shot.spike is not None:
            if advance_spike(maze, shot, dt):
                active.append(shot)
            continue
        if shot.target is not None and shot.target.active:
            toward = shot.target.position - shot.position
            if toward.length:
                shot.direction = toward.normalized()
        distance = min(shot.remaining, dt * shot.speed)
        end = shot.position + shot.direction * distance
        contact = resolve_shot(maze, shot, end)
        if contact is not None:
            continue
        if shot.remaining > distance:
            shot.position = end
            shot.remaining -= distance
            active.append(shot)
        elif shot.effect == "gravity":
            open_gravity_well(maze, shot, end)
        elif shot.effect == "molotov":
            explode_molotov(maze, shot, end)
    maze.projectiles = active


def shot_contacts(maze: "Maze", shot: Projectile, end: Vec2d):
    radius = 9 if shot.effect == "pulse" else shot.radius
    hits = maze.physics.sweep(shot.position, end, radius, PROJECTILE_MASK)
    return (hit for hit in hits if hit.shape not in (shot.owner.shape, shot.emitter))


def shot_hit(maze: "Maze", shot: Projectile, end: Vec2d):
    return min(shot_contacts(maze, shot, end), key=lambda hit: hit.alpha, default=None)


def advance_spike(maze: "Maze", shot: Projectile, dt: float) -> bool:
    spike = shot.spike
    if maze.wielding.time >= spike.expires or spike.hp <= 0:
        maze.wielding.release_spike(spike)
        return False
    if spike.resting:
        if (shape := spike.embedded) is not None:
            actor = maze.actors_by_shape.get(shape)
            if maze.physics.contains(shape) and (actor is None or actor.active):
                tumble = actor.tumble_angle if actor is not None else 0
                shot.position = shape.body.local_to_world(spike.anchor.rotated(tumble))
                shot.direction = spike.local_direction.rotated(shape.body.angle + tumble)
                collider = spike.collider
                collider.body.position, collider.body.angle = shot.position, shot.direction.angle
                maze.physics.refresh_shape(collider)
            else:
                spike.embedded = None
                maze.wielding.release_spike_tow(spike)
        return True

    distance = min(shot.remaining, dt * shot.speed)
    end = shot.position + shot.direction * distance
    # A lance resolves every new contact in travel order, stopping at the first solid surface.
    for hit in sorted(shot_contacts(maze, shot, end), key=lambda contact: contact.alpha):
        shape = hit.shape
        if not maze.physics.contains(shape) or shape.body in spike.struck:
            continue
        solid = bool(shape.filter.categories & SOLID)
        room_eye = shape in maze.eyes_by_shape
        apply_shot_hit(maze, shot, hit, force=0.56 if solid and spike.piercing else 1.0)
        if spike.piercing and not solid:
            if room_eye:
                continue
            spike.struck.add(shape.body)
            maze.wielding.load_spike(shot, shape, hit.point)
            continue
        actor = maze.actors_by_shape.get(shape)
        if maze.physics.contains(shape) and (actor is not None or solid and shape not in maze.doors_by_shape):
            spike.embedded = shape
            tumble = actor.tumble_angle if actor is not None else 0
            spike.anchor = shape.body.world_to_local(hit.point).rotated(-tumble)
            spike.local_direction = shot.direction.rotated(-shape.body.angle - tumble)
        shot.position = hit.point
        spike.expires = maze.wielding.time + (7 if spike.piercing else 3.5)
        maze.wielding.lodge_spike(shot)
        return True

    shot.remaining -= distance
    shot.position = end
    if shot.remaining <= 0:
        maze.wielding.release_spike_tow(spike)
        spike.expires = maze.wielding.time + (7 if spike.piercing else 3.5)
        maze.wielding.lodge_spike(shot)
        return True
    if spike.tow is not None:
        spike.tow.position = end
        spike.tow.velocity = shot.direction * min(shot.speed, shot.remaining / dt)
    return True


def resolve_shot(maze: "Maze", shot: Projectile, end: Vec2d) -> Vec2d | None:
    hit = shot_hit(maze, shot, end)
    if hit is None:
        return None
    if shot.effect == "gravity":
        open_gravity_well(maze, shot, hit.point + hit.normal * 4)
        shot_noise(maze, shot, hit)
    else:
        apply_shot_hit(maze, shot, hit)
        if shot.effect == "molotov":
            explode_molotov(maze, shot, hit.point + hit.normal * 5)
            if (victim := maze.actors_by_shape.get(hit.shape)) is not None:
                maze.ignite_actor(victim, shot.owner, shot.power, weapon=shot.weapon or "Fire")
    return hit.point


def shot_noise(maze: "Maze", shot: Projectile, hit: SweepHit) -> None:
    if shot.owner is maze.player:
        maze.make_noise(hit.point + hit.normal * 4, 9 if shot.mode == "magic" else 8, shot.position)


def apply_shot_hit(maze: "Maze", shot: Projectile, hit: SweepHit, *, force: float = 1.0) -> None:
    shot_noise(maze, shot, hit)
    delivery = "magic" if shot.mode == "magic" else "projectile"
    if (item := maze.wielding.by_shape.get(hit.shape)) is not None:
        item.body.apply_impulse_at_world_point(shot.direction * (shot.shove * 0.5), hit.point)
        maze.wielding.attribute_motion(item.shape, shot.owner, 0.9, delivery=delivery, weapon=shot.weapon)
    maze.apply_hit(shot.owner, hit.shape, shot.attack, force * shot.power, position=hit.point,
                   shove=shot.direction * shot.shove, delivery=delivery, weapon=shot.weapon)


def open_gravity_well(maze: "Maze", shot: Projectile, position: Vec2d) -> None:
    charge = shot.charge
    maze.gravity_wells.append(GravityWell(position, shot.owner, maze.wielding.time + 1.0 + 1.7 * charge,
                                         radius=78 + 135 * charge, strength=(0.58 + 1.02 * charge) * shot.power))


def explode_molotov(maze: "Maze", shot: Projectile, position: Vec2d) -> None:
    radius = 56
    maze.spread_fire(position, shot.owner, radius, shot.power, weapon=shot.weapon or "Fire")
    maze.weapon_flashes.append(WeaponFlash(position, radius, (251, 145, 55), maze.wielding.time))


def spray_flame(maze: "Maze", shot: Projectile) -> None:
    struck = set()
    start = shot.position
    emitter = maze.wielding.by_shape.get(shot.emitter)
    now = maze.wielding.time
    weapon = shot.weapon or "Fire"
    for spread in (-0.19, -0.10, 0.0, 0.10, 0.19):
        direction = shot.direction.rotated(spread)
        end = start + direction * shot.remaining
        hit = shot_hit(maze, shot, end)
        if hit is not None:
            end = hit.point - direction * 5
            if hit.shape not in struck:
                struck.add(hit.shape)
                if (victim := maze.actors_by_shape.get(hit.shape)) is not None:
                    maze.ignite_actor(victim, shot.owner, shot.power, weapon=weapon)
                elif emitter is not None and now >= emitter.impact_times.get(hit.shape, 0):
                    emitter.impact_times[hit.shape] = now + 0.46
                    maze.apply_hit(shot.owner, hit.shape, shot.attack, shot.power, position=hit.point,
                                   delivery="projectile", weapon=shot.weapon)
        maze.flame_trails.append((start, end, now + 0.14))
        if spread in (-0.19, 0, 0.19) and start.get_distance(end) > 12:
            maze.ignite(end, shot.owner, shot.power, weapon=weapon)
            if spread == 0:
                maze.ignite((start + end) / 2, shot.owner, shot.power, weapon=weapon)
