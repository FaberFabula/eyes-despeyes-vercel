"""Persistent item effects shared by the player and enemies."""

from dataclasses import dataclass, field, replace
from math import pi
from typing import TYPE_CHECKING

from shapely.geometry import Point, Polygon

from .physics import ShapeFilter, SweepHit, Vec2d
from .effects import Beam, GravityWell, WeaponFlash
from .physics import ACTOR, SOLID
from .actor import Actor
from .item import AttackProfile
from .layout import TILE, center, tile_at
from .physics import GEAR_CATEGORY, SPIKE_CATEGORY
from .item_physics import LAUNCHER_IMPACT, HeldGear
from .item import Gear

if TYPE_CHECKING:
    from .item_actions import ItemActions

EPIC_PURPLE = (185, 113, 243)
MAGIC_MASK = ACTOR | SOLID | GEAR_CATEGORY | SPIKE_CATEGORY
STOP = Vec2d(0, 0)
GRIMOIRE_PAGES = 6
GRIMOIRE_COOLDOWN = 2.8
GRIMOIRE_COST = AttackProfile(0, 1, 0, "arcane")
DRAIN_RADIUS = 43
SEAL_LIMIT = 6
SEAL_SECONDS_PER_MARK = 1.0
SEAL_INTERVAL = 0.46
SEAL_TIGHTEN_TIME = 0.55
SEAL_COLLAPSE_TIME = 0.5
SEAL_SLACK = 8
SEAL_STRETCH = 18
SEAL_PADDING = 1
SEAL_SQUEEZE_SPEED = 105
SEAL_VERTEX_HIT = AttackProfile(0, 2, 0, "arcane")
WHIP_LINKS = 7
LAUNCHER_CAPTURE_REACH = 172
LAUNCHER_ITEM_HOLD = 12.0
LAUNCHER_ITEM_SPEED = 1460
LAUNCHER_ITEM_SPEED_CAP = 1760
LAUNCHER_HOLD = {"mimic": 7, "elite": 9, "armored": 9, "guard": 11}


@dataclass
class DrainField:
    owner: Actor
    position: Vec2d
    expires: float
    next_tick: float
    power: float
    healing_credit: int = 0


@dataclass
class SealField:
    owner: Actor
    power: float
    marks: list[Vec2d]
    normals: list[Vec2d]
    captives: dict[Actor | Gear, Vec2d]
    origin: Vec2d
    squeeze_center: Vec2d
    started: float
    expires: float
    next_damage: float
    outline: list[Vec2d] = field(default_factory=list)
    collapse: tuple[float, list[Vec2d]] | None = None


@dataclass
class Tail:
    points: list[Vec2d]
    angles: list[float]
    turns: list[float]


@dataclass
class WhipMotion:
    tails: list[Tail] = field(default_factory=list)
    count: int = 1
    extension: float = 0.0
    extension_seconds: float = 0.0


@dataclass
class LauncherLoad:
    captive: Actor | Gear
    expires: float
    fire_at: float
    origin: Vec2d
    shape_filter: ShapeFilter | None = None


def ordered_vertices(points: list[Vec2d]) -> list[Vec2d]:
    center = sum(points, STOP) / len(points)
    return sorted(points, key=lambda point: ((point - center).angle, point.get_dist_sqrd(center)))


def seal_normals(points: list[Vec2d], origin: Vec2d) -> list[Vec2d]:
    """Find outward normals for the placed sides of the seal."""
    if len(points) < 3:
        return []
    ring = list(zip(points, (*points[1:], points[0])))
    normals = [-(b - a).perpendicular_normal() for a, b in ring]
    turns = [a.get_angle_between(b) for a, b in zip(normals, (*normals[1:], normals[0]))]
    if not all(pi / 24 <= turn <= 5 * pi / 6 for turn in turns):
        # Even spacing keeps every corner for concave or very narrow casts.
        direction = max(normals, key=lambda normal: (points[0] - origin).dot(normal))
        normals = [direction.rotated(index * 2 * pi / len(points)) for index in range(len(points))]
    return normals


def seal_surface(target: Actor | Gear) -> list[tuple[Vec2d, float]]:
    """World points and collision radii that define the full footprint."""
    if isinstance(target, Actor):
        return [(target.body.local_to_world(target.shape.offset), target.shape.radius)]
    points = [(target.body.local_to_world(vertex), target.shape.radius) for vertex in target.shape.get_vertices()]
    for jaw in target.jaws:
        points.extend((target.body.local_to_world(vertex), jaw.radius) for vertex in (jaw.a, jaw.b))
    return points


def gear_seal_normals(
    item: Gear, footprint: list[tuple[Vec2d, float]], count: int,
) -> list[Vec2d]:
    """Fit the same number of sides to the item's rotating collision footprint."""
    axis = item.axis
    across = axis.perpendicular()
    if count == 3:
        extents = []
        for direction in (axis, across):
            projections = [(point.dot(direction), radius) for point, radius in footprint]
            extent = max(distance + radius for distance, radius in projections)
            extent -= min(distance - radius for distance, radius in projections)
            extents.append(extent + 2 * SEAL_PADDING)
        length, width = extents
        if width > length:
            axis, across = across, -axis
            length, width = width, length
        # Use a long side as the base. The other two sides meet above its center.
        return [-across, (across * length + axis * (2 * width)).normalized(),
                (across * length - axis * (2 * width)).normalized()]

    # Four strands follow the box axes. Extra strands bevel opposite corners.
    normals = [axis, across, -axis, -across]
    if count >= 5:
        normals.insert(1, (axis + across).normalized())
    if count == 6:
        normals.insert(4, (-axis - across).normalized())
    return normals


def visible_target(actions: "ItemActions", actor: Actor, item: Gear, cursor: Vec2d | None, *, clamp: bool) -> Vec2d | None:
    maze = actions.maze
    reach = 12 * TILE if actor is maze.player else item.spec.attack.reach
    origin = actor.position
    if cursor is None:
        cursor = maze.player.position if actor is not maze.player else origin + actor.aim * reach

    def visible(point: Vec2d) -> bool:
        if actor is maze.player:
            return maze.player_sees(point)
        return maze.clear_path(origin, point)

    offset = cursor - origin
    if offset.length <= reach and visible(cursor):
        return cursor
    if not clamp:
        return None
    direction = offset.normalized() if offset.length else actor.aim
    distance = min(offset.length, reach)
    end = origin + direction * distance
    if (obstruction := maze.hit_solid(origin, end)) is not None:
        distance = max(0, distance * obstruction.alpha - 2)
    # A lit patch may follow darkness on this ray, so search inward from its far edge.
    while distance > 0:
        point = origin + direction * distance
        if visible(point):
            return point
        distance -= 2
    return origin


def can_use_advanced_special(actions: "ItemActions", item: Gear) -> bool:
    if actions.now < item.special_ready or item.shape not in actions.maze.wielding.by_shape:
        return False
    effect = item.spec.effect
    if effect == "grimoire":
        return True
    if effect == "seal":
        return not actions.seals and bool(actions.marks.get(item))
    if effect == "launcher":
        return item not in actions.loads
    return effect == "whip"


def use_advanced_primary(actions: "ItemActions", actor: Actor, item: Gear, cursor: Vec2d | None) -> bool:
    effect = item.spec.effect
    if effect == "grimoire":
        return cast_grimoire(actions, actor, item, cursor)
    if effect == "seal":
        return place_seal_mark(actions, actor, item, cursor)
    return fire_launcher(actions, actor, item)


def use_advanced_special(actions: "ItemActions", actor: Actor, held: HeldGear, cursor: Vec2d | None) -> bool:
    item = held.item
    if not can_use_advanced_special(actions, item):
        return False
    effect = item.spec.effect
    if effect == "grimoire":
        actions.book_streaks.pop(item, None)
        item.special_ready = actions.now + GRIMOIRE_COOLDOWN
        actions.mark_use(actor, item)
    elif effect == "seal":
        return bind_seal(actions, actor, item)
    elif effect == "whip":
        held.whip_extending = True
    elif effect == "launcher":
        return load_launcher(actions, actor, item, cursor)
    return True


def cast_grimoire(actions: "ItemActions", actor: Actor, item: Gear, cursor: Vec2d | None) -> bool:
    if actions.now < item.special_ready:
        return False
    maze = actions.maze
    position = visible_target(actions, actor, item, cursor, clamp=True)
    actions.mark_use(actor, item)
    actor.attack_cooldown = item.spec.attack.recovery
    actions.book_streaks[item] = actions.book_streaks.get(item, 0) + 1
    grimoire_cost(actions, actor, item)
    if actor.alive:
        roll = maze.rng.randrange(5)
        if roll == 0:
            grimoire_blast(actions, actor, item, position)
        elif roll == 1:
            actions.drains.append(DrainField(actor, position, actions.now + 3.1, actions.now,
                                     actor.intelligence_multiplier))
        elif roll == 2:
            grimoire_cost(actions, actor, item)
            maze.ignite_actor(actor, actor, actor.intelligence_multiplier)
            maze.weapon_flashes.append(WeaponFlash(actor.position, 33, (247, 107, 93), actions.now))
        elif roll == 3:
            grimoire_lightning(actions, actor, item, position)
        else:
            maze.gravity_wells.append(GravityWell(position, actor, actions.now + 2.8, 140,
                                                 actor.intelligence_multiplier, item.spec.name))
    if actions.book_streaks[item] >= GRIMOIRE_PAGES:
        grimoire_blast(actions, actor, item, item.position, overload=True)
        if actor.alive and actor.position.get_distance(item.position) <= 107:
            maze.apply_hit(actor, actor.shape, AttackProfile(0, 5, 0, "heat"),
                           delivery="magic", weapon=item.spec.name, unavoidable=True)
            maze.ignite_actor(actor, actor, actor.intelligence_multiplier)
        actions.book_streaks[item] = 0
        item.special_ready = actions.now + GRIMOIRE_COOLDOWN
    return True


def grimoire_blast(actions: "ItemActions", actor: Actor, item: Gear, position: Vec2d, *, overload: bool = False) -> None:
    radius, damage, shove = (107, 18, 1110) if overload else (83, 13, 940)
    actions.radial(actor, item, position, radius, damage, shove, (255, 157, 67), kind="heat")
    actions.maze.spread_fire(position, actor, radius, actor.intelligence_multiplier, weapon=item.spec.name)


def grimoire_lightning(actions: "ItemActions", actor: Actor, item: Gear, position: Vec2d) -> None:
    maze = actions.maze
    targets = list(maze.enemies_of(actor, disguised=True))
    for damage in (10, 8, 6, 4):
        nearby = [victim for victim in targets if victim.position.get_distance(position) <= 78
                  and maze.clear_path(position, victim.position)]
        if not nearby:
            break
        victim = min(nearby, key=lambda target: target.position.get_dist_sqrd(position))
        route = victim.position - position
        bend = route.normalized().perpendicular() * 6
        bolt = [position, position + route / 3 + bend, position + route * 2 / 3 - bend, victim.position]
        maze.beams.extend(Beam(a, b, actions.now + 0.18, width=4, slashing=True)
                          for a, b in zip(bolt, bolt[1:]))
        targets.remove(victim)
        direction = (victim.position - position).normalized()
        position = victim.position
        maze.apply_hit(actor, victim.shape, AttackProfile(0, damage, 0, "arcane"),
                       actor.intelligence_multiplier, position=position, shove=direction * 270,
                       delivery="magic", weapon=item.spec.name)
    maze.weapon_flashes.append(WeaponFlash(position, 19, EPIC_PURPLE, actions.now))


def grimoire_cost(actions: "ItemActions", actor: Actor, item: Gear) -> None:
    actions.maze.apply_hit(actor, actor.shape, GRIMOIRE_COST, delivery="magic",
                        weapon=item.spec.name, unavoidable=True)


def update_drains(actions: "ItemActions") -> None:
    maze = actions.maze
    actions.drains = [drain for drain in actions.drains if drain.expires > actions.now]
    for drain in actions.drains:
        if actions.now < drain.next_tick:
            continue
        drain.next_tick += 0.50
        attack = AttackProfile(DRAIN_RADIUS, 2, 0, "arcane")
        for victim in maze.enemies_of(drain.owner, disguised=True):
            if (victim.position.get_distance(drain.position) <= DRAIN_RADIUS + victim.radius
                    and maze.clear_path(drain.position, victim.position)):
                drain.healing_credit += maze.apply_hit(
                    drain.owner, victim.shape, attack, drain.power,
                    position=victim.position, delivery="magic", weapon="Chaotic Grimoire",
                )
        healing, drain.healing_credit = divmod(drain.healing_credit, 3)
        if drain.owner.alive:
            drain.owner.hp = min(drain.owner.max_hp, drain.owner.hp + healing)


def place_seal_mark(actions: "ItemActions", actor: Actor, item: Gear, cursor: Vec2d | None) -> bool:
    if actions.seals:
        return False
    marks = actions.marks.setdefault(item, [])
    position = visible_target(actions, actor, item, cursor, clamp=False)
    if position is None or len(marks) == SEAL_LIMIT or any(mark.get_distance(position) < 12 for mark in marks):
        return False
    if len(marks) >= 2:
        polygon = Polygon(ordered_vertices([*marks, position]))
        if not polygon.is_valid or polygon.area < 52:
            return False
    marks.append(position)
    item.activated_at = actions.now
    actor.attack_cooldown = item.spec.attack.recovery
    return True


def bind_seal(actions: "ItemActions", actor: Actor, item: Gear) -> bool:
    if actions.seals or not actions.marks.get(item):
        return False
    points = ordered_vertices(actions.marks.pop(item))
    origin = sum(points, STOP) / len(points)
    targets = [target for target in actions.maze.actors if target.active and not target.shape.sensor]
    targets.extend(gear for gear in actions.maze.wielding.items if not gear.temporary)
    if len(points) >= 3:
        polygon = Polygon(points)
        enclosed = [target for target in targets if polygon.covers(Point(target.position))]
    else:
        # A point or line can touch a collider even though it encloses no area.
        enclosed = []
        for target in targets:
            shapes = (target.shape,) if isinstance(target, Actor) else target.shapes
            if len(points) == 1:
                touched = any(shape.point_query(points[0]).distance <= 0 for shape in shapes)
            else:
                touched = any(shape.segment_query(points[0], points[1]) is not None for shape in shapes)
            if touched:
                enclosed.append(target)
    captives = {target: target.position for target in enclosed}
    squeeze_center = sum(captives.values(), STOP) / len(captives) if captives else origin
    expires = actions.now + SEAL_SECONDS_PER_MARK * len(points)
    seal = SealField(actor, actor.intelligence_multiplier, points, seal_normals(points, origin),
                captives, origin, squeeze_center, actions.now, expires, actions.now + 0.24, outline=points[:])
    if captives:
        fit_seal(actions, seal)
    else:
        collapse_empty_seal(actions, seal)
    actions.seals.append(seal)
    actions.mark_use(actor, item)
    item.special_ready = actions.now + 0.42
    return True


def enemy_seal(actions: "ItemActions", actor: Actor, item: Gear, target: Vec2d) -> None:
    if actions.seals or actor.attack_cooldown > 0 or item in actions.enemy_seals:
        return
    offset = target - actor.position
    for radius in (36, 24):
        points = [target + Vec2d(radius, 0).rotated(offset.angle + index * 2 * pi / 3)
                  for index in range(3)]
        if all(visible_target(actions, actor, item, point, clamp=False) is not None for point in points):
            actions.marks[item] = points
            actions.enemy_seals[item] = (actor, actions.now + 0.52)
            actor.attack_cooldown = 1.45
            return
    actor.attack_cooldown = 0.35


def squeeze_captives(actions: "ItemActions", seal: SealField, dt: float) -> None:
    if len(seal.captives) < 2:
        return
    for target, anchor in seal.captives.items():
        offset = seal.squeeze_center - target.position
        distance = offset.length
        if distance < anchor.get_distance(seal.squeeze_center):
            # The tether takes up slack as the body moves toward the pile.
            seal.captives[target] = target.position
        if not distance:
            continue
        axis = offset / distance
        is_actor = isinstance(target, Actor)
        velocity = target.desired_velocity + target.external_velocity if is_actor else target.body.velocity
        speed = min(SEAL_SQUEEZE_SPEED, distance * 6)
        push = axis * max(0, speed - velocity.dot(axis)) * min(1, 12 * dt)
        # Push the actual bodies and let their collisions decide how tightly they pack.
        if is_actor:
            target.push(push)
        else:
            target.body.velocity += push


def constrain_captives(actions: "ItemActions", seal: SealField, dt: float) -> None:
    maze = actions.maze
    for target, anchor in seal.captives.items():
        body = target.body
        offset = body.position - anchor
        distance = offset.length
        if distance <= SEAL_SLACK:
            continue
        direction = offset.normalized()
        # A taut seal only pulls back against movement that stretches it.
        speed = min(160, (distance - SEAL_SLACK) * 13)
        is_actor = isinstance(target, Actor)
        velocity = target.desired_velocity + target.external_velocity if is_actor else body.velocity
        change = max(0, velocity.dot(direction) + speed) * min(1, 12 * dt)
        if distance > SEAL_STRETCH:
            change = max(change, velocity.dot(direction))
            translation = -direction * (distance - SEAL_STRETCH)
            if all(maze.clear_path(point, point + translation, max(0, radius - 0.5))
                   for point, radius in seal_surface(target)):
                body.position += translation
                maze.physics.refresh_body(body)
        body.velocity -= direction * min(change, max(0, body.velocity.dot(direction) + speed))
        if is_actor:
            target.external_velocity -= direction * change


def fit_seal(actions: "ItemActions", seal: SealField) -> None:
    normals = seal.normals
    if not normals:
        return

    progress = min(1, (actions.now - seal.started) / SEAL_TIGHTEN_TIME)
    slack = 1 - progress
    footprint = [surface for target in seal.captives for surface in seal_surface(target)]
    if len(seal.captives) == 1:
        target = next(iter(seal.captives))
        first = normals[0]
        if isinstance(target, Gear):
            fitted = gear_seal_normals(target, footprint, len(normals))
            # Match sides in order so turning them keeps a convex enclosure.
            index = min(range(len(fitted)), key=lambda i: abs(first.get_angle_between(fitted[i])))
            fitted = fitted[index:] + fitted[:index]
        else:
            # Equal spacing tucks in long corners around a single round collider.
            step = 2 * pi / len(normals)
            fitted = [first.rotated(index * step) for index in range(len(normals))]
        facing = first.rotated(progress * first.get_angle_between(fitted[0]))
        old_angles = [first.get_angle_between(normal) % (2 * pi) for normal in normals]
        new_angles = [fitted[0].get_angle_between(normal) % (2 * pi) for normal in fitted]
        normals = [facing.rotated(slack * old + progress * new)
                   for old, new in zip(old_angles, new_angles)]

    offsets = []
    for normal in normals:
        # Project the colliders onto this side's normal. The outermost point
        # sets how far the side can move in, including while an item rotates.
        support = max((point - seal.origin).dot(normal) + radius for point, radius in footprint)
        tight = support + SEAL_PADDING
        initial = max((point - seal.origin).dot(normal) for point in seal.marks) if slack else tight
        offsets.append(tight + slack * max(0, initial - tight))
    seal.outline = []
    for index, normal in enumerate(normals):
        previous = normals[index - 1]
        # The intersection of neighboring sides is one of the placed corners.
        corner = (previous.perpendicular() * offsets[index] - normal.perpendicular() * offsets[index - 1])
        seal.outline.append(seal.origin + corner / previous.cross(normal))


def collapse_empty_seal(actions: "ItemActions", seal: SealField) -> None:
    if seal.collapse is None:
        seal.collapse = actions.now, seal.outline[:]
        seal.expires = min(seal.expires, actions.now + SEAL_COLLAPSE_TIME)
    started, outline = seal.collapse
    center = sum(outline, STOP) / len(outline)
    duration = (seal.expires - started) * 0.84
    progress = min(1, (actions.now - started) / duration)
    seal.outline = [center] if progress >= 1 else [point.interpolate_to(center, progress) for point in outline]


def update_seals(actions: "ItemActions", dt: float) -> None:
    maze = actions.maze
    actions.seals = [seal for seal in actions.seals if actions.now < seal.expires]
    for seal in actions.seals:
        seal.captives = {target: anchor for target, anchor in seal.captives.items()
                         if (target.active if isinstance(target, Actor)
                             else target.shape in maze.wielding.by_shape)}
        if not seal.captives:
            collapse_empty_seal(actions, seal)
            continue
        squeeze_captives(actions, seal, dt)
        constrain_captives(actions, seal, dt)
        fit_seal(actions, seal)
        if actions.now < seal.next_damage:
            continue
        seal.next_damage = actions.now + SEAL_INTERVAL
        for victim in maze.actors:
            if not victim.active:
                continue
            vertex_reach_sq = (victim.radius + 8) ** 2
            touching_vertices = sum(
                victim.position.get_dist_sqrd(vertex) <= vertex_reach_sq
                for vertex in seal.outline
            )
            if touching_vertices:
                source = seal.owner if seal.owner is not victim else None
                hit = replace(SEAL_VERTEX_HIT, damage=SEAL_VERTEX_HIT.damage * touching_vertices)
                maze.apply_hit(source, victim.shape, hit, seal.power,
                               delivery="magic", weapon="Sealing Equipment")


def credit_kill(actions: "ItemActions", item: Gear) -> None:
    if item.spec.effect == "whip" and item.shape in actions.maze.wielding.by_shape:
        whip = actions.whips.setdefault(item, WhipMotion())
        whip.count = min(9, whip.count + 1)


def update_whips(actions: "ItemActions", dt: float) -> None:
    maze = actions.maze
    actions.whips = {item: whip for item, whip in actions.whips.items()
                     if item.shape in maze.wielding.by_shape or item in actions.stowed or item in maze.swallowed_gear}
    for item in tuple(maze.wielding.items):
        if item.spec.effect != "whip":
            continue
        whip = actions.whips.setdefault(item, WhipMotion())
        actor = item.holder
        held = actions.held_by(actor) if actor is not None else None
        extending = held is not None and held.whip_extending
        if extending and whip.count > 1:
            whip.extension_seconds += dt
            if whip.extension_seconds >= 5:
                whip.count -= 1
                whip.extension_seconds -= 5
        whip.extension += (int(extending) - whip.extension) * min(1, dt * 7)
        root = item.point_on_axis(1)
        active = maze.wielding.active_swing(item)
        heading = item.axis
        if active and held is not None and held.rotation is None and actor is maze.player:
            # The wrist starts the flick right away; the free links follow with inertia.
            heading = (heading + actor.aim.rotated(held.aim_offset) * 2).normalized()
        del whip.tails[whip.count:]
        for index in range(whip.count):
            fan = 0.075 * ((index + 1) // 2) * (-1 if index % 2 else 1)
            axis = heading.rotated(fan)
            spacing = (65 + 47 * whip.extension - 1.8 * (index % 3)) / WHIP_LINKS
            reset = index == len(whip.tails)
            if reset:
                points = [root + axis * (link * spacing) for link in range(WHIP_LINKS + 1)]
                whip.tails.append(Tail(points, [axis.angle] * WHIP_LINKS, [0.0] * WHIP_LINKS))
            tail = whip.tails[index]
            if tail.points[0].get_dist_sqrd(root) > 110 ** 2:
                # Reset after a teleport or release from All-launcher.
                reset = True
                tail.points = [root] * (WHIP_LINKS + 1)
                tail.angles = [axis.angle] * WHIP_LINKS
                tail.turns = [0.0] * WHIP_LINKS
            hit = step_tail(actions, tail, root, axis, spacing, dt, item if active and not reset else None)
            if hit is not None:
                contact, velocity = hit
                whip_impact(actions, actor, item, whip, contact, velocity)


def step_tail(actions: "ItemActions", tail: Tail, root: Vec2d, axis: Vec2d, spacing: float,
              dt: float, weapon: Gear | None) -> tuple[SweepHit, Vec2d] | None:
    maze = actions.maze
    before = tail.points
    previous_angles = tail.angles[:]
    previous_turns = tail.turns[:]
    points = [root]
    contact = None
    swing_step = STOP

    def difference(target: float, current: float) -> float:
        return (target - current + pi) % (2 * pi) - pi

    # The first link follows the wrist. Each lighter link inherits the
    # preceding link's turn on the following tick and can swing past it.
    tail.angles[0] = axis.angle
    tail.turns[0] = max(-30, min(30, difference(axis.angle, previous_angles[0]) / dt))
    for index in range(WHIP_LINKS):
        if index:
            bend = difference(previous_angles[index - 1], previous_angles[index])
            torque = (590 - 44 * index) * bend + 29 * (previous_turns[index - 1] - previous_turns[index])
            turn = max(-34, min(34, previous_turns[index] + torque * dt))
            angle = previous_angles[index] + turn * dt
            parent = tail.angles[index - 1]
            flex = 0.26 + 0.034 * index
            tail.angles[index] = parent + max(-flex, min(flex, difference(angle, parent)))
            tail.turns[index] = difference(tail.angles[index], previous_angles[index]) / dt

        aim = Vec2d(spacing, 0).rotated(tail.angles[index])
        # Root movement and extension can translate a tail without swinging it.
        # Measure the change in link angles when deciding whether to strike.
        swing_step += aim - Vec2d(spacing, 0).rotated(previous_angles[index])
        proposal = points[-1] + aim
        previous = before[index + 1]
        velocity = swing_step / dt
        wall = maze.hit_solid(points[-1], proposal, 1.1)
        if wall is None and weapon is not None and index >= WHIP_LINKS - 2:
            wall = maze.hit_solid(previous, proposal, 1.1)
        endpoint = wall.point + wall.normal * 1.4 if wall is not None else proposal
        if wall is not None:
            tail.turns[index] *= -0.18
        points.append(endpoint)
        if weapon is None or index < WHIP_LINKS - 2 or velocity.length_squared < 84 ** 2:
            continue
        hits = maze.physics.sweep(previous, endpoint, 2.9, MAGIC_MASK)
        for hit in sorted(hits, key=lambda entry: entry.alpha):
            if hit.shape is weapon.holder.shape or hit.shape.body is weapon.body or hit.shape.sensor:
                continue
            if hit.shape.filter.categories & SOLID and velocity.dot(-hit.normal) < 128:
                continue
            if contact is None or velocity.length_squared > contact[1].length_squared:
                contact = hit, velocity
            break
        if wall is not None and velocity.dot(-wall.normal) >= 128 and (
                contact is None or velocity.length_squared > contact[1].length_squared):
            contact = wall, velocity
    tail.points = points
    return contact


def whip_impact(actions: "ItemActions", actor: Actor, item: Gear, whip: WhipMotion,
                hit: SweepHit, velocity: Vec2d) -> None:
    maze = actions.maze
    shape = hit.shape
    gear = maze.wielding.by_shape.get(shape)
    target = gear.shape if gear is not None else shape
    if actions.now < item.impact_times.get(target, 0):
        return
    item.impact_times[target] = actions.now + 0.27
    speed = velocity.length
    force = actor.strength_multiplier * min(1.6, 0.62 + speed / 550)
    force *= 1 + 0.055 * (whip.count - 1)
    direction = velocity.normalized()
    if shape.filter.categories & SOLID:
        force *= 0.78
    if gear is not None:
        gear.body.apply_impulse_at_world_point(direction * min(125, speed * 0.23), hit.point)
    maze.apply_hit(actor, target, item.spec.attack, force, position=hit.point,
                   shove=direction * min(955, 340 + speed), weapon=item.spec.name, weapon_item=item)
    if actor is maze.player:
        maze.make_noise(hit.point, 6, actor.position)


def load_launcher(actions: "ItemActions", actor: Actor, item: Gear, cursor: Vec2d | None) -> bool:
    maze = actions.maze
    if item in actions.loads:
        return False
    if cursor is None and actor is not maze.player:
        cursor = maze.player.position
    aim = cursor - actor.position if cursor is not None else actor.aim * LAUNCHER_CAPTURE_REACH
    direction = aim.normalized() if aim.length else actor.aim
    end = actor.position + direction * min(LAUNCHER_CAPTURE_REACH, aim.length + 5)
    hits = maze.physics.sweep(actor.position, end, 5, MAGIC_MASK)
    candidates = []
    for hit in sorted(hits, key=lambda contact: contact.alpha):
        shape = hit.shape
        if shape.filter.categories & SOLID:
            break
        target = maze.actors_by_shape.get(shape) or maze.wielding.by_shape.get(shape)
        if target is None or target is actor or target is item or shape.sensor:
            continue
        if isinstance(target, Actor) and not target.active:
            continue
        candidates.append((target, shape))
    if not candidates:
        return False
    if cursor is not None and aim.length <= LAUNCHER_CAPTURE_REACH:
        aimed = [target for target, shape in candidates if shape.point_query(cursor).distance <= 0]
        if aimed:
            return capture_launcher_payload(actions, actor, item, aimed[0])
    return capture_launcher_payload(actions, actor, item, candidates[0][0])


def capture_launcher_payload(actions: "ItemActions", actor: Actor, item: Gear, target: Actor | Gear) -> bool:
    maze = actions.maze
    if item in actions.loads or any(target in seal.captives for seal in actions.seals):
        return False
    if actor is maze.player:
        if maze.progress.xp < 1:
            return False
        maze.progress.xp -= 1
    elif actor.experience < 1:
        return False
    else:
        actor.experience -= 1
    now = actions.now
    origin = target.position
    shape_filter = None
    if isinstance(target, Actor):
        guard = maze.guard_for(target)
        seconds = 7 if guard is not None and guard.boss else LAUNCHER_HOLD.get(target.kind, 12)
        maze.wielding.release_victim(target)
        maze.wielding.release(target)
        actions.end_sword(target)
        target.captured = True
        target.sprinting = False
        target.desired_velocity = target.external_velocity = target.body.velocity = STOP
        shape_filter = maze.physics.suspend_collisions(target.shape, categories=ACTOR)
        radius = target.radius
    else:
        seconds = LAUNCHER_ITEM_HOLD
        radius = max(target.spec.length, target.spec.width) / 2
        # Keep the same gear object, including its condition and epic upgrades.
        maze.wielding.remove(target)
        actions.stowed[target] = item
        actions.marks.pop(target, None)
        target.body.velocity = STOP
        target.body.angular_velocity = 0
    actions.loads[item] = LauncherLoad(target, now + seconds, now + 0.72, origin, shape_filter)
    actions.mark_use(actor, item)
    maze.weapon_flashes.append(WeaponFlash(origin, radius + 12, EPIC_PURPLE, now))
    return True


def fire_launcher(actions: "ItemActions", actor: Actor, item: Gear) -> bool:
    if item not in actions.loads or not unload_launcher(actions, item, shooter=actor):
        return False
    actions.mark_use(actor, item)
    actor.attack_cooldown = item.spec.attack.recovery
    return True


def outer_launcher(actions: "ItemActions", item: Gear) -> Gear:
    # A launcher can carry a loaded launcher. Stowed weapons are absent from the
    # world queries, so every chain leads to a single outer weapon.
    while item in actions.stowed:
        item = actions.stowed[item]
    return item


def release_location(actions: "ItemActions", load: LauncherLoad, item: Gear, direction: Vec2d, *, firing: bool) -> Vec2d | None:
    maze = actions.maze
    payload = load.captive
    if isinstance(payload, Actor):
        clearance = payload.radius + 1
        free = maze.physics.placement_probe(clearance)
    else:
        length, width = payload.spec.length + 3, payload.spec.width + 3
        clearance = max(length, width) / 2 + 1
        angle = direction.angle if firing else payload.body.angle
        free = maze.physics.placement_probe(clearance, (length, width), angle=angle)

    holder = item.holder
    muzzle = actions.muzzle(holder, item) if holder is not None else item.point_on_axis(1)
    origin = holder.position if holder is not None else item.position
    angles = (0, 0.40, -0.40, 0.8, -0.8) if firing else (0, 0.62, -0.62, 1.57, -1.57, pi)
    for angle in angles:
        step = direction.rotated(angle)
        for distance in (clearance + 3, clearance + 18, clearance + 37):
            point = muzzle + step * distance
            if maze.clear_path(origin, point) and free(point):
                return point
    if firing:
        return None
    if free(load.origin):
        return load.origin
    for radius in (clearance + 4, clearance + 24, clearance + 55):
        for index in range(8):
            point = load.origin + Vec2d(radius, 0).rotated(index * pi / 4)
            if free(point):
                return point
    return None


def unload_launcher(actions: "ItemActions", item: Gear, shooter: Actor | None = None) -> bool:
    load = actions.loads[item]
    payload = load.captive
    maze = actions.maze
    carrier = outer_launcher(actions, item)
    direction = shooter.aim if shooter is not None else carrier.axis
    position = release_location(actions, load, carrier, direction, firing=shooter is not None)
    if position is None:
        return False
    actions.loads.pop(item)
    payload.body.position = position
    if isinstance(payload, Actor):
        radius = payload.radius
        maze.physics.restore_collisions(payload.shape, load.shape_filter)
        payload.captured = False
        if shooter is not None:
            maze.knock_actor(shooter, payload, direction * (1000 * shooter.dexterity_multiplier),
                             delivery="projectile", weapon=LAUNCHER_IMPACT)
    else:
        radius = max(payload.spec.length, payload.spec.width) / 2
        del actions.stowed[payload]
        if shooter is not None:
            payload.body.angle = direction.angle
            speed = min(LAUNCHER_ITEM_SPEED_CAP,
                        LAUNCHER_ITEM_SPEED * shooter.dexterity_multiplier / payload.body.mass ** 0.25)
            payload.body.velocity = direction * speed
        maze.wielding.restore(payload)
        if shooter is not None:
            for shape in payload.shapes:
                maze.wielding.attribute_motion(shape, shooter, 1.35, delivery="projectile",
                                              weapon=LAUNCHER_IMPACT)
    maze.weapon_flashes.append(WeaponFlash(position, radius + 12, EPIC_PURPLE, actions.now))
    return True


def update_launcher_loads(actions: "ItemActions") -> None:
    for item, load in tuple(actions.loads.items()):
        carrier = outer_launcher(actions, item)
        if carrier.shape not in actions.maze.wielding.by_shape or actions.now >= load.expires:
            if unload_launcher(actions, item):
                continue
        elif (load.captive is actions.maze.player and carrier is item and item.holder is not None
              and actions.now >= load.fire_at and fire_launcher(actions, item.holder, item)):
            continue
        load.captive.body.position = carrier.position
        load.captive.body.velocity = STOP


def update_item_effects(actions: "ItemActions", dt: float) -> None:
    update_launcher_loads(actions)
    update_drains(actions)
    update_seals(actions, dt)
    for item, (actor, deadline) in tuple(actions.enemy_seals.items()):
        if item.holder is not actor:
            actions.enemy_seals.pop(item)
            actions.marks.pop(item, None)
        elif actions.now >= deadline:
            actions.enemy_seals.pop(item)
            if not bind_seal(actions, actor, item):
                actions.marks.pop(item, None)
    update_whips(actions, dt)
    by_shape = actions.maze.wielding.by_shape
    actions.book_streaks = {item: count for item, count in actions.book_streaks.items()
                            if item.shape in by_shape or item in actions.stowed or item in actions.maze.swallowed_gear}
    actions.marks = {item: marks for item, marks in actions.marks.items() if item.shape in by_shape}
