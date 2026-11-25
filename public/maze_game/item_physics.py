"""Physical gear, an off-center mouse grab, and collision impact events."""

from dataclasses import dataclass, field
from math import atan2, copysign, cos, radians, sin

import pymunk
from .physics import Vec2d

from .actor import Actor
from .item import AttackProfile, Equipment, Gear
from .physics import ACTOR_TYPE, GEAR_CATEGORY, GEAR_TYPE, SPIKE_CATEGORY, SPIKE_TYPE, Physics, catcher_jaw_line

GRAB_RANGE = 63
MIN_HAND_REACH = 16
HAND_REACH = 52
HAND_CONE = radians(23)
GUARD_SWING_TIME = 0.38
MIN_IMPACT_SPEED = 58
HARD_SURFACE_SPEED = 125
HARD_SURFACE_IMPULSE_SPEED = 10500
SURFACE_DAMAGE_SCALE = 0.60
HAMMER_KNOCKBACK_SCALE = 1.35
IMPACT_INTERVAL = 0.3
CLAMP_TIME = 0.18
BASE_GRIP_FORCE = 2950
LOOSE_ITEM_DECELERATION = 370
LOOSE_ITEM_SPIN_DECELERATION = 19
WIELDER_FORCE_MULTIPLIER = 2.0
WIELDER_STAMINA_DRAIN = 17
CHALLENGER_STAMINA_DRAIN = 27
SHARP_GRIP_INTERVAL = 0.85
SHIELD_RESISTANCE_PERCENT = 40
SHIELD_SWING_SHOVE_SCALE = 1.6
SHIELD_RUSH_ATTACK = AttackProfile(0, 4, 0)
COLLISION_ATTACK = AttackProfile(0, 2, 0)
CATCHER_COLLISION = AttackProfile(0, 5, 0)
LAUNCHER_COLLISION = AttackProfile(0, 8, 0)
LAUNCHER_IMPACT = "All-launcher impact"
GRIP_CUT = AttackProfile(0, 1, 0, "slashing")
CATCHER_CONTROL_SECONDS = 4.0
CATCHER_GRIP_FORCE = 4900

GEAR_COLORS = {
    "melee": (220, 208, 180),
    "ranged": (153, 213, 156),
    "magic": (198, 158, 250),
    "utility": (253, 195, 104),
    "shield": (186, 209, 225),
}


@dataclass
class Spike:
    expires: float
    length: float
    piercing: bool = False
    embedded: pymunk.Shape | None = None
    anchor: Vec2d = Vec2d(0, 0)
    local_direction: Vec2d = Vec2d(1, 0)
    tow: pymunk.Body | None = None
    struck: set[pymunk.Body] = field(default_factory=set)
    hp: int = 2
    collider: pymunk.Segment | None = None

    @property
    def resting(self) -> bool:
        return self.collider is not None


@dataclass
class Projectile:
    owner: Actor
    emitter: pymunk.Shape
    position: Vec2d
    direction: Vec2d
    attack: AttackProfile
    remaining: float
    mode: str
    speed: float
    effect: str | None = None
    radius: float = 2
    shove: float = 515
    charge: float = 0
    target: Actor | None = None
    weapon: str | None = None
    spike: Spike | None = None
    power: float = 1.0


@dataclass(frozen=True)
class Impact:
    source: Actor | None
    target: pymunk.Shape
    attack: AttackProfile
    force: float
    position: Vec2d
    shove: Vec2d | None = None
    snaring_gear: "Gear | None" = None
    delivery: str = "melee"
    weapon: str | None = None
    weapon_item: "Gear | None" = None


@dataclass(frozen=True)
class AttributedMotion:
    source: Actor
    expires: float
    delivery: str
    weapon: str | None
    weapon_item: "Gear | None" = None


@dataclass
class ShieldRush:
    direction: Vec2d
    ends_at: float
    speed: float
    next_hit: dict[pymunk.Shape, float] = field(default_factory=dict)


@dataclass
class HeldGear:
    item: "Gear"
    hand: pymunk.Body
    joint: pymunk.PivotJoint
    sharp: bool = False
    sharp_strain_time: float = 0.0
    off_axis_time: float = 0.0
    strain_time: float = 0.0
    rotation: float | None = None
    aim_offset: float = 0.0
    primary_down: bool = False
    charge: float = 0
    gravity_charging: bool = False
    bow_lance_charging: bool = False
    rush: ShieldRush | None = None
    sword_destination: Vec2d | None = None
    lock_target: Actor | None = None
    whip_extending: bool = False

    @property
    def grip(self) -> Vec2d:
        return self.item.body.local_to_world(self.joint.anchor_b)


@dataclass
class Snare:
    victim: Actor
    joint: pymunk.PivotJoint
    started: float
    strain_time: float = 0
    control_until: float = 0


@dataclass
class SpikeLoad:
    joint: pymunk.PivotJoint
    actor: Actor | None


def hand_target(origin: Vec2d, facing: Vec2d, cursor: Vec2d, reach: float = HAND_REACH) -> Vec2d:
    offset = cursor - origin
    angle = facing.get_angle_between(offset) if offset.length else 0.0
    angle = min(HAND_CONE, max(-HAND_CONE, angle))
    distance = min(reach, max(MIN_HAND_REACH, offset.length))
    return origin + facing.rotated(angle) * distance


def impact_wear(impulse: float, surface_contact: bool) -> int:
    return max(1, int(impulse // (225 if surface_contact else 155)))


class Wielding:
    def __init__(self, physics: Physics, actors: dict[pymunk.Shape, Actor]):
        self.actors = actors
        self.space = physics.space
        self.items: list[Gear] = []
        self.by_shape: dict[pymunk.Shape, Gear] = {}
        self.held: dict[Actor, HeldGear] = {}
        self.impacts: list[Impact] = []
        self.wear_events: list[tuple[Gear, int]] = []
        self.broken: list[tuple[Equipment, Vec2d]] = []
        self.snares: dict[Gear, Snare] = {}
        self.tethers: dict[Actor, Snare] = {}
        self.spike_loads: dict[pymunk.Body, SpikeLoad] = {}
        self.spikes: dict[pymunk.Shape, Spike] = {}
        self.contact_noise: list[tuple[Actor, Vec2d, int]] = []
        self.approach_speeds: dict[tuple[pymunk.Shape, pymunk.Shape], tuple[float, float]] = {}
        self.momentum_sources: dict[pymunk.Shape, AttributedMotion] = {}
        self.hard_contacts: dict[frozenset[pymunk.Shape], float] = {}
        self.jostled: set[Actor] = set()
        self.mirror_touched = False
        self.time = 0.0
        for first, second in (
            (GEAR_TYPE, 0), (GEAR_TYPE, GEAR_TYPE), (GEAR_TYPE, ACTOR_TYPE),
            (ACTOR_TYPE, 0), (ACTOR_TYPE, ACTOR_TYPE),
            (GEAR_TYPE, SPIKE_TYPE),
        ):
            self.space.on_collision(first, second, pre_solve=self.before_contact, post_solve=self.after_contact)

    def add(self, spec: Equipment, position: Vec2d, angle: float = 0) -> Gear:
        item = Gear.spawn(self.space, spec, position, angle)
        self.items.append(item)
        self.by_shape.update((shape, item) for shape in item.shapes)
        return item

    def restore(self, item: Gear) -> None:
        self.space.add(item.body, *item.shapes)
        self.items.append(item)
        self.by_shape.update((shape, item) for shape in item.shapes)

    def apply_floor_drag(self, dt: float) -> None:
        """Remove loose gear's sliding and spin speed along the floor."""
        speed_loss = LOOSE_ITEM_DECELERATION * dt
        spin_loss = LOOSE_ITEM_SPIN_DECELERATION * dt
        for item in self.items:
            if (item.holder is not None or item.contender is not None
                    or item in self.snares or item.body in self.spike_loads):
                continue
            body = item.body
            # In a top-down world the floor isn't a collision shape. Constant
            # deceleration gives the object a real stop instead of a slow drift.
            speed = body.velocity.length
            if speed:
                body.velocity *= max(0, 1 - speed_loss / speed)
            spin = body.angular_velocity
            if spin:
                body.angular_velocity = copysign(max(0, abs(spin) - spin_loss), spin)

    def can_grab(self, actor: Actor, item: Gear, *, contest: bool = False,
                 replace_held: bool = False) -> bool:
        held = self.held.get(actor)
        return (
            actor.active and actor.can_wield and actor.stamina > 0
            and (held is None or replace_held and held.item is not item)
            and item.shape in self.by_shape
            and (item.holder is None or (contest and item.holder is not actor and item.contender is None))
        )

    def grab(self, actor: Actor, item: Gear, point: Vec2d, *, contest: bool = False,
             replace_held: bool = False) -> bool:
        if not self.can_grab(actor, item, contest=contest, replace_held=replace_held):
            return False
        if replace_held:
            self.release(actor)
        hand = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        hand.position = point
        joint = pymunk.PivotJoint(hand, item.body, (0, 0), item.body.world_to_local(point))
        joint.max_force = BASE_GRIP_FORCE * actor.grip_multiplier
        joint.max_bias = 210
        self.space.add(hand, joint)
        sharp = item.sharp_at(point)
        self.held[actor] = HeldGear(item, hand, joint, sharp=sharp)
        if sharp:
            self.impacts.append(Impact(item.holder, actor.shape, GRIP_CUT, 1.0, point))
        if item.holder is None:
            item.holder = actor
            actor.equipment = item.spec
        else:
            item.contender = actor
            if item.spec.mode == "melee":
                self.held[item.holder].joint.max_force *= WIELDER_FORCE_MULTIPLIER
        return True

    def release(self, actor: Actor) -> None:
        held = self.held.pop(actor, None)
        if held is None:
            return
        self.space.remove(held.joint, held.hand)
        item = held.item
        actor.equipment = None
        if item.contender is actor:
            item.contender = None
            wielder = self.held[item.holder]
            wielder.strain_time = wielder.sharp_strain_time = 0
            wielder.joint.max_force = BASE_GRIP_FORCE * item.holder.grip_multiplier
            return
        item.holder = item.contender
        item.contender = None
        item.impact_times.clear()
        actor.swing_time = 0
        if item.spec.effect == "catcher":
            self.unhook(item)
        if item.holder is not None:
            item.holder.equipment = item.spec
            wielder = self.held[item.holder]
            wielder.strain_time = wielder.sharp_strain_time = 0
        else:
            self.attribute_motion(item.shape, actor, 0.8, weapon=f"Flying {item.spec.name}")

    def resolve_grips(self, dt: float) -> None:
        slipping = []
        for actor, held in self.held.items():
            item = held.item
            if item.contender is None:
                continue
            error = held.grip.get_distance(held.hand.position)
            tension = min(1.0, held.joint.impulse / (held.joint.max_force * dt))
            if item.spec.mode == "melee":
                wielding = item.holder is actor
                drain = WIELDER_STAMINA_DRAIN if wielding else CHALLENGER_STAMINA_DRAIN
                actor.spend_stamina(drain * (0.35 + 0.65 * tension) * dt)
                if held.sharp and tension > 0.48:
                    held.sharp_strain_time += dt
                    if held.sharp_strain_time >= SHARP_GRIP_INTERVAL:
                        other = item.contender if wielding else item.holder
                        self.impacts.append(Impact(other, actor.shape, GRIP_CUT, 1.0, held.grip))
                        held.sharp_strain_time -= SHARP_GRIP_INTERVAL
            if error > 13 and tension > 0.82:
                held.strain_time += dt
            else:
                held.strain_time = max(0, held.strain_time - 2 * dt)
            if held.strain_time >= 0.25 * actor.grip_multiplier:
                slipping.append((held.strain_time / actor.grip_multiplier, actor))
        exhausted = [actor for actor in self.held if actor.stamina == 0]
        # Release both exhausted hands before transferring control to a winner.
        for actor in sorted(exhausted, key=lambda actor: self.held[actor].item.holder is actor):
            self.release(actor)
        for _, actor in sorted(slipping, key=lambda entry: -entry[0]):
            if (held := self.held.get(actor)) is not None and held.item.contender is not None:
                self.release(actor)
        for item, snare in tuple(self.snares.items()):
            joint = snare.joint
            if snare.control_until and self.time >= snare.control_until:
                snare.control_until = 0
                snare.victim.restraint = 0.21
                joint.max_force, joint.max_bias = CATCHER_GRIP_FORCE, 275
            tip = item.body.local_to_world(joint.anchor_a)
            contact = snare.victim.body.local_to_world(joint.anchor_b)
            snare.strain_time = snare.strain_time + dt if tip.get_distance(contact) > 19 else 0
            if snare.strain_time > 0.33:
                self.unhook(item)
            elif item.holder is not None:
                self.attribute_motion(snare.victim.shape, item.holder, 0.9, weapon=item.spec.name)

        for item in self.items:
            if not item.jaws:
                continue
            amount = dt / CLAMP_TIME
            closure = min(1.0, item.jaw_closure + amount) if item in self.snares else max(0.0, item.jaw_closure - amount)
            if item.jaw_closure == closure:
                continue
            item.jaw_closure = closure
            for side, jaw in zip((-1, 1), item.jaws):
                jaw.unsafe_set_endpoints(*catcher_jaw_line(item.spec, side, closure))
                self.space.reindex_shape(jaw)

    def hook(self, item: Gear, victim: Actor, position: Vec2d) -> bool:
        if (item.shape not in self.by_shape or item in self.snares or victim.restraint is not None
                or not victim.alive or item.spec.effect not in ("spear", "catcher")):
            return False
        catcher = item.spec.effect == "catcher"
        if catcher and victim.kind == "mimic":
            return False
        joint = pymunk.PivotJoint(item.body, victim.body, position)
        joint.max_force = CATCHER_GRIP_FORCE if catcher else 2850
        joint.max_bias = 275 if catcher else 235
        joint.collide_bodies = False
        self.space.add(joint)
        victim.restraint = 0.21 if catcher else 0.54
        self.snares[item] = Snare(victim, joint, self.time)
        return True

    def unhook(self, item: Gear) -> None:
        if (snare := self.snares.pop(item, None)) is not None:
            self.space.remove(snare.joint)
            snare.victim.restraint = None

    def take_control(self, item: Gear) -> bool:
        snare = self.snares.get(item)
        if snare is None or snare.control_until > self.time:
            return False
        snare.control_until = self.time + CATCHER_CONTROL_SECONDS
        snare.victim.restraint = 0.04
        snare.joint.max_force, snare.joint.max_bias = 14000, 520
        return True

    def tether_actor(self, source: Actor, victim: Actor, *, force: float, mobility: float) -> Snare | None:
        if source in self.tethers or victim.restraint is not None or not source.alive or not victim.alive:
            return None
        joint = pymunk.PivotJoint(source.body, victim.body, victim.position)
        joint.max_force = force
        joint.max_bias = 345
        self.space.add(joint)
        victim.restraint = mobility
        snare = Snare(victim, joint, self.time)
        self.tethers[source] = snare
        return snare

    def release_tether(self, source: Actor) -> None:
        if (snare := self.tethers.pop(source, None)) is not None:
            self.space.remove(snare.joint)
            snare.victim.restraint = None

    def release_victim(self, victim: Actor) -> None:
        for item, snare in tuple(self.snares.items()):
            if snare.victim is victim:
                self.unhook(item)
        for source, snare in tuple(self.tethers.items()):
            if victim in (source, snare.victim):
                self.release_tether(source)
        self.release_spike_load(victim.body)

    def load_spike(self, shot: Projectile, shape: pymunk.Shape, point: Vec2d) -> None:
        spike = shot.spike
        actor = self.actors.get(shape)
        item = self.by_shape.get(shape)
        body = shape.body
        if body in self.spike_loads or shape.space is not self.space:
            return
        if actor is not None:
            if not actor.active or actor.restraint is not None:
                return
            if actor.kind == "mimic" and self.time < actor.split_safe_until:
                return
        if actor is None and item is None:
            return
        if spike.tow is None:
            spike.tow = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            spike.tow.position = shot.position
            self.space.add(spike.tow)
        extent = actor.radius if actor is not None else max(item.spec.length, item.spec.width) / 2
        # A new target gets the front of the shaft. Move the older loads back so they stay in hit order.
        spacing = shot.direction * (2 * extent + 4)
        for load in self.spike_loads.values():
            if load.joint.a is spike.tow:
                load.joint.anchor_a -= spacing
        trail = extent + max(0, (body.position - point).dot(shot.direction)) + 7
        joint = pymunk.PivotJoint(spike.tow, body, -shot.direction * trail, body.world_to_local(point))
        joint.max_force = 5100 * min(2.1, body.mass ** 0.48)
        joint.max_bias = 380
        self.space.add(joint)
        self.spike_loads[body] = SpikeLoad(joint, actor)
        if actor is not None:
            actor.restraint = 0.22

    def lodge_spike(self, shot: Projectile) -> None:
        spike = shot.spike
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        collider = pymunk.Segment(body, (-spike.length, 0), (0, 0), shot.radius)
        collider.filter = pymunk.ShapeFilter(categories=SPIKE_CATEGORY, mask=GEAR_CATEGORY)
        collider.collision_type = SPIKE_TYPE
        spike.collider = collider
        self.spikes[collider] = spike
        body.position, body.angle = shot.position, shot.direction.angle
        self.space.add(body, collider)
        tow = spike.tow
        if tow is None:
            return
        tow.position = shot.position
        tow.velocity = Vec2d(0, 0)
        for body, load in self.spike_loads.items():
            if load.joint.a is not tow:
                continue
            # The shaft absorbs forward momentum as it lodges. Sideways motion remains physical.
            body.velocity -= shot.direction * max(0, body.velocity.dot(shot.direction) - 300)
            if load.actor is not None:
                velocity = load.actor.external_velocity
                load.actor.external_velocity = velocity - shot.direction * max(0, velocity.dot(shot.direction) - 300)

    def release_spike_load(self, body: pymunk.Body) -> None:
        if (load := self.spike_loads.pop(body, None)) is not None:
            self.space.remove(load.joint)
            if load.actor is not None:
                load.actor.restraint = None

    def release_spike(self, spike: Spike) -> None:
        self.release_spike_tow(spike)
        if spike.collider is not None and spike.collider in self.spikes:
            collider = spike.collider
            self.spikes.pop(collider)
            self.space.remove(collider, collider.body)

    def release_spike_tow(self, spike: Spike) -> None:
        if spike.tow is not None:
            for body, load in tuple(self.spike_loads.items()):
                if load.joint.a is spike.tow:
                    self.release_spike_load(body)
            self.space.remove(spike.tow)
            spike.tow = None

    def wear(self, item: Gear, amount: int = 1) -> None:
        if item.shape not in self.by_shape or item.durability is None:
            return
        item.durability = max(0, item.durability - amount)
        if item.durability:
            return
        if not item.temporary:
            self.broken.append((item.spec, item.position))
        self.remove(item)

    def remove(self, item: Gear) -> None:
        self.unhook(item)
        self.release_spike_load(item.body)
        if item.contender is not None:
            self.release(item.contender)
        if item.holder is not None:
            self.release(item.holder)
        self.space.remove(*item.shapes, item.body)
        for shape in item.shapes:
            del self.by_shape[shape]
        self.momentum_sources.pop(item.shape, None)
        self.items.remove(item)

    def attribute_motion(self, shape: pymunk.Shape, source: Actor, seconds: float,
                         *, delivery: str = "melee", weapon: str | None = None,
                         weapon_item: Gear | None = None) -> None:
        self.momentum_sources[shape] = AttributedMotion(source, self.time + seconds, delivery, weapon, weapon_item)

    def start_rotation(self, actor: Actor) -> None:
        if (held := self.held.get(actor)) is not None:
            held.rotation = held.item.body.angle

    def rotate(self, actor: Actor, radians_delta: float) -> None:
        if (held := self.held.get(actor)) is not None and held.rotation is not None:
            held.rotation += radians_delta

    def stop_rotation(self, actor: Actor) -> None:
        if (held := self.held.get(actor)) is not None and held.rotation is not None:
            held.aim_offset = held.rotation - actor.aim.angle
            held.rotation = None

    def active_swing(self, item: Gear) -> bool:
        actor = item.holder
        if actor is None or item.spec.mode not in ("melee", "shield") or item.spec.attack is None:
            return False
        held = self.held[actor]
        return held.rush is None and (held.primary_down or actor.swing_time > 0)

    def update_hand(self, actor: Actor, cursor: Vec2d, dt: float, *, player_control: bool = False) -> None:
        held = self.held.get(actor)
        if held is None:
            return
        grip = held.grip
        if actor.stamina == 0 or grip.get_distance(actor.position) > GRAB_RANGE * 1.65 + actor.radius - 10:
            self.release(actor)
            return
        # The solver moves the selected local anchor. The center can swing around it.
        facing = actor.aim
        offset = grip - actor.position
        outside = offset.length < 8 or abs(facing.get_angle_between(offset)) > HAND_CONE
        held.off_axis_time = held.off_axis_time + dt if actor.sprinting and outside else 0.0
        if held.off_axis_time >= 0.20 * actor.grip_multiplier:
            self.release(actor)
            return
        active = self.active_swing(held.item)
        reach = HAND_REACH + actor.radius - 10
        rush = held.rush
        if rush is not None:
            facing = rush.direction
            cursor = actor.position + facing * (0.66 * reach)
        elif not player_control and (cursor - actor.position).length:
            facing = (cursor - actor.position).normalized()
        target = hand_target(actor.position, facing, cursor, reach)
        delta = target - held.hand.position
        held.hand.velocity = delta.normalized() * min(650 if active or rush is not None else 480, delta.length / dt)
        if held.item.holder is not actor:
            return
        if rush is not None:
            held.item.steer(rush.direction.angle, held.joint.anchor_b)
        elif held.rotation is not None:
            held.item.steer(held.rotation, held.joint.anchor_b, manual=True)
        elif held.item.spec.effect in ("spear", "catcher", "shield", "whip") or (not player_control and active) or (
            player_control and held.item.spec.mode in ("ranged", "magic")
        ):
            angle = actor.aim.angle
            if not player_control and active and held.item.spec.effect not in ("spear", "catcher"):
                angle = facing.angle
            held.item.steer(angle + held.aim_offset, held.joint.anchor_b)

    def before_contact(self, arbiter: pymunk.Arbiter, _space, _data) -> None:
        first, second = arbiter.shapes
        first_item, second_item = self.by_shape.get(first), self.by_shape.get(second)
        own_contact = (
            (first_item is not None and self.actors.get(second) is not None
             and self.actors[second] in (first_item.holder, first_item.contender))
            or (second_item is not None and self.actors.get(first) is not None
                and self.actors[first] in (second_item.holder, second_item.contender))
        )
        arbiter.process_collision = not own_contact
        points = arbiter.contact_point_set.points
        if own_contact or not points:
            return
        contact = points[0]
        normal = arbiter.normal
        self.approach_speeds[first, second] = (
            first.body.velocity_at_world_point(contact.point_a).dot(normal),
            second.body.velocity_at_world_point(contact.point_b).dot(-normal),
        )

    def after_contact(self, arbiter: pymunk.Arbiter, _space, _data) -> None:
        # Collect impacts here and apply their damage outside Space.step().
        first, second = arbiter.shapes
        first_speed, second_speed = self.approach_speeds.pop((first, second), (0.0, 0.0))
        speed = max(0.0, first_speed + second_speed)
        impulse = arbiter.total_impulse.length
        points = arbiter.contact_point_set.points
        if not points:
            return
        contact = points[0]
        normal = arbiter.normal
        parts = (
            (first, second, normal, contact.point_a - normal * 3, first_speed),
            (second, first, -normal, contact.point_b + normal * 3, second_speed),
        )
        mirror_contact = False
        safe_mirror_contact = False
        for shape in (first, second):
            if (item := self.by_shape.get(shape)) is not None and item.spec.effect == "mirror":
                mirror_contact = True
                launched = self.momentum_sources.get(shape)
                safe_mirror_contact = launched is None or launched.weapon != LAUNCHER_IMPACT
        if mirror_contact and any(self.actors.get(shape) is not None and self.actors[shape].kind == "player"
                                  for shape in (first, second)):
            self.mirror_touched = True
        propelled = None
        captive = None
        rush_contact = False
        for shape, other, toward, position, incoming in parts:
            item = self.by_shape.get(shape)
            actor = self.actors.get(shape)
            driver = item.holder if item is not None else actor
            rushing = not safe_mirror_contact and self.record_rush_contact(driver, other, position, toward)
            rush_contact = rush_contact or rushing
            if item is not None and not safe_mirror_contact and not rushing:
                victim = self.actors.get(other)
                strike_speed = speed if victim is None or victim.sprinting else min(speed, incoming)
                self.record_gear_contact(item, other, strike_speed, impulse, position, toward)
                launch = self.momentum_sources.get(shape)
                launched_payload = launch is not None and launch.weapon == LAUNCHER_IMPACT
                if item.holder is None and (item.spec.effect != "catcher" or launched_payload) and strike_speed >= 145:
                    propelled = shape if propelled is None else propelled
            if actor is not None:
                if actor.external_velocity.dot(toward) > 85 or actor.restraint is not None:
                    propelled = shape
                if actor.restraint is not None and incoming >= 76:
                    for catcher, snare in self.snares.items():
                        if snare.victim is actor and catcher.spec.effect == "catcher" and catcher.holder is not None:
                            captive = (shape, catcher)
                            break
                actor.absorb_collision(toward)
                if other.collision_type == 0 and actor.sprinting and impulse >= 125 * actor.grip_multiplier:
                    self.jostled.add(actor)
        surface_contact = first not in self.actors and second not in self.actors
        minimum_speed, minimum_impact = (165, 22000) if surface_contact else (145, 11500)
        if captive is not None:
            propelled, catcher = captive
            minimum_speed, minimum_impact = 90, 5400
        if safe_mirror_contact or rush_contact or propelled is None or speed < minimum_speed or impulse * speed < minimum_impact:
            return
        pair = frozenset((first, second))
        if self.time < self.hard_contacts.get(pair, 0):
            return
        self.hard_contacts[pair] = self.time + IMPACT_INTERVAL
        attributed = self.momentum_sources.get(propelled)
        source, delivery, weapon = None, "melee", None
        weapon_item = None
        if attributed is not None and attributed.expires > self.time:
            source, delivery, weapon = attributed.source, attributed.delivery, attributed.weapon
            weapon_item = attributed.weapon_item
        force = min(2.8, impulse * speed / 24500)
        attack = COLLISION_ATTACK
        if captive is not None:
            source = catcher.holder
            attack = CATCHER_COLLISION
            weapon = catcher.spec.name
            delivery = "melee"
            force = max(0.72, min(1.7, impulse * speed / 18000))
        elif weapon == LAUNCHER_IMPACT:
            attack = LAUNCHER_COLLISION
            force = max(0.66, min(1.8, impulse * speed / 25000))
        elif weapon is None and (thrown := self.by_shape.get(propelled)) is not None:
            weapon = f"Flying {thrown.spec.name}"
        if surface_contact:
            force *= SURFACE_DAMAGE_SCALE
        for shape, _, toward, position, _ in parts:
            throws_actor = shape in self.actors and shape is not propelled and (
                propelled in self.by_shape or captive is not None or attack is LAUNCHER_COLLISION
            )
            shove = -toward * min(540, speed * 1.25) if throws_actor else None
            self.impacts.append(Impact(source, shape, attack, force, position, shove,
                                       delivery=delivery, weapon=weapon, weapon_item=weapon_item))

    def record_rush_contact(self, actor: Actor | None, other: pymunk.Shape, origin: Vec2d, toward: Vec2d) -> bool:
        held = self.held.get(actor)
        if held is None or (rush := held.rush) is None or toward.dot(rush.direction) < 0.47:
            return False
        item = self.by_shape.get(other)
        target = item.shape if item is not None else other
        if self.time >= rush.next_hit.get(target, 0):
            rush.next_hit[target] = self.time + 0.43
            force = max(0.5, rush.speed / 310) * actor.strength_multiplier
            shove = rush.direction * (310 + rush.speed * 0.53)
            self.impacts.append(Impact(actor, target, SHIELD_RUSH_ATTACK, force, origin, shove, weapon="Shield rush"))
            self.wear_events.append((held.item, 1))
            self.contact_noise.append((actor, origin, 6))
        if target in self.actors or item is not None:
            self.attribute_motion(target, actor, 0.9, weapon="Shield rush")
        return True

    def record_gear_contact(self, item: Gear, other: pymunk.Shape, speed: float, impulse: float,
                            origin: Vec2d, toward: Vec2d) -> None:
        if item.spec.effect == "whip":
            return  # Whip strikes are measured at the flexible tips.
        actor = item.holder
        active = self.active_swing(item)
        catch = active and item.spec.effect == "catcher"
        minimum = 30 if catch else MIN_IMPACT_SPEED
        if actor is None or speed < minimum or self.time < item.impact_times.get(other, 0):
            return
        surface_contact = other not in self.actors
        if (surface_contact and other not in self.spikes
                and (speed < HARD_SURFACE_SPEED or impulse * speed < HARD_SURFACE_IMPULSE_SPEED)):
            return
        item.impact_times[other] = self.time + IMPACT_INTERVAL
        self.contact_noise.append((actor, origin, 4 + min(3, int(impulse / 55))))
        attack = item.spec.attack
        if active or impulse > 95:
            self.wear_events.append((item, impact_wear(impulse, surface_contact)))
        if active and (catch or impulse >= 6):
            force = min(1.28, max(0.46, (speed / 205 + min(impulse, 156) / 195) / 2))
            if surface_contact:
                force *= SURFACE_DAMAGE_SCALE
            local_x = item.body.world_to_local(origin).x
            tip_start = 0.19 if item.spec.effect == "catcher" else 0.27
            tip = item.spec.effect in ("spear", "catcher") and local_x >= item.spec.length * tip_start
            snare = item if tip and other in self.actors else None
            shove = None
            if attack.damage and not catch:
                knockback = min(760, 290 + speed * 1.13)
                if item.spec.effect == "hammer":
                    knockback *= HAMMER_KNOCKBACK_SCALE
                elif item.spec.effect == "shield":
                    knockback *= SHIELD_SWING_SHOVE_SCALE
                shove = toward * knockback
            self.impacts.append(Impact(actor, other, attack, force * actor.strength_multiplier,
                                       origin, shove, snare, weapon=item.spec.name, weapon_item=item))

    def clear_old_contacts(self) -> None:
        self.hard_contacts = {pair: until for pair, until in self.hard_contacts.items() if until > self.time}
        self.momentum_sources = {shape: owner for shape, owner in self.momentum_sources.items()
                                 if owner.expires > self.time}
