"""Weapon inputs, charged shots, and shared player and enemy abilities."""

from dataclasses import dataclass, field, replace
from math import cos, inf, radians
from typing import TYPE_CHECKING


from .projectiles import resolve_shot, spray_flame
from .physics import ALL_CATEGORIES, Shape, SweepHit, Vec2d
from .effects import Beam, WeaponFlash
from .physics import ACTOR, SOLID
from .actor import Actor
from .item import AttackProfile, healing_pack_hp
from . import item_effects
from .item_effects import DrainField, LauncherLoad, SealField, WhipMotion
from .layout import COLS, ROWS, TILE
from .physics import GEAR_CATEGORY, SPIKE_CATEGORY
from .item_physics import (
    GUARD_SWING_TIME,
    HAMMER_KNOCKBACK_SCALE,
    HeldGear,
    Projectile,
    ShieldRush,
    Spike,
    impact_wear,
)
from .item import Gear

if TYPE_CHECKING:
    from .world import Maze

USE_WEAR = 1
GRAVITY_CHARGE_TIME = 1.75
SWORD_DASH_SPEED = 1640
LOCKED_SHOT_SPEED = 1630
BOW_LANCE_DRAW_TIME = 1.65
SHIELD_RUSH_SPEED = 370
SHIELD_RUSH_ACCELERATION = 740
SHIELD_RUSH_TURN = radians(65)
SHIELD_RUSH_STAMINA = 30
PULSE_SHOVE = 1060
LASER_SHOVE = 760
WAND_SHOVE = 705
PULSE_BURST = (116, 4, 1140, (135, 233, 243))
WAND_FAN = AttackProfile(255, 4, 1.1, "arcane")
MOLOTOV = AttackProfile(270, 3, 0.24, "heat")
STOP = Vec2d(0, 0)
SPECIAL_COOLDOWN = {
    "hammer": 2.5, "sword": 2.45, "spear": 0.34, "catcher": 4.0,
    "bow": 4.5, "sling": 1.0, "beam": 2.6, "gravity": 1.5,
    "flame": 2.4, "wand": 1.9, "pulse": 2.2, "shield": 1.8,
}
EPIC_EFFECTS = ("grimoire", "seal", "whip", "launcher")


@dataclass
class SwordDash:
    item: Gear
    destination: Vec2d
    struck: set[Shape] = field(default_factory=set)


class ItemActions:
    def __init__(self, maze: "Maze"):
        self.maze = maze
        self.dashes: dict[Actor, SwordDash] = {}
        self.book_streaks: dict[Gear, int] = {}
        self.drains: list[DrainField] = []
        self.marks: dict[Gear, list[Vec2d]] = {}
        self.enemy_seals: dict[Gear, tuple[Actor, float]] = {}
        self.seals: list[SealField] = []
        self.whips: dict[Gear, WhipMotion] = {}
        self.loads: dict[Gear, LauncherLoad] = {}
        self.stowed: dict[Gear, Gear] = {}

    @property
    def now(self) -> float:
        return self.maze.wielding.time

    def held_by(self, actor: Actor) -> HeldGear | None:
        held = self.maze.wielding.held.get(actor)
        return held if actor.active and held is not None and held.item.holder is actor else None

    def press_primary(self, actor: Actor, cursor: Vec2d | None = None) -> None:
        held = self.held_by(actor)
        if held is None or held.primary_down or held.bow_lance_charging or held.rush is not None:
            return
        item = held.item
        if item.spec.effect == "catcher" and item in self.maze.wielding.snares:
            self.maze.wielding.unhook(item)
            return
        held.primary_down = True
        if item.spec.mode in ("melee", "shield"):
            self.maze.observe("aim", actor.aim, item.spec.effect)
        elif not item.spec.draw_time:
            self.maze.attack(actor, cursor=cursor)

    def release_primary(self, actor: Actor) -> None:
        held = self.held_by(actor)
        if held is None or not held.primary_down:
            return
        held.primary_down = False
        if held.item.spec.draw_time:
            charge, held.charge = held.charge, 0
            self.maze.attack(actor, charge=charge)

    def clear_input(self, actor: Actor) -> None:
        if actor is self.maze.player and self.maze.debug is not None:
            self.maze.debug.dragging_slider = False
        if (held := self.held_by(actor)) is not None:
            held.primary_down = False
            held.bow_lance_charging = False
            held.whip_extending = False
            if held.item.spec.draw_time:
                held.charge = 0
            self.end_shield_rush(actor, held, coast=False)
        self.maze.wielding.stop_rotation(actor)

    def update(self, actor: Actor, dt: float) -> None:
        if (dash := self.dashes.get(actor)) is not None:
            self.advance_sword(actor, dash, dt)
            return
        held = self.held_by(actor)
        if held is None:
            return
        spec = held.item.spec
        if actor is not self.maze.player and held.whip_extending and actor.swing_time == 0:
            held.whip_extending = False
        if held.rush is not None:
            self.advance_shield_rush(actor, held, dt)
            return
        if held.lock_target is not None and not held.lock_target.active:
            held.lock_target = None
        if held.primary_down:
            if spec.draw_time:
                held.charge = min(1.0, held.charge + dt / spec.draw_time)
                self.maze.observe("aim", held.item.axis, spec.effect)
            elif spec.effect == "flame":
                self.maze.attack(actor)
            elif spec.mode in ("melee", "shield"):
                self.maze.observe("aim", actor.aim, spec.effect)
        if held.bow_lance_charging:
            held.charge = min(1.0, held.charge + dt / BOW_LANCE_DRAW_TIME)
            if actor is self.maze.player:
                self.maze.observe("aim", held.item.axis, spec.effect)
        if held.gravity_charging:
            held.charge = min(1.0, held.charge + dt / GRAVITY_CHARGE_TIME)
            self.pull_nearby_gear(actor, held.item, held.charge, dt)
            if actor is self.maze.player:
                self.maze.observe("aim", held.item.axis, spec.effect)

    def attack_held(self, actor: Actor, held: HeldGear, charge: float, cursor: Vec2d | None = None) -> bool:
        item = held.item
        spec = item.spec
        attack = spec.attack
        maze = self.maze
        if spec.effect == "debug" and actor is maze.player and maze.debug is not None:
            return maze.debug.open("items", cursor)
        if spec.effect == "heal":
            if actor.hp == actor.max_hp:
                return False
            actor.hp += healing_pack_hp(actor.max_hp)
            maze.wielding.remove(item)
            actor.attack_cooldown = 0.45
            return True
        if spec.effect == "radar":
            if actor is maze.player:
                if not maze.start_radar():
                    return False
            elif (guard := maze.guard_for(actor)) is not None:
                maze.track_with_radar(guard)
            else:
                return False
            actor.attack_cooldown = 0.45
            item.activated_at = maze.wielding.time
            maze.wielding.wear(item)
            return True
        if attack is None:
            return False
        if spec.effect in ("grimoire", "seal", "launcher"):
            return item_effects.use_advanced_primary(self, actor, item, cursor)
        if spec.effect == "flame":
            self.fire(actor, item, attack, effect="flame")
            actor.attack_cooldown = attack.recovery
            item.flame_pulses = (item.flame_pulses + 1) % 5
            if item.flame_pulses == 1:
                self.mark_use(actor, item)
                maze.wielding.wear(item)
            return True
        if spec.mode in ("melee", "shield"):
            if actor is maze.player:
                return False
            actor.attack_cooldown = attack.recovery
            actor.swing_time = GUARD_SWING_TIME
            actor.swing_reach = 0
            maze.mob_sound(actor, "attack")
            return True
        if spec.effect == "gravity":
            return self.release_gravity(actor, held)
        if spec.effect == "bow":
            spike = replace(attack, damage=round(attack.damage * (0.75 + 0.25 * charge)),
                            reach=attack.reach * (0.62 + 0.38 * charge))
            self.fire(actor, item, spike, effect="bow_spike", speed=spec.projectile_speed * (0.66 + 0.34 * charge),
                      radius=1.2, shove=920 + 420 * charge, charge=charge)
        elif spec.draw_time:
            power = 0.46 + 0.54 * charge
            damage = max(2, round(attack.damage * power))
            target = held.lock_target if held.lock_target is not None and held.lock_target.active else None
            reach = 545 if target is not None else attack.reach * power
            speed = LOCKED_SHOT_SPEED if target is not None else spec.projectile_speed * power
            shove = (475 + 230 * charge) if target is not None else (395 + 228 * charge)
            self.fire(actor, item, replace(attack, reach=reach, damage=damage),
                      speed=speed, shove=shove, target=target)
        else:
            shove = {"pulse": PULSE_SHOVE, "beam": LASER_SHOVE, "wand": WAND_SHOVE}.get(spec.effect, 530)
            self.fire(actor, item, attack, effect=spec.effect, radius=3 if spec.effect == "wand" else 2,
                      shove=shove)
        actor.attack_cooldown = attack.recovery
        self.mark_use(actor, item)
        maze.wielding.wear(item)
        return True

    def can_special(self, item: Gear) -> bool:
        if item.spec.effect in EPIC_EFFECTS:
            return item_effects.can_use_advanced_special(self, item)
        if item.spec.effect == "spear":
            return item in self.maze.wielding.snares and self.maze.wielding.time >= item.special_ready
        return (
            item.spec.effect in SPECIAL_COOLDOWN and self.maze.wielding.time >= item.special_ready
            and item.durability >= USE_WEAR
        )

    def special(self, actor: Actor, cursor: Vec2d | None = None) -> bool:
        # Players and guards share these item-specific abilities and their cooldowns.
        held = self.held_by(actor)
        if held is None or actor in self.dashes:
            return False
        item = held.item
        effect = item.spec.effect
        maze = self.maze
        if effect == "debug" and actor is maze.player and maze.debug is not None:
            return maze.debug.open("mobs", cursor)
        if effect in EPIC_EFFECTS:
            return item_effects.use_advanced_special(self, actor, held, cursor)
        if effect == "gravity" and held.gravity_charging:
            held.gravity_charging = False
            held.charge = 0
            return True
        if not self.can_special(item):
            return False
        if effect == "gravity":
            held.gravity_charging = True
            held.charge = 0
            return True
        if effect == "bow":
            if held.bow_lance_charging:
                return False
            held.primary_down = False
            held.bow_lance_charging = True
            held.charge = 0
            return True
        if effect == "shield":
            return self.start_shield_rush(actor, held)
        if effect == "catcher":
            if not maze.wielding.take_control(item):
                return False
        elif effect == "hammer":
            origin = self.muzzle(actor, item)
            self.radial(actor, item, origin, 84, 4, 715 * HAMMER_KNOCKBACK_SCALE, (244, 208, 149))
        elif effect == "sword":
            if held.sword_destination is None:
                return self.mark_sword(actor, cursor if cursor is not None else actor.position + actor.aim * 90)
            if not self.start_sword(actor, held):
                return False
        elif effect == "spear":
            victim = maze.wielding.snares[item].victim
            maze.wielding.unhook(item)
            item.impact_times[victim.shape] = maze.wielding.time + 0.24
            item.body.apply_impulse_at_local_point((-item.body.mass * 160, 0))
        elif effect == "sling":
            target = maze.closest_enemy(actor)
            if target is None or target is held.lock_target:
                return False
            held.lock_target = target
        elif effect == "beam":
            self.ricochet(actor, item)
        elif effect == "flame":
            self.fire(actor, item, MOLOTOV, effect="molotov", speed=474, radius=5, shove=225)
        elif effect == "wand":
            self.wand_fan(actor, item)
        elif effect == "pulse":
            self.radial(actor, item, actor.position, *PULSE_BURST)
        self.mark_use(actor, item)
        item.special_ready = maze.wielding.time + SPECIAL_COOLDOWN[effect]
        actor.attack_cooldown = max(actor.attack_cooldown, 0.24)
        if effect != "sword":
            maze.wielding.wear(item, USE_WEAR)
        return True

    def release_special(self, actor: Actor) -> bool:
        held = self.held_by(actor)
        if held is None:
            return False
        if held.whip_extending:
            held.whip_extending = False
            return True
        if held.rush is not None:
            self.end_shield_rush(actor, held)
            return True
        if not held.bow_lance_charging:
            return False
        item = held.item
        charge = held.charge
        held.bow_lance_charging = False
        held.charge = 0
        if not self.can_special(item):
            return False
        attack = AttackProfile(285 + 255 * charge, round(9 + 5 * charge), 0.72, "piercing")
        self.fire(actor, item, attack, effect="bow_lance", speed=680 + 300 * charge,
                  radius=2.4, shove=580 + 400 * charge, charge=charge)
        actor.attack_cooldown = max(actor.attack_cooldown, attack.recovery)
        item.special_ready = self.maze.wielding.time + SPECIAL_COOLDOWN["bow"]
        self.mark_use(actor, item)
        self.maze.wielding.wear(item, USE_WEAR)
        return True

    def start_shield_rush(self, actor: Actor, held: HeldGear) -> bool:
        if held.rush is not None or actor.stamina <= 0 or actor.dodge_time > 0 or actor.restraint is not None:
            return False
        now = self.maze.wielding.time
        ends_at = inf if actor is self.maze.player else now + 0.75
        starting_speed = max(180, actor.body.velocity.dot(actor.aim))
        held.rush = ShieldRush(actor.aim, ends_at, min(SHIELD_RUSH_SPEED, starting_speed))
        held.primary_down = False
        actor.swing_time = 0
        held.item.special_ready = now + SPECIAL_COOLDOWN["shield"]
        self.mark_use(actor, held.item)
        return True

    def end_shield_rush(self, actor: Actor, held: HeldGear, *, coast: bool = True) -> None:
        rush = held.rush
        if rush is None:
            return
        held.rush = None
        if coast:
            actor.push(rush.direction * min(145, rush.speed * 0.4))

    def advance_shield_rush(self, actor: Actor, held: HeldGear, dt: float) -> None:
        rush = held.rush
        now = self.maze.wielding.time
        interrupted = actor.dodge_time > 0 or actor.restraint is not None or actor.stamina <= 0
        if interrupted or now >= rush.ends_at:
            self.end_shield_rush(actor, held, coast=not interrupted)
            return
        # Turn by a bounded angle per tick. Holding the button keeps the forward momentum.
        turn = SHIELD_RUSH_TURN * dt
        angle = rush.direction.get_angle_between(actor.aim)
        rush.direction = rush.direction.rotated(max(-turn, min(turn, angle)))
        rush.speed = min(SHIELD_RUSH_SPEED, rush.speed + SHIELD_RUSH_ACCELERATION * dt)
        actor.sprinting = False
        actor.desired_velocity = rush.direction * rush.speed
        actor.spend_stamina(SHIELD_RUSH_STAMINA * dt)
        held.item.special_ready = now + SPECIAL_COOLDOWN["shield"]

    def release_gravity(self, actor: Actor, held: HeldGear) -> bool:
        if not held.gravity_charging:
            return False
        charge = held.charge
        held.gravity_charging = False
        held.charge = 0
        item = held.item
        if not self.can_special(item):
            return False
        attack = item.spec.attack
        self.fire(actor, item, replace(attack, reach=158 + 180 * charge), effect="gravity",
                  speed=365 + 160 * charge, radius=3 + 4 * charge, charge=charge)
        now = self.maze.wielding.time
        item.special_ready = now + SPECIAL_COOLDOWN["gravity"]
        actor.attack_cooldown = attack.recovery
        self.mark_use(actor, item)
        self.maze.wielding.wear(item, USE_WEAR)
        return True

    def pull_nearby_gear(self, actor: Actor, emitter: Gear, charge: float, dt: float) -> None:
        maze = self.maze
        radius = 44 + 118 * charge
        for item in maze.wielding.items:
            if item is emitter or item.spec.effect in ("key", "mirror"):
                continue
            offset = actor.position - item.position
            distance = offset.length
            if 6 < distance < radius and maze.clear_path(actor.position, item.position):
                acceleration = (710 + 2260 * charge) * (1 - 0.55 * distance / radius)
                velocity = item.body.velocity + offset.normalized() * acceleration * dt / item.body.mass ** 0.5
                item.body.velocity = velocity.normalized() * min(640, velocity.length)
                maze.wielding.attribute_motion(item.shape, actor, 0.75, delivery="magic", weapon="Gravity Gun")

    def muzzle(self, actor: Actor, item: Gear) -> Vec2d:
        muzzle = item.point_on_axis(1) + item.axis * 3
        hit = self.maze.hit_solid(actor.position, muzzle)
        return muzzle if hit is None else hit.point + hit.normal * 4

    def fire(self, actor: Actor, item: Gear, attack: AttackProfile, *, effect: str | None = None,
             speed: float | None = None, radius: float = 2, shove: float = 515,
             charge: float = 0, spread: float = 0, target: Actor | None = None) -> None:
        direction = item.axis.rotated(spread)
        start = self.muzzle(actor, item)
        if target is not None and target.position != start:
            direction = (target.position - start).normalized()
        shot = Projectile(actor, item.shape, start, direction, attack, attack.reach,
                          item.spec.mode, speed or item.spec.projectile_speed, effect, radius, shove, charge,
                          target, item.spec.name, power=actor.weapon_power(item.spec))
        maze = self.maze
        if effect == "bow_spike":
            shot.spike = Spike(maze.wielding.time + 1.5, 17 + 6 * charge)
            shot.weapon = "Bow spike"
        elif effect == "bow_lance":
            shot.spike = Spike(maze.wielding.time + 1.5, 60 + 30 * charge, piercing=True, hp=6)
            shot.weapon = "Bow lance"
        if effect == "beam":
            end = start + direction * shot.remaining
            contact = resolve_shot(maze, shot, end)
            maze.beams.append(Beam(start, end if contact is None else contact, maze.wielding.time + 0.14))
        elif effect == "flame":
            spray_flame(maze, shot)
            return
        else:
            maze.projectiles.append(shot)
        item.body.apply_impulse_at_world_point(-direction * (11 + charge * 8), item.position)

    def wand_fan(self, actor: Actor, item: Gear) -> None:
        for angle in (-0.19, 0.0, 0.19):
            self.fire(actor, item, WAND_FAN, effect="wand", radius=3, spread=angle, shove=WAND_SHOVE)

    def mark_use(self, actor: Actor, item: Gear) -> None:
        maze = self.maze
        item.activated_at = maze.wielding.time
        if actor is maze.player:
            maze.observe("aim", actor.aim if item.spec.mode in ("melee", "shield") else item.axis, item.spec.effect)
            maze.make_noise(actor.position, 4)
        else:
            maze.mob_sound(actor, "attack")

    def area_targets(self, actor: Actor, item: Gear, origin: Vec2d, radius: float,
                     direction: Vec2d | None, half_angle: float):
        maze = self.maze
        struck_gear = set()
        for shape in maze.physics.shapes_near(origin, radius):
            if shape is actor.shape:
                continue
            gear = maze.wielding.by_shape.get(shape)
            if gear is not None and (gear is item or gear in struck_gear):
                continue
            query = shape.point_query(origin)
            if query.distance > radius:
                continue
            point = shape.body.position if shape in maze.actors_by_shape else query.point
            offset = point - origin
            if direction is not None and offset.length and offset.normalized().dot(direction) < cos(half_angle):
                continue
            obstruction = maze.hit_solid(origin, point)
            if obstruction is not None and obstruction.shape is not shape:
                continue
            if gear is not None:
                struck_gear.add(gear)
            yield shape, query.point, max(0, query.distance), gear

    def radial(self, actor: Actor, item: Gear, origin: Vec2d, radius: float, damage: int, shove: float,
               color: tuple[int, int, int], *, direction: Vec2d | None = None, half_angle: float = 0,
               kind: str = "arcane") -> None:
        maze = self.maze
        attack = AttackProfile(radius, damage, 0, kind)
        for shape, contact, distance, gear in list(self.area_targets(actor, item, origin, radius, direction, half_angle)):
            force = 1 - 0.55 * distance / radius
            victim = maze.actors_by_shape.get(shape)
            push = ((victim.position if victim else contact) - origin).normalized() * shove * force
            if gear is not None:
                gear.body.apply_impulse_at_world_point(push * 0.6, contact)
                maze.wielding.attribute_motion(gear.shape, actor, 0.9, weapon=item.spec.name)
            if damage:
                delivery = "magic" if item.spec.mode == "magic" else "projectile" if item.spec.mode == "ranged" else "melee"
                maze.apply_hit(actor, shape, attack, force * actor.weapon_power(item.spec),
                               position=contact, shove=push, delivery=delivery, weapon=item.spec.name)
                if kind == "heat" and victim is not None:
                    maze.ignite_actor(victim, actor, actor.weapon_power(item.spec), weapon=item.spec.name)
            elif victim is not None and victim.active:
                maze.knock_actor(actor, victim, push, contact)
        maze.weapon_flashes.append(WeaponFlash(origin, radius, color, maze.wielding.time, direction, half_angle))

    def mark_sword(self, actor: Actor, point: Vec2d) -> bool:
        held = self.held_by(actor)
        if held is None or held.item.spec.effect != "sword" or not self.can_special(held.item):
            return False
        margin = actor.radius + 2
        point = Vec2d(max(margin, min(COLS * TILE - margin, point.x)),
                      max(margin, min(ROWS * TILE - margin, point.y)))
        if point.get_distance(actor.position) < 8:
            return False
        held.sword_destination = point
        return True

    def start_sword(self, actor: Actor, held: HeldGear) -> bool:
        destination = held.sword_destination
        if actor.restraint is not None:
            return False
        held.sword_destination = None
        if destination is None or destination.get_distance(actor.position) < 8:
            return False
        self.dashes[actor] = SwordDash(held.item, destination)
        actor.dodge_time = 0
        actor.external_velocity = STOP
        self.maze.make_noise(actor.position, 9)
        return True

    def end_sword(self, actor: Actor) -> None:
        if (dash := self.dashes.pop(actor, None)) is not None:
            self.maze.wielding.wear(dash.item, USE_WEAR)

    def advance_sword(self, actor: Actor, dash: SwordDash, dt: float) -> None:
        maze = self.maze
        held = self.held_by(actor)
        if held is None or held.item is not dash.item or actor.restraint is not None:
            self.end_sword(actor)
            return
        start = actor.position
        guard = maze.guard_for(actor)
        destination = maze.limit_to_mirror_territory(dash.destination) if guard is not None and guard.boss else dash.destination
        toward = destination - start
        direction = toward.normalized()
        distance = min(toward.length, SWORD_DASH_SPEED * dt)
        end = start + direction * distance
        if distance <= 0:
            self.end_sword(actor)
            return
        actor.desired_velocity = STOP
        actor.external_velocity = STOP
        actor.aim = actor.facing = direction
        hits = maze.physics.sweep(start, end, actor.radius + 7, SOLID | ACTOR | GEAR_CATEGORY | SPIKE_CATEGORY)
        halted = False
        for hit in sorted(hits, key=lambda contact: contact.alpha):
            gear = maze.wielding.by_shape.get(hit.shape)
            shape = gear.shape if gear is not None else hit.shape
            if shape is actor.shape or gear is dash.item or shape in dash.struck:
                continue
            dash.struck.add(shape)
            self.slash_contact(actor, dash.item, hit, gear, direction)
            if not actor.alive:
                return
            solid_left = (shape.filter.categories == SOLID and maze.physics.contains(shape))
            if solid_left or dash.item.holder is not actor:
                end = start + direction * max(0, distance * hit.alpha - 1.5)
                halted = True
                break
        displacement = end - start
        actor.body.position = end
        actor.body.velocity = STOP
        maze.physics.refresh_body(actor.body)
        if dash.item.holder is actor:
            held.hand.position += displacement
            held.hand.velocity = STOP
            dash.item.body.position += displacement
            maze.physics.refresh_body(dash.item.body)
        maze.beams.append(Beam(start, end, maze.wielding.time + 0.22, 9, slashing=True))
        if halted or distance >= toward.length - 1e-6:
            self.end_sword(actor)

    def slash_contact(self, actor: Actor, blade: Gear, hit: SweepHit,
                      gear: Gear | None, direction: Vec2d) -> None:
        maze = self.maze
        shape, position = hit.shape, hit.point
        victim = maze.actors_by_shape.get(shape)
        solid = shape.filter.categories == SOLID
        if solid:
            damage, recoil, resistance = 11, 5, 1.0
        elif gear is not None:
            damage, recoil, resistance = (10, 3, 0.55) if gear.spec.mode == "shield" else (7, 2, 0.35)
            gear.body.apply_impulse_at_world_point(direction * 430, position)
            maze.wielding.attribute_motion(gear.shape, actor, 1.1, weapon="Sword dash")
        elif victim is not None:
            guard = maze.guard_for(victim)
            if guard is not None and guard.boss:
                damage, recoil, resistance = 12, 4, 0.34
            elif victim.kind in ("elite", "armored"):
                damage, recoil, resistance = 9, 3, 0.24
            elif victim.kind in ("scout", "claw", "guard") and victim not in maze.targets:
                damage, recoil, resistance = victim.hp, 1, 0.12
            else:
                damage, recoil, resistance = 7, 2, 0.17
        elif shape in maze.wielding.spikes:
            damage, recoil, resistance = 6, 0, 0
        else:
            return
        maze.apply_hit(actor, shape, AttackProfile(0, damage, 0, "arcane"),
                       position=position, shove=direction * 690, weapon="Sword dash")
        impulse = (actor.body.mass + blade.body.mass) * SWORD_DASH_SPEED * resistance
        maze.wielding.wear(blade, impact_wear(impulse, victim is None))
        label = "Wall" if solid else gear.spec.name if gear else "Enemy"
        maze.apply_hit(None, actor.shape, AttackProfile(0, recoil, 0, "arcane"),
                       position=position, weapon=f"Sword dash into {label.lower()}")

    def ricochet(self, actor: Actor, item: Gear) -> None:
        maze = self.maze
        start, direction = self.muzzle(actor, item), item.axis
        remaining = 720
        for bounce in range(4):
            end = start + direction * remaining
            hit_wall = maze.hit_solid(start, end, 5)
            if hit_wall is not None:
                end = start.interpolate_to(end, hit_wall.alpha)
            hits = maze.physics.sweep(start, end, 5, ALL_CATEGORIES & ~SOLID)
            struck = set()
            attack = AttackProfile(remaining, 8, 0, "piercing")
            blocked = False
            for hit in sorted(hits, key=lambda query: query.alpha):
                gear = maze.wielding.by_shape.get(hit.shape)
                if hit.shape is item.shape or hit.shape is actor.shape and bounce == 0:
                    continue
                target = gear.shape if gear is not None else hit.shape
                if target in struck:
                    continue
                struck.add(target)
                source = None if hit.shape is actor.shape else actor
                maze.apply_hit(source, target, attack, (1 - 0.16 * bounce) * actor.dexterity_multiplier,
                               position=hit.point, shove=direction * (865 - 82 * bounce),
                               delivery="projectile", weapon="Ricochet laser")
                if gear is not None and gear.spec.mode == "shield":
                    end, blocked = hit.point, True
                    break
            maze.beams.append(Beam(start, end, maze.wielding.time + 0.21, 9))
            remaining -= start.get_distance(end) + 3
            if blocked or hit_wall is None or remaining < 10:
                break
            maze.apply_hit(actor, hit_wall.shape, attack, actor.dexterity_multiplier, position=hit_wall.point)
            direction = direction - hit_wall.normal * (2 * direction.dot(hit_wall.normal))
            start = end + hit_wall.normal * 3
