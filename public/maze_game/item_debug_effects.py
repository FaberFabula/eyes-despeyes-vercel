"""Standalone sandbox effects that call the normal weapon and mob mechanics."""

from dataclasses import dataclass
from math import pi


from . import item_effects
from .physics import Vec2d
from .item_actions import LASER_SHOVE, MOLOTOV, PULSE_BURST, PULSE_SHOVE, WAND_SHOVE
from .effects import GravityWell, WeaponFlash
from .item import (
    BLADE,
    BOW,
    CLAW_ATTACK,
    FLAMETHROWER,
    GRAVITY_GUN,
    GRIMOIRE,
    ALL_LAUNCHER,
    HAMMER,
    MAN_CATCHER,
    PULSE_GUN,
    SCOUT_BITE,
    SCOUT_SCRATCH,
    SEALING,
    SHIELD,
    SLINGSHOT,
    SPEAR,
    WAND,
    WHIP,
    LASER,
    AttackProfile,
    Equipment,
)
from .actor import Actor
from .item_effects import DrainField, EPIC_PURPLE, WhipMotion
from .enemy import Enemy
from .enemy_behaviors import ATTACK_CUES, WINDUP, EnemyAttack, reveal_disguise
from .item_physics import HAMMER_KNOCKBACK_SCALE, HAND_REACH
from .item import Gear
from .layout import center


@dataclass(frozen=True)
class ShotEffect:
    spec: Equipment
    effect: str | None
    radius: float = 2
    speed: float | None = None
    shove: float = 530
    charge: float = 0


SHOTS = {
    "bow_spike": ShotEffect(BOW, "bow_spike", radius=1.2, shove=1340, charge=1),
    "bow_lance": ShotEffect(BOW, "bow_lance", radius=2.4, shove=980, charge=1),
    "sling": ShotEffect(SLINGSHOT, "sling", shove=623, charge=1),
    "sling_lock": ShotEffect(SLINGSHOT, "sling", speed=1630, shove=705, charge=1),
    "beam": ShotEffect(LASER, "beam", shove=LASER_SHOVE),
    "pulse": ShotEffect(PULSE_GUN, "pulse", shove=PULSE_SHOVE),
    "gravity": ShotEffect(GRAVITY_GUN, "gravity", radius=7, speed=525, charge=1),
    "flame": ShotEffect(FLAMETHROWER, "flame", shove=520),
    "wand": ShotEffect(WAND, "wand", radius=3, shove=WAND_SHOVE),
    "molotov": ShotEffect(FLAMETHROWER, "molotov", radius=5, speed=474, shove=225),
}
MELEE = {"sword": BLADE, "spear": SPEAR, "catcher": MAN_CATCHER, "whip": WHIP, "shield": SHIELD}
RADIALS = {
    "hammer": (84, 4, 715 * HAMMER_KNOCKBACK_SCALE, (244, 208, 149)),
    "pulse_burst": PULSE_BURST,
}
INNATE = {"scratch": SCOUT_SCRATCH, "bite": SCOUT_BITE, "claw": CLAW_ATTACK}
EFFECT_OPTIONS = (
    ("Hammer ground slam", "hammer"), ("Sword dash", "sword"),
    ("Spear thrust", "spear"), ("Man Catcher clamp", "catcher"),
    ("Shield rush", "shield"), ("Nine-tail whip sweep", "whip"),
    ("Bow spike", "bow_spike"), ("Bow piercing lance", "bow_lance"),
    ("Slingshot stone", "sling"), ("Slingshot target lock shot", "sling_lock"),
    ("Laser beam", "beam"), ("Ricocheting laser", "ricochet"),
    ("Pulse ring", "pulse"), ("Pulse shockwave", "pulse_burst"),
    ("Gravity core", "gravity"), ("Flamethrower spray", "flame"), ("Firebomb", "molotov"),
    ("Fire patch", "fire"), ("Wand bolt", "wand"),
    ("Wand triple cast", "triple"), ("Grimoire fire blast", "explosion"),
    ("Grimoire drain", "drain"), ("Grimoire backfire", "backfire"),
    ("Grimoire detonation", "book_explosion"), ("Grimoire lightning", "lightning"),
    ("Grimoire gravity well", "grimoire_well"), ("Sealing net", "seal"),
    ("All-launcher mob launch", "launcher_mob"), ("All-launcher item launch", "launcher_item"),
    ("Radar blinks", "radar"), ("Eye Scout scratch", "scratch"),
    ("Eye Scout bite", "bite"), ("Claw Eye slash", "claw"),
    ("Mimic impale", "impale"), ("Mimic tendril swing", "swing"),
    ("Mimic body slam", "bash"), ("Merchant call for help", "help"),
    ("Scribe call for guards", "scribe_call"), ("Angry Eye", "eye"),
)


@dataclass
class TimedEffect:
    actor: Actor
    item: Gear | None
    started: float
    expires: float
    kind: str
    mimic: Enemy | None = None


class DebugEffects:
    def __init__(self, maze):
        self.maze = maze
        self.active: list[TimedEffect] = []

    def emitter(self, spec: Equipment, position: Vec2d) -> Gear:
        axis = self.maze.player.aim
        # Position the working end at the requested point.
        item = self.maze.wielding.add(spec, position - axis * (spec.length / 2 + 3), axis.angle)
        item.temporary = True
        return item

    def trigger(self, kind: str, position: Vec2d) -> bool:
        maze = self.maze
        player = maze.player
        now = maze.wielding.time
        if kind in SHOTS:
            shot = SHOTS[kind]
            item = self.emitter(shot.spec, position)
            target = maze.closest_enemy(player) if kind == "sling_lock" else None
            attack = shot.spec.attack
            if kind == "bow_lance":
                attack = AttackProfile(540, 14, 0.72, "piercing")
            elif kind == "molotov":
                attack = MOLOTOV
            maze.item_actions.fire(player, item, attack, effect=shot.effect, radius=shot.radius, speed=shot.speed,
                             shove=shot.shove, charge=shot.charge, target=target)
            maze.wielding.remove(item)
        elif kind in MELEE:
            return self.weapon_effect(kind, position)
        elif kind in RADIALS:
            spec = HAMMER if kind == "hammer" else PULSE_GUN
            item = self.emitter(spec, position)
            maze.item_actions.radial(player, item, position, *RADIALS[kind])
            maze.wielding.remove(item)
        elif kind in WINDUP:
            return self.mimic_effect(kind, position)
        elif kind in INNATE:
            profile = INNATE[kind]
            point = player.position + player.aim * (profile.reach * 0.85)
            actor = self.actor(point)
            actor.kind = "claw" if kind == "claw" else "scout"
            actor.innate_attack = profile
            actor.aim = -player.aim
            maze.attack(actor)
            maze.weapon_flashes.append(WeaponFlash(point, profile.reach, (216, 139, 116), now,
                                                    actor.aim, pi / 5))
            self.active.append(TimedEffect(actor, None, now, now + 0.3, kind))
        elif kind == "fire":
            maze.ignite(position, player, 1.0)
        elif kind in ("explosion", "book_explosion", "lightning"):
            item = self.emitter(GRIMOIRE, position)
            if kind == "lightning":
                item_effects.grimoire_lightning(maze.item_actions, player, item, position)
            else:
                item_effects.grimoire_blast(maze.item_actions, player, item, position, overload=kind == "book_explosion")
            maze.wielding.remove(item)
        elif kind == "grimoire_well":
            maze.gravity_wells.append(GravityWell(position, player, now + 2.8, 140,
                                                 player.intelligence_multiplier, GRIMOIRE.name))
        elif kind == "drain":
            maze.item_actions.drains.append(DrainField(player, position, now + 3.1, now, player.intelligence_multiplier))
        elif kind == "backfire":
            maze.apply_hit(None, player.shape, AttackProfile(0, 1, 0, "arcane"),
                           delivery="magic", weapon="Grimoire backfire", unavoidable=True)
            maze.weapon_flashes.append(WeaponFlash(player.position, 33, (247, 107, 93), now))
        elif kind == "seal":
            points = [position + Vec2d(48, 0).rotated(index * pi / 3) for index in range(6)]
            if any(maze.hit_solid(position, point) is not None for point in points):
                return False
            item = self.emitter(SEALING, position)
            maze.item_actions.marks[item] = points
            cast = item_effects.bind_seal(maze.item_actions, player, item)
            maze.wielding.remove(item)
            return cast
        elif kind == "radar":
            maze.radar_started = now - 10
            maze.start_radar()
        elif kind == "eye":
            if maze.gate_exposed_at is None:
                maze.gate_exposed_at = now
            maze.announce_eye_anger()
        elif kind == "help":
            maze.make_cue("help", position, 12)
        elif kind == "scribe_call":
            actor = self.actor(position)
            actor.kind = "scribe"
            maze.call_armored_guards(actor)
            self.active.append(TimedEffect(actor, None, now, now + 0.15, kind))
        elif kind == "ricochet":
            item = self.emitter(LASER, position)
            maze.item_actions.ricochet(player, item)
            maze.wielding.remove(item)
        elif kind == "triple":
            item = self.emitter(WAND, position)
            maze.item_actions.wand_fan(player, item)
            maze.wielding.remove(item)
        elif kind in ("launcher_mob", "launcher_item"):
            actions = maze.item_actions
            targets = maze.enemies_of(player) if kind == "launcher_mob" else [
                gear for gear in maze.wielding.items if gear.holder is None and not gear.temporary
            ]
            target = min((target for target in targets if not any(target in seal.captives for seal in actions.seals)),
                         key=lambda target: target.position.get_dist_sqrd(position), default=None)
            if target is None:
                return False
            item = self.emitter(ALL_LAUNCHER, position)
            loaded = item_effects.capture_launcher_payload(actions, player, item, target)
            if loaded and not item_effects.unload_launcher(actions, item, shooter=player):
                item_effects.unload_launcher(actions, item)
            maze.wielding.remove(item)
            return loaded
        else:
            return False
        return True

    def actor(self, position: Vec2d) -> Actor:
        maze = self.maze
        actor = Actor(maze.physics, (0, 0), speed=100, hp=80, kind="player")
        actor.body.position = position
        actor.aim = actor.facing = maze.player.aim
        # A summoned weapon has a force-driven hand without an extra solid character.
        maze.physics.suspend_collisions(actor.shape)
        maze.add_actor(actor)
        return actor

    def weapon_effect(self, kind: str, position: Vec2d) -> bool:
        maze = self.maze
        spec = MELEE[kind]
        cell = maze.debug.spawn_cell(spec)
        item = maze.place_gear(spec, cell) if cell is not None else None
        if item is None:
            return False
        item.temporary = True
        actor = self.actor(item.point_on_axis(-1) - item.axis * 18)
        actor.aim = actor.facing = item.axis
        maze.wielding.grab(actor, item, item.point_on_axis(-0.7))
        held = maze.wielding.held[actor]
        held.primary_down = True
        duration = 0.95
        if kind == "sword":
            held.sword_destination = position + actor.aim * 138
            maze.item_actions.start_sword(actor, held)
        elif kind == "shield":
            maze.item_actions.start_shield_rush(actor, held)
            duration = 1.1
        elif kind == "whip":
            maze.item_actions.whips[item] = WhipMotion(count=9)
            held.whip_extending = True
        now = maze.wielding.time
        actor.swing_time = duration
        self.active.append(TimedEffect(actor, item, now, now + duration, kind))
        return True

    def mimic_effect(self, kind: str, position: Vec2d) -> bool:
        maze = self.maze
        cell = maze.debug.spawn_cell()
        if cell is None:
            return False
        mimic = maze.spawn_mimic(cell)
        mimic.preview_attack = True
        actor = mimic
        actor.body.position = center(cell)
        maze.physics.refresh_shape(actor.shape)
        reveal_disguise(mimic, maze)
        toward = maze.player.position - actor.position
        direction = toward.normalized() if toward.length else -maze.player.aim
        actor.aim = actor.facing = direction
        mimic.attack = EnemyAttack(kind, direction, maze.wielding.time + WINDUP[kind], 1, maze.player)
        mimic.attack_ready = float("inf")
        maze.mob_sound(actor, "attack", cue=ATTACK_CUES[kind])
        now = maze.wielding.time
        self.active.append(TimedEffect(actor, None, now, now + 3.0, kind, mimic))
        return True

    def update(self, dt: float) -> None:
        maze = self.maze
        now = maze.wielding.time
        for effect in tuple(self.active):
            actor, item = effect.actor, effect.item
            done = (now >= effect.expires or not actor.alive or (
                effect.mimic is not None and effect.mimic.attack is None
            ))
            if done:
                self.remove(effect)
                continue
            if item is None:
                continue
            if item.holder is not actor:
                self.remove(effect)
                continue
            actor.tick_timers(dt)
            held = maze.wielding.held[actor]
            if effect.kind != "sword":
                progress = (now - effect.started) / (effect.expires - effect.started)
                extension = 0.42 + 0.58 * min(1, progress * 3)
                angle = 0 if effect.kind in ("spear", "catcher", "shield") else -0.78 + 1.56 * min(1, progress * 2)
                direction = actor.aim.rotated(angle)
                maze.wielding.update_hand(actor, actor.position + direction * HAND_REACH * extension, dt)
            maze.item_actions.update(actor, dt)

    def remove(self, effect: TimedEffect) -> None:
        maze = self.maze
        actor, item = effect.actor, effect.item
        maze.item_actions.end_sword(actor)
        maze.wielding.release_victim(actor)
        maze.wielding.release(actor)
        if item is not None and item.shape in maze.wielding.by_shape:
            maze.wielding.remove(item)
        if maze.physics.contains(actor.shape):
            maze.physics.remove_actor(actor.shape)
        actor.hp = 0
        if actor in maze.mimics:
            maze.mimics.remove(actor)
        maze.actors_by_shape.pop(actor.shape, None)
        if actor in maze.actors:
            maze.actors.remove(actor)
        self.active.remove(effect)
