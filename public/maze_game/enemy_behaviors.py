"""Disguise and scribe behavior for the shared Enemy class."""

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING


from .physics import Vec2d
from .enemy import Enemy
from .item import EPIC_ITEMS, GATE_KEY, HEAL_PACK, LOOT, AttackProfile
from .actor import Actor
from .layout import TILE, Cell, center, find_path, tile_at
from .memory import PlayerHabits, lateral_axis
from .mimic_body import (
    MIMIC_BASE_RADIUS, MIMIC_FOLDED_RADIUS, edible_gear, fit_mimic, flee_mimic, fold_mimic,
    mob_growth, swallow_gear, travel_mimic,
)

if TYPE_CHECKING:
    from .world import Maze

TRANSFORM_DELAY = 0.45
CHASE_LOSS = 2.0
MIMIC_FORAGE_REACH = 3 * TILE
MIMIC_HUNT_RADIUS = 6 * TILE
WRITE_SECONDS = 1.8
ANKLE_GAP = 10
DISGUISES = ("item", "merchant", "chest", "key", "door", "gate")
DISGUISE_TINTS = ((255, 218, 218), (255, 234, 207))
WALL_DISGUISES = ("door", "gate")
STOP = Vec2d(0, 0)

WINDUP = {"impale": 0.40, "swing": 0.52, "bash": 0.46}
ATTACKS = {
    "impale": AttackProfile(116, 5, 1.25, "piercing"),
    "swing": AttackProfile(101, 4, 1.40),
    "bash": AttackProfile(172, 7, 1.55),
}
ATTACK_NAMES = {"impale": "Mimic spike", "swing": "Mimic tendril", "bash": "Mimic body slam"}
ATTACK_CUES = {"impale": "mimic_attack", "swing": "mimic_swing", "bash": "mimic_bash"}
PIN_SECONDS = 1.10
SWING_SECONDS = 0.85


@dataclass
class EnemyAttack:
    kind: str
    direction: Vec2d
    ends_at: float
    swing_side: int
    prey: Actor
    phase: str = "windup"
    tether_reach: float = 0.0


def start_disguise(enemy: Enemy, maze: "Maze") -> None:
    enemy.eye_habits = maze.habits
    enemy.legs = maze.rng.randint(10, 13)
    enemy.full_radius = enemy.radius
    enemy.cargo = []
    enemy.nourishment = 0.0
    enemy.feed_at = 0.0
    enemy.prey: Actor | None = None
    enemy.hunt_at = 0.0
    enemy.split_safe_until = 0.0
    enemy.defeated_form = False
    enemy.preview_attack = False
    enemy.form = ""
    enemy.wall_cells: tuple[Cell, ...] = ()
    enemy.wall_face = STOP
    enemy.state = "disguised"
    enemy.hidden_for = 0.0
    enemy.next_roam = maze.rng.uniform(3, 7)
    enemy.disguise_until = 0.0
    enemy.player_was_inside = False
    enemy.departed = False
    enemy.exit_direction = STOP
    enemy.destination: Cell | None = None
    enemy.attack: EnemyAttack | None = None
    enemy.attack_ready = 0.0
    enemy.alt_step = False
    fold_mimic(enemy, maze)
    choose_disguise(enemy, maze, initial=True)


def is_disguised(enemy: Enemy) -> bool:
    return enemy.active and enemy.state == "disguised"


def disguise_eye_width(enemy: Enemy) -> int:
    samples = sum(enemy.eye_habits.count(kind) for kind in ("move", "dodge", "aim"))
    return max(2, round(8 / (1 + 0.015 * samples)))


def disguise_position(enemy: Enemy) -> Vec2d:
    if enemy.wall_cells:
        return sum((center(cell) for cell in enemy.wall_cells), STOP) / len(enemy.wall_cells)
    return enemy.position


def hit_disguise(enemy: Enemy, position: Vec2d) -> bool:
    if enemy.form not in ("item", "key"):
        return False
    offset = (position - enemy.position).rotated(enemy.tilt * enemy.imperfection * pi / 180)
    return abs(offset.x) <= enemy.disguise_item.length / 2 + 4 and abs(offset.y) <= enemy.disguise_item.width / 2 + 4


def feed_mimic(enemy: Enemy, maze: "Maze", *, forage: bool = False) -> bool:
    """Swallow loose gear and keep it in the stomach."""
    now = maze.wielding.time
    if not enemy.can_be_hostile or enemy.preview_attack or now < enemy.feed_at:
        return False
    if any(enemy in seal.captives for seal in maze.item_actions.seals):
        return False
    origin = enemy.position
    bite_reach = enemy.radius + 9
    search_reach = max(bite_reach, MIMIC_FORAGE_REACH) if forage else bite_reach
    food = []
    for shape in maze.physics.shapes_near(origin, search_reach):
        item = maze.wielding.by_shape.get(shape)
        if (item is None or shape is not item.shape or item.holder is not None
                or now < item.physics_safe_until or not edible_gear(maze, item)):
            continue
        hit = item.point_query(origin)
        food.append((item, hit.point if hit.distance > 0 else origin))
    for target, point in sorted(food, key=lambda candidate: origin.get_dist_sqrd(candidate[1])):
        distance = origin.get_distance(point)
        if distance <= bite_reach and maze.clear_path(origin, point):
            swallow_gear(enemy, maze, target)
            return True
        if (forage and distance <= search_reach
                and maze.clear_path(origin, point, min(MIMIC_FOLDED_RADIUS, enemy.radius))):
            travel_mimic(enemy, maze, point, 0)
            return True
    return False


def reveal_disguise(enemy: Enemy, maze: "Maze") -> None:
    if not enemy.active or enemy.state in ("spider", "fleeing"):
        return
    enemy.state = "spider"
    enemy.hidden_for = 0.0
    enemy.destination = None
    enemy.attack_ready = maze.wielding.time + 0.28
    enemy.wall_cells = ()
    fit_mimic(enemy, maze)
    enemy.alert.become_hostile()
    maze.make_cue("mimic", enemy.position, 10)


def choose_disguise(enemy: Enemy, maze: "Maze", *, initial: bool = False) -> None:
    options = [form for form in DISGUISES if form != enemy.form]
    maze.rng.shuffle(options)
    enemy.wall_cells = ()
    enemy.form = next(form for form in options
                     if form not in WALL_DISGUISES or mount_disguise(enemy, maze, form, initial))
    enemy.disguise_item = (GATE_KEY if enemy.form == "key"
                          else maze.rng.choice((*LOOT, HEAL_PACK, *EPIC_ITEMS)))
    enemy.epic_chest = enemy.form == "chest" and maze.rng.random() < 0.14
    epic_disguise = enemy.epic_chest or (enemy.form == "item" and enemy.disguise_item in EPIC_ITEMS)
    enemy.tint = (255, 218, 216) if epic_disguise else maze.rng.choice(DISGUISE_TINTS)
    enemy.eye_offset = Vec2d(maze.rng.uniform(-1, 1), maze.rng.uniform(-1, 1))
    enemy.tilt = maze.rng.choice((-1, 1))
    enemy.imperfection = max(1, 7 - maze.progress.level)


def mount_disguise(enemy: Enemy, maze: "Maze", form: str, initial: bool) -> bool:
    room = maze.layout.room_at(tile_at(enemy.position))
    if room is None:
        return False
    limit = float("inf") if initial else 2 * TILE
    if not initial and maze.player_sees(enemy.position):
        limit = TILE * 0.55
    choices = []
    for x, y in room.cells():
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            wall = (x + dx, y + dy)
            if wall not in maze.walls or maze.reinforced(wall):
                continue
            cells = (wall, (wall[0] + abs(dy), wall[1] + abs(dx))) if form == "gate" else (wall,)
            if any(cell not in maze.walls or maze.reinforced(cell) for cell in cells):
                continue
            faces = [(wx - dx, wy - dy) for wx, wy in cells]
            if any(not room.contains(face) for face in faces):
                continue
            face = Vec2d(dx, dy)
            wall_center = sum((center(cell) for cell in cells), STOP) / len(cells)
            position = wall_center - face * (TILE / 2 + enemy.radius + 2)
            distance = enemy.position.get_distance(position)
            if distance > limit or not maze.clear_path(enemy.position, position, enemy.radius):
                continue
            if any(other.active and other is not enemy
                   and other.position.get_distance(position) < enemy.radius + other.radius + 2
                   for other in maze.actors):
                continue
            choices.append((distance, position, cells, face))
    if not choices:
        return False
    _, enemy.body.position, enemy.wall_cells, enemy.wall_face = min(choices, key=lambda candidate: candidate[0])
    maze.physics.refresh_shape(enemy.shape)
    return True


def assume_disguise(enemy: Enemy, maze: "Maze", inside: bool) -> None:
    choose_disguise(enemy, maze)
    enemy.state = "disguised"
    enemy.player_was_inside = inside
    enemy.departed = False
    enemy.hidden_for = 0.0
    enemy.disguise_until = maze.wielding.time + 1.4
    enemy.next_roam = maze.wielding.time + maze.rng.uniform(4, 8)
    enemy.destination = None
    enemy.move(STOP)


def choose_destination(enemy: Enemy, maze: "Maze") -> Cell | None:
    start = tile_at(enemy.position)
    player_cell = tile_at(maze.player.position)
    room = maze.layout.room_at(start)
    player_room = maze.layout.room_at(player_cell)
    options = []
    for other in maze.layout.rooms:
        if other in (room, maze.layout.gate_room) or (enemy.departed and other is player_room):
            continue
        cells = other.cells()
        maze.rng.shuffle(cells)
        for cell in cells:
            if not maze.spawn_clear(cell, gap=22) or maze.player_sees(center(cell)):
                continue
            route = find_path(start, cell, maze.current_tile)
            if not route:
                continue
            first_exit = next((step for step in route if room is None or not room.contains(step)), route[0])
            direction = (center(first_exit) - center(start)).normalized()
            preference = direction.dot(enemy.exit_direction) if enemy.departed else other is not player_room
            options.append((preference, len(route), cell))
            break
    return min(options)[2] if options else None


def begin_relocation(enemy: Enemy, maze: "Maze") -> None:
    enemy.hidden_for = 0.0
    enemy.next_roam = maze.wielding.time + 4.0
    enemy.destination = choose_destination(enemy, maze)
    if enemy.destination is not None:
        enemy.state = "relocating"
        fold_mimic(enemy, maze)
        enemy.move(STOP)


def mimic_attack_reach(enemy: Enemy, kind: str) -> float:
    return ATTACKS[kind].reach + 1.4 * (enemy.radius - MIMIC_BASE_RADIUS)


def prey_style(prey: Actor) -> str:
    return prey.equipment.mode if prey.equipment is not None else "innate" if prey.innate_attack else "empty"


def choose_prey(enemy: Enemy, maze: "Maze") -> Actor | None:
    from .enemy_policy import finish_trial, start_trial

    now = maze.wielding.time
    last = enemy.prey
    if now < enemy.hunt_at and last is not None and last.active:
        return last
    if now < enemy.hunt_at and last is None:
        return None
    enemy.hunt_at = now + 0.35
    if not enemy.can_be_hostile:
        return None
    prey = [actor for actor in maze.actors
            if actor is not enemy and actor.active and actor.distance_to(enemy.position) <= MIMIC_HUNT_RADIUS
            and maze.clear_path(enemy.position, actor.position)]
    if not prey:
        return None
    context = ("small" if enemy.full_radius < MIMIC_BASE_RADIUS else "large",)

    def food_value(actor: Actor) -> float:
        distance = actor.distance_to(enemy.position)
        effort = 1 + distance / (2 * TILE) + actor.hp / 48
        outcome = enemy.mimic_memory.policy.estimate("mimic_hunt", context, prey_style(actor))[0]
        if maze.progress.hard_mode:
            outcome += 0.45 * maze.progress.policy.estimate("mimic_hunt", context, prey_style(actor))[0]
        return mob_growth(actor) / effort + 0.4 * outcome + 0.32 * (actor is last)

    target = max(prey, key=food_value)
    if target is not last:
        finish_trial(maze, enemy, "mimic_hunt")
    if enemy.state == "spider" and "mimic_hunt" not in enemy.trials:
        start_trial(maze, enemy, "mimic_hunt", context, prey_style(target), 2.6)
    return target


def begin_attack(enemy: Enemy, maze: "Maze", prey: Actor) -> bool:
    from .enemy_policy import aim_offset, decide

    if not enemy.can_be_hostile:
        return False
    offset = prey.position - enemy.position
    if (maze.wielding.time < enemy.attack_ready or enemy.dodge_time > 0 or prey.restraint is not None
            or offset.length > mimic_attack_reach(enemy, "bash")
            or not maze.clear_path(enemy.position, prey.position)):
        return False
    attacks = tuple(kind for kind in ATTACKS if offset.length <= mimic_attack_reach(enemy, kind))
    habits = enemy.combat_habits(maze)
    read = min(0.9, habits.combat_read() * enemy.learning_weight)
    dodge_bias = habits.bias("dodge") if prey is maze.player else 0.0
    context = (prey_style(prey), "close" if offset.length < TILE * 2 else "reach")
    kind = decide(maze, enemy, "mimic_attack", attacks, 1.4, context=context)
    target = (aim_offset(maze, enemy, offset, habits, WINDUP[kind], "melee")
              if prey is maze.player else offset + prey.body.velocity * 0.12)
    direction = target.normalized() if target.length else enemy.facing
    turn = 1 if dodge_bias > 0.12 else -1 if dodge_bias < -0.12 else maze.rng.choice((-1, 1))
    enemy.attack = EnemyAttack(kind, direction, maze.wielding.time + WINDUP[kind] * (1 - 0.17 * read), turn, prey)
    if isinstance(prey, Enemy) and prey.kind != "mimic":
        maze.notice_mimic(prey, enemy)
    enemy.aim = direction
    enemy.move(STOP)
    maze.mob_sound(enemy, "attack", cue=ATTACK_CUES[kind])
    return True


def finish_attack(enemy: Enemy, maze: "Maze") -> None:
    maze.wielding.release_tether(enemy)
    enemy.attack_ready = maze.wielding.time + ATTACKS[enemy.attack.kind].recovery
    enemy.attack = None


def strike_tendril(enemy: Enemy, maze: "Maze", attack: EnemyAttack) -> None:
    prey = attack.prey
    profile = ATTACKS[attack.kind]
    end = enemy.position + attack.direction * mimic_attack_reach(enemy, attack.kind)
    hit = prey.shape.segment_query(enemy.position, end, 3)
    if hit is None or not maze.clear_path(enemy.position, hit.point, 2):
        finish_attack(enemy, maze)
        return
    blocker = maze.shield_between(enemy.position, prey)
    target = blocker.shape if blocker is not None else prey.shape
    maze.apply_hit(enemy, target, profile, enemy.strength_multiplier,
                   position=hit.point, weapon=ATTACK_NAMES[attack.kind])
    swinging = attack.kind == "swing"
    if (blocker is not None or not prey.active or getattr(prey, "state", None) == "fleeing"
            or maze.wielding.tether_actor(
        enemy, prey, force=(7800 if swinging else 6300) * enemy.grip_multiplier,
        mobility=0.18 if swinging else 0.34,
    ) is None):
        finish_attack(enemy, maze)
        return
    attack.phase = "hold"
    attack.ends_at = maze.wielding.time + (SWING_SECONDS if swinging else PIN_SECONDS)
    offset = prey.position - enemy.position
    if offset.length:
        attack.direction = offset.normalized()
    attack.tether_reach = prey.position.get_distance(enemy.position)


def advance_attack(enemy: Enemy, maze: "Maze") -> None:
    attack = enemy.attack
    actor, prey = enemy, attack.prey
    if not prey.active:
        finish_attack(enemy, maze)
        return
    now = maze.wielding.time
    actor.aim = attack.direction
    actor.move(STOP)
    momentum = 1 + max(0, actor.radius - MIMIC_BASE_RADIUS) / 50
    if attack.phase == "windup":
        if now < attack.ends_at:
            return
        if attack.kind == "bash":
            attack.phase = "dash"
            attack.ends_at = now + 0.40
            actor.push(attack.direction * (540 * momentum))
            maze.make_noise(actor.position, 5)
        else:
            strike_tendril(enemy, maze, attack)
        return

    if attack.phase == "dash":
        offset = prey.position - actor.position
        contact = (offset.length <= actor.radius + prey.radius + 8
                   and offset.dot(attack.direction) >= -3
                   and maze.clear_path(actor.position, prey.position))
        if contact:
            blocker = maze.shield_between(actor.position, prey)
            target = blocker.shape if blocker is not None else prey.shape
            maze.apply_hit(actor, target, ATTACKS["bash"], actor.strength_multiplier,
                           position=prey.position - offset.normalized() * prey.radius,
                           shove=attack.direction * (1010 * momentum), weapon=ATTACK_NAMES["bash"])
        elif now < attack.ends_at and actor.external_velocity.dot(attack.direction) > 30:
            actor.move(attack.direction)
            return
        finish_attack(enemy, maze)
        return

    # The grab moves a real Pymunk anchor around the Mimic, so the victim can hit nearby walls.
    snare = maze.wielding.tethers.get(actor)
    if now >= attack.ends_at or snare is None:
        if attack.kind == "swing" and snare is not None:
            radial = (prey.position - actor.position).normalized()
            tangent = radial.rotated(attack.swing_side * pi / 2)
            maze.knock_actor(actor, prey, tangent * (770 * momentum), weapon=ATTACK_NAMES["swing"])
        finish_attack(enemy, maze)
    elif attack.kind == "swing":
        progress = 1 - (attack.ends_at - now) / SWING_SECONDS
        radial = attack.direction.rotated(progress * 2.6 * attack.swing_side)
        snare.joint.anchor_a = radial.rotated(-actor.body.angle) * attack.tether_reach
        actor.move(-radial.rotated(attack.swing_side * pi / 2))
        maze.wielding.attribute_motion(prey.shape, actor, 0.4, weapon=ATTACK_NAMES["swing"])
    else:
        actor.move(-attack.direction * 0.5)


def tendril_tip(enemy: Enemy, maze: "Maze") -> Vec2d | None:
    attack = enemy.attack
    if attack is None or attack.kind == "bash":
        return None
    if attack.phase == "hold":
        return attack.prey.position
    growth = 1 - (attack.ends_at - maze.wielding.time) / WINDUP[attack.kind]
    reach = enemy.radius + max(0.0, growth) * (mimic_attack_reach(enemy, attack.kind) - enemy.radius)
    end = enemy.position + attack.direction * reach
    obstruction = maze.hit_solid(enemy.position, end, 2)
    return obstruction.point if obstruction is not None else end


def update_disguise(enemy: Enemy, maze: "Maze", dt: float) -> None:
    enemy.tick_timers(dt)
    enemy.update_stamina(dt)
    if flee_mimic(enemy, maze, dt):
        enemy.alert.visible = False
        return
    prey = enemy.attack.prey if enemy.attack is not None else choose_prey(enemy, maze)
    enemy.prey = prey
    visible = maze.player_sees(enemy.position)
    if (enemy.state != "spider" and prey is not None and prey is not maze.player
            and enemy.distance_to(prey.position) <= 3 * TILE):
        reveal_disguise(enemy, maze)
    if enemy.state == "spider":
        enemy.knows_player = prey is maze.player and prey.active and maze.clear_path(enemy.position, prey.position)
        enemy.alert.update(enemy.knows_player, dt, tracked=prey is not None)
        if enemy.knows_player:
            enemy.alert.become_hostile()
        if enemy.attack is not None:
            fit_mimic(enemy, maze)
            advance_attack(enemy, maze)
        elif enemy.dodge_time > 0 or (enemy.knows_player
                and enemy.try_learned_dodge(maze, maze.player.position - enemy.position)):
            enemy.move(STOP)
        elif prey is not None:
            if not begin_attack(enemy, maze, prey):
                feed_mimic(enemy, maze)
                if enemy.distance_to(prey.position) > enemy.radius + prey.radius + 5:
                    travel_mimic(enemy, maze, prey.position, dt)
                else:
                    enemy.move(STOP)
        else:
            enemy.move(STOP)
            feed_mimic(enemy, maze, forage=True)
        enemy.facing = enemy.aim
        enemy.hidden_for = 0.0 if prey is not None or visible else enemy.hidden_for + dt
        if enemy.attack is None and enemy.hidden_for >= CHASE_LOSS and not visible:
            enemy.departed = True
            enemy.exit_direction = (maze.player.position - enemy.position).normalized()
            begin_relocation(enemy, maze)
        return

    # The small collider lets an unseen Mimic slip through normal doors while it changes rooms.
    room = maze.layout.room_at(tile_at(enemy.position))
    inside = room is not None and room.contains(tile_at(maze.player.position))
    if enemy.wall_cells and any(cell not in maze.walls for cell in enemy.wall_cells):
        reveal_disguise(enemy, maze)
        return
    if enemy.state == "relocating" and enemy.destination is not None:
        if inside or visible or enemy.position.get_distance(center(enemy.destination)) <= 7:
            assume_disguise(enemy, maze, inside)
        elif not feed_mimic(enemy, maze, forage=True):
            enemy.follow_route(maze, enemy.destination, dt)
            enemy.facing = enemy.aim
        return

    enemy.move(STOP)
    if enemy.player_was_inside and not inside:
        enemy.departed = True
        enemy.exit_direction = (maze.player.position - enemy.position).normalized()
        enemy.hidden_for = 0.0
    enemy.player_was_inside = inside
    ready = maze.wielding.time >= enemy.disguise_until
    enemy.hidden_for = enemy.hidden_for + dt if not visible and ready else 0.0
    if enemy.hidden_for >= TRANSFORM_DELAY:
        if inside:
            reveal_disguise(enemy, maze)
        elif (not feed_mimic(enemy, maze, forage=not enemy.wall_cells)
              and (enemy.departed or maze.wielding.time >= enemy.next_roam)):
            begin_relocation(enemy, maze)


def start_scribe(scribe: Enemy) -> None:
    scribe.observations = PlayerHabits()
    scribe.tactical_notes = []
    scribe.escort: Enemy | None = None
    scribe.writing = 0.0
    scribe.uploaded = False
    scribe.call_again_at = 0.0


def scribe_sees_player(scribe: Enemy, maze: "Maze") -> bool:
    return (
        scribe.active and not scribe.uploaded and scribe.escort is not None
        and scribe.escort.active and scribe.escort.alert.hostile
        and scribe.position.get_distance(scribe.escort.position) <= 2.7 * TILE
        and scribe.can_see_player(maze)
    )


def update_scribe(scribe: Enemy, maze: "Maze", dt: float) -> bool:
    """Return true once, when the scribe closes the completed book."""
    from .enemy_policy import settle

    actor = scribe
    actor.tick_timers(dt)
    actor.update_stamina(dt)
    if scribe.defend_against_mimic(maze, dt):
        scribe.facing = scribe.aim
        return False
    scribe.sense(maze, dt)
    settle(maze, scribe)
    if scribe.alert.visible and maze.wielding.time >= scribe.call_again_at:
        maze.call_armored_guards(actor)
        scribe.call_again_at = maze.wielding.time + 2.8
    if scribe.escort is None or not scribe.escort.active:
        scribe.escort = min(
            (guard for guard in maze.guards
             if guard.active and not maze.layout.gate_room.contains(tile_at(guard.position))),
            key=lambda guard: (guard.kind not in ("armored", "guard"),
                               actor.position.get_distance(guard.position)),
            default=None,
        )
    if scribe.escort is None:
        if scribe.alert.hostile:
            scribe.update(maze, dt)
        elif not scribe.investigate_eye_tip(maze, dt):
            actor.move(STOP)
        actor.facing = actor.aim
        return False

    escort = scribe.escort
    separation = actor.radius + escort.radius + ANKLE_GAP
    rear = escort.position - escort.facing * separation
    if maze.current_tile(tile_at(rear)) == "#":
        side = escort.facing.rotated(1.5) * separation
        candidates = [escort.position + side, escort.position - side]
        rear = next((point for point in candidates if maze.current_tile(tile_at(point)) != "#"),
                    escort.position)
    forward = rear - actor.position
    if maze.clear_path(actor.position, rear, actor.radius):
        actor.move(forward if forward.length > 7 else STOP)
        actor.aim = (escort.position - actor.position).normalized()
        if not actor.aim.length:
            actor.aim = escort.facing
    else:
        scribe.follow_route(maze, tile_at(rear), dt)
    actor.facing = actor.aim
    if scribe_sees_player(scribe, maze):
        scribe.writing = min(WRITE_SECONDS, scribe.writing + dt)
        if scribe.writing >= WRITE_SECONDS and any(scribe.observations.count(kind) for kind in ("move", "dodge", "aim")):
            scribe.uploaded = True
            return True
    return False
