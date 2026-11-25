"""Connect the Eye's action values to enemies and observable outcomes."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .layout import TILE, center, tile_at
from .learning import Context
from .memory import lateral_axis

if TYPE_CHECKING:
    from .enemy import Enemy
    from .entity import RoomEye
    from .item import Gear
    from .memory import PlayerHabits
    from .physics import Vec2d
    from .world import Maze

TAKE_WEAPON = "take:"
NO_HIT_COST = 0.20
PRESSURE_AFTER = 3.5
ATTACK_RESULT = "attack_result"


@dataclass
class Trial:
    context: Context
    action: str
    until: float
    hp: int
    damage: int
    sightings: int
    distance: float
    eye: "RoomEye | None"
    scribes: tuple["Enemy", ...]
    reach: float = 0.0
    target_item: "Gear | None" = None
    nourishment: float = 0.0


def facing_context(maze: "Maze", enemy: "Enemy") -> Context:
    own = enemy.equipment.mode if enemy.equipment else ("innate" if enemy.innate_attack else "empty")
    player = enemy.last_player_mode
    if enemy.alert.visible:
        player = maze.player.equipment.mode if maze.player.equipment else "empty"
    location = maze.player.position if enemy.knows_player else center(enemy.last_seen) if enemy.last_seen else None
    near = location is not None and enemy.distance_to(location) <= 3 * TILE
    return own, player, "near" if near else "far"


def eye_witness(maze: "Maze", enemy: "Enemy") -> "RoomEye | None":
    eye = maze.watching_eye
    if eye is not None and eye.alive and eye.room.contains(tile_at(enemy.position)):
        if maze.clear_path(eye.position, enemy.position):
            return eye
    return None


def scribe_witnesses(maze: "Maze", enemy: "Enemy") -> tuple["Enemy", ...]:
    from .enemy_behaviors import scribe_sees_player

    return tuple(scribe for scribe in maze.scribes if scribe.escort is enemy and scribe_sees_player(scribe, maze))


def feedback(maze: "Maze", enemy: "Enemy", task: str, trial: Trial) -> float:
    total = enemy.weapon_damage.get(trial.action, 0) if task == "weapon" else enemy.tactical_damage
    inflicted = total - trial.damage
    received = max(0, trial.hp - enemy.hp)
    danger = received / max(3, enemy.max_hp * 0.4)
    combat = inflicted / 5 - 0.7 * danger
    if task == "patrol":
        acquired = enemy.sightings - trial.sightings
        return 0.42 * min(1, acquired) + combat - 0.08
    if task == "dodge":
        return combat - 0.06 * (trial.action != "hold")
    if task == "mimic_hunt":
        combat += (enemy.nourishment - trial.nourishment) / 3.5
    if task == "combat":
        combat += combat_progress(maze, enemy, trial)
    return combat - NO_HIT_COST * (inflicted == 0)


def combat_progress(maze: "Maze", enemy: "Enemy", trial: Trial) -> float:
    """Credit useful setup even before a hit lands."""
    action = trial.action
    distance = enemy.distance_to(maze.player.position)
    if action.startswith(TAKE_WEAPON) or action in ("search_weapon", "contest", "secure_key"):
        newly_armed = trial.context[0] in ("empty", "utility") and enemy.attack_profile is not None
        discovered = action == "search_weapon" and bool(enemy.seen_weapons)
        denied_key = (action == "secure_key" and maze.key.holder is enemy
                      and distance < 6 * TILE and maze.exit_gate.locked)
        item = trial.target_item
        approach = 0.0
        if item is not None and item.holder in (None, maze.player) and item.shape in maze.wielding.by_shape:
            approach = trial.distance - enemy.distance_to(item.position)
        return 0.68 * newly_armed + 0.28 * discovered + 0.38 * denied_key + 0.14 * max(0, min(1, approach / TILE))
    if action == "attack" or trial.reach <= 0:
        return 0.0
    gain = max(0, trial.distance - trial.reach) - max(0, distance - trial.reach)
    return 0.14 * max(-1, min(1, gain / TILE))


def finish_trial(maze: "Maze", enemy: "Enemy", task: str) -> None:
    trial = enemy.trials.pop(task, None)
    if trial is None:
        return
    task = "combat" if task == ATTACK_RESULT else task
    result = feedback(maze, enemy, task, trial)
    if enemy.kind == "mimic":
        enemy.mimic_memory.policy.learn(task, trial.context, trial.action, result)
        if maze.progress.hard_mode:
            maze.progress.policy.learn(task, trial.context, trial.action, result)
            return
    if trial.eye is not None and eye_witness(maze, enemy) is trial.eye:
        maze.progress.policy.learn(task, trial.context, trial.action, result)
    else:
        for scribe in scribe_witnesses(maze, enemy) if trial.scribes else ():
            if scribe in trial.scribes:
                scribe.tactical_notes.append((task, trial.context, trial.action, result))


def settle(maze: "Maze", enemy: "Enemy", *, final: bool = False) -> None:
    for task, trial in tuple(enemy.trials.items()):
        if final or maze.wielding.time >= trial.until:
            finish_trial(maze, enemy, task)


def start_trial(maze: "Maze", enemy: "Enemy", task: str, context: Context,
                action: str, seconds: float) -> None:
    damage = enemy.weapon_damage.get(action, 0) if task == "weapon" else enemy.tactical_damage
    enemy.trials[task] = Trial(context, action, maze.wielding.time + seconds, enemy.hp,
                               damage, enemy.sightings, enemy.distance_to(maze.player.position),
                               eye_witness(maze, enemy), scribe_witnesses(maze, enemy),
                               nourishment=enemy.nourishment if enemy.kind == "mimic" else 0.0)


def learning_confidence(maze: "Maze", enemy: "Enemy") -> float:
    if enemy.kind == "mimic":
        return enemy.learning_weight
    first_floor = maze.progress.level == 1 and not maze.progress.hard_mode
    return min(1.6, enemy.learning_weight * (0.50 if first_floor else 1.0))


def decide(maze: "Maze", enemy: "Enemy", task: str, actions: tuple[str, ...], seconds: float,
           *, context: Context | None = None, priors: dict[str, float] | None = None) -> str:
    context = facing_context(maze, enemy) if context is None else context
    trial = enemy.trials.get(task)
    if trial is not None and trial.context == context and trial.action in actions:
        return trial.action
    model = maze.progress.policy
    if enemy.kind == "mimic":
        priors = {action: (priors or {}).get(action, 0) + 0.45 * model.estimate(task, context, action)[0]
                  for action in actions}
        model = enemy.mimic_memory.policy
    action = model.choose(task, context, actions, maze.rng,
                          confidence=learning_confidence(maze, enemy), priors=priors)
    start_trial(maze, enemy, task, context, action, seconds)
    return action


def choose_combat(maze: "Maze", enemy: "Enemy", actions: tuple[str, ...],
                  held: "Gear | None", items: dict[str, "Gear"], preferred_move: str, reach: float) -> str:
    """Compare attacks, movement, and equipment in the same context."""
    context = facing_context(maze, enemy)
    model = maze.progress.policy
    current_weapon = held.spec.name if held is not None and held.spec.attack is not None else None
    current = model.estimate("weapon", context[1:], current_weapon)[0] if current_weapon else 0.0
    weight = learning_confidence(maze, enemy)
    priors = {preferred_move: 0.13, "attack": 0.23 + 0.32 * weight * current}
    weapon_to_drop = 0.18 if current_weapon else 0.0
    for action, item in items.items():
        other = model.estimate("weapon", context[1:], item.spec.name)[0]
        distance_cost = min(0.12, 0.02 * enemy.distance_to(item.position) / TILE)
        if action == "secure_key":
            other = 0.0
        priors[action] = 0.85 * weight * (other - current) - weapon_to_drop - distance_cost
    if current_weapon:
        priors["search_weapon"] = -0.12 - 0.22 * weight * current
    if enemy.seconds_without_hit >= PRESSURE_AFTER:
        # While stuck, try a real attack or make progress toward one.
        advancing = ("attack",) if "attack" in actions else ("close", "pursue", "search_weapon")
        progress = tuple(action for action in actions if action in items or action in advancing)
        actions = progress or actions
        priors["attack"] += 0.22
    previous = enemy.trials.get("combat")
    action = decide(maze, enemy, "combat", actions, 1.15, context=context, priors=priors)
    trial = enemy.trials["combat"]
    if trial is not previous:
        trial.reach = reach
        trial.target_item = items.get(action)
        if trial.target_item is not None:
            trial.distance = enemy.distance_to(trial.target_item.position)
    return action


def wait_for_attack(maze: "Maze", enemy: "Enemy", seconds: float) -> None:
    """Keep the attack's result separate from choices made during recovery."""
    trial = enemy.trials.get("combat")
    if trial is not None and trial.action == "attack":
        del enemy.trials["combat"]
        if ATTACK_RESULT not in enemy.trials:
            trial.until = maze.wielding.time + seconds
            enemy.trials[ATTACK_RESULT] = trial


def watch_weapon(maze: "Maze", enemy: "Enemy", item: "Gear", seconds: float) -> None:
    if "weapon" not in enemy.trials:
        start_trial(maze, enemy, "weapon", facing_context(maze, enemy)[1:], item.spec.name, seconds)


def aim_offset(maze: "Maze", enemy: "Enemy", offset: "Vec2d", habits: "PlayerHabits",
               windup: float, mode: str, travel: float = 0.0) -> "Vec2d":
    task = "aim_" + mode
    choices = ["direct"]
    predicted = habits.predicted_velocity()
    bias = habits.bias("dodge")
    if predicted.length > 12:
        choices.append("lead")
    if abs(bias) > 0.08:
        choices.append("counter_dodge")
    action = decide(maze, enemy, task, tuple(choices), max(0.9, windup + travel + 0.5),
                    priors={"direct": 0.02})
    if action == "lead":
        lead = predicted * min(0.60, windup * 0.74 + travel)
    elif action == "counter_dodge":
        lead = lateral_axis(offset.normalized()) * (62 * bias)
    else:
        return offset
    if lead.length > 0.44 * offset.length:
        lead = lead.normalized() * (0.44 * offset.length)
    return offset + lead


def patrol_choice(maze: "Maze", enemy: "Enemy", *, contact: bool = False) -> str:
    context = ("trail" if contact else "quiet", "armed" if enemy.innate_attack else "empty")
    actions = ("search", "guard", "patrol") if contact else ("patrol", "guard")
    return decide(maze, enemy, "patrol", actions, 2.4, context=context,
                  priors={"patrol": 0.13} if not contact else {"search": 0.04, "guard": 0.04})
