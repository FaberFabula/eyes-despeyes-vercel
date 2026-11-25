"""Enemy movement, awareness, and combat decisions."""

from collections import deque
from dataclasses import dataclass
from math import cos, radians
from typing import TYPE_CHECKING


from . import enemy_policy, item_effects
from .physics import Physics, Vec2d
from .aggro import Alert
from .actor import Actor
from .item import AttackProfile
from .layout import TILE, Cell, center, find_path, tile_at, walk_distance
from .memory import PlayerHabits, lateral_axis
from .item_physics import GRAB_RANGE, HAND_REACH
from .item import Gear

if TYPE_CHECKING:
    from .world import Maze

SIGHT_DISTANCE = 6 * TILE
SIGHT_DOT = cos(radians(42))
RANGED_SPREAD = radians(14)
MELEE_SPREAD = radians(15)
BOSS_TRAIL_SECONDS = 7.0
ITEM_SWITCH_COOLDOWN = 2.5
STOP = Vec2d(0, 0)


@dataclass
class GripContest:
    deadline: float
    weapon: Gear


class Enemy(Actor):
    def __init__(self, physics: Physics, cell: Cell, speed: float, hp: int,
                 innate_attack: AttackProfile | None = None, organic: bool = True,
                 radius: float = 10, kind: str = "guard", *, patrol: tuple[Cell, ...] = (),
                 eye_form: str | None = None, boss: bool = False, summoned: bool = False,
                 can_be_hostile: bool = True):
        super().__init__(physics, cell, speed, hp, innate_attack, organic, radius, kind,
                         can_be_hostile=can_be_hostile)
        self.patrol = patrol
        self.patrol_index = 0
        self.last_seen = None
        self.last_seen_at = float("-inf")
        self.last_player_mode = "unknown"
        self.route_goal = None
        self.route_wall_revision = -1
        self.route_can_breach = False
        self.route_mirror_position: Vec2d | None = None
        self.route_returning = False
        self.route = deque()
        self.windup = 0.0
        self.attack_target = self.position
        self.attack_item: Gear | None = None
        self.attack_special = False
        self.attack_read = 0.0
        self.align_left = 0.0
        self.wounded_until = 0.0
        self.next_evasion_check = 0.0
        self.alert = Alert(can_be_hostile=self.can_be_hostile)
        self.heard: Cell | None = None
        self.heard_for = 0.0
        self.heard_volume = -1
        self.post: Cell | None = None
        self.search_index = 0
        self.eye_form = eye_form
        self.boss = boss
        self.summoned = summoned
        self.summon_in = 1.25
        self.summon_weapon = True
        self.radar_until = 0.0
        self.knows_player = False
        self.following_eye_tip = False
        self.attempt: GripContest | None = None
        self.next_contest = 0.0
        self.trials: dict[str, enemy_policy.Trial] = {}
        self.tactical_damage = 0
        self.seconds_without_hit = 0.0
        self.sightings = 0
        self.weapon_damage: dict[str, int] = {}
        self.seen_weapons: dict[str, Gear] = {}
        self.next_swap = 0.0
        self.gear_scan_at = 0.0
        self.gear_search_goal: Cell | None = None
        self.key_take_until = 0.0
        self.key_retry_at = 0.0
        self.home_cell = cell
        self.scout_ally: Enemy | None = None
        self.scout_cell: Cell | None = None
        self.scout_until = 0.0
        self.next_report = 0.0
        self.reported_cell: Cell | None = None
        self.report_expires = 0.0
        self.next_breach_hit = 0.0
        self.ambush_key: tuple[Cell, Cell, int] | None = None
        self.ambush_post: Cell | None = None
        self.duel_habits = PlayerHabits() if eye_form in ("elite", "mimic") or boss else None
        self.rival: Actor | None = None
        self.rival_until = 0.0
        self.rival_windup = 0.0

    @property
    def learning_weight(self) -> float:
        if self.boss:
            return 1.65 if self.eye_form == "elite" else 1.36
        if self.eye_form == "elite":
            return 1.40
        if self.eye_form == "mimic":
            return 1.65
        return 1.02 if self.kind == "armored" else 0.72

    def combat_habits(self, maze: "Maze") -> PlayerHabits:
        private = self.duel_habits
        if private is not None and private.combat_read() > maze.habits.combat_read():
            return private
        return maze.habits

    def can_see_player(self, maze: "Maze") -> bool:
        actor = self
        offset = maze.player.position - actor.position
        return (
            actor.can_be_hostile and maze.player.active and offset.length <= SIGHT_DISTANCE
            and (self.boss or maze.wielding.time < self.wounded_until
                 or not offset.length or offset.normalized().dot(actor.facing) >= SIGHT_DOT)
            and maze.clear_path(actor.position, maze.player.position)
        )

    def sense(self, maze: "Maze", dt: float) -> None:
        if self.boss and not maze.sandbox and not maze.gate_fight_started:
            return
        visible = self.can_see_player(maze)
        if visible:
            self.last_player_mode = maze.player.equipment.mode if maze.player.equipment else "empty"
        eye_pursuit = maze.eye_angry and not self.boss
        tracked = eye_pursuit or (self.boss and (
            maze.mirror.holder is maze.player or maze.wielding.time < self.wounded_until
        )) or (
            maze.wielding.time < self.radar_until
        )
        was_hostile = self.alert.hostile
        if not was_hostile and (eye_pursuit or self.boss):
            self.alert.become_hostile()
        if visible and not self.alert.visible:
            self.sightings += 1
        self.alert.update(visible, dt, tracked=tracked or self.boss)
        if self.alert.hostile and not was_hostile:
            maze.mob_sound(self, "alert")
        self.knows_player = visible or tracked
        if self.alert.hostile and self.knows_player:
            self.seconds_without_hit += dt
        elif not self.alert.engaged:
            self.seconds_without_hit = 0.0
        if self.knows_player:
            self.last_seen = tile_at(maze.player.position)
            self.last_seen_at = maze.wielding.time
            if visible and self.alert.hostile and self.position.get_distance(maze.player.position) <= 3 * TILE:
                self.alert.pressure_seconds += dt
        self.heard_for = max(0.0, self.heard_for - dt)
        if self.heard_for == 0:
            self.heard = None

    def hear(self, cell: Cell, volume: int) -> None:
        if self.alert.hostile or self.alert.role == "guard":
            return
        if self.heard_for > 0 and volume < self.heard_volume:
            return
        self.alert.hear()
        self.heard = cell
        self.heard_volume = volume
        self.heard_for = 3.0

    def investigate(self, maze: "Maze", goal: Cell, dt: float) -> None:
        if self.position.get_distance(center(goal)) <= 7:
            self.move(STOP)
            self.aim = self.aim.rotated(dt * 1.5)
        else:
            self.follow_route(maze, goal, dt)

    def investigate_eye_tip(self, maze: "Maze", dt: float) -> bool:
        goal = maze.eye_investigation(self)
        if goal is None:
            return False
        self.following_eye_tip = True
        self.investigate(maze, goal, dt)
        return True

    def search(self, maze: "Maze", dt: float) -> None:
        anchor = self.post
        if self.heard is not None:
            self.investigate(maze, self.heard, dt)
            return
        if anchor is None:
            return
        x, y = anchor
        offsets = ((0, 0), (2, 0), (0, 2), (-2, 0), (0, -2), (3, 1), (-3, -1))
        cells = [(x+dx, y+dy) for dx, dy in offsets if maze.current_tile((x+dx, y+dy)) != "#"]
        if not cells:
            return
        goal = cells[self.search_index % len(cells)]
        current = tile_at(self.position)
        if self.position.get_distance(center(goal)) < 7 or (
            self.route_goal == goal and not self.route and current != goal
        ):
            self.search_index += 1
            goal = cells[self.search_index % len(cells)]
        self.follow_route(maze, goal, dt)

    def return_to_mirror(self, maze: "Maze", dt: float) -> bool:
        actor = self
        if maze.in_mirror_territory(actor.position):
            return False
        # Keep the physical knockback. Drop the attack plan and walk back using a clear route.
        self.cancel_windup(maze)
        self.end_grip_contest(maze)
        maze.item_actions.end_sword(actor)
        if (held := maze.item_actions.held_by(actor)) is not None:
            maze.item_actions.end_shield_rush(actor, held, coast=False)
        actor.dodge_time = actor.swing_time = 0
        self.follow_route(maze, tile_at(maze.mirror.position), dt, returning=True)
        return True

    def find_ambush_post(self, maze: "Maze", last_seen: Cell) -> Cell | None:
        """Find a reachable tile behind cover near the player's learned travel lane."""
        velocity = self.combat_habits(maze).predicted_velocity()
        if velocity.length < 20:
            return None
        anchor = center(last_seen)
        predicted = anchor + velocity.normalized() * min(2.5 * TILE, 0.9 * velocity.length)
        x, y = tile_at(predicted)
        reserved = {guard.ambush_post for guard in maze.guards
                    if guard is not self and guard.boss and guard.active}
        actor = self
        candidates = []
        for tx in range(x - 2, x + 3):
            for ty in range(y - 2, y + 3):
                cell = (tx, ty)
                point = maze.route_point(actor, cell)
                if (cell in reserved or maze.current_tile(cell) == "#" or not maze.in_mirror_territory(point)
                        or (point - anchor).dot(velocity) <= 0 or maze.clear_path(anchor, point)
                        or maze.physics.near_solid(point, actor.radius + 0.25)):
                    continue
                score = predicted.get_dist_sqrd(point) + 0.18 * actor.position.get_dist_sqrd(point)
                candidates.append((score, cell))

        for _, cell in sorted(candidates)[:4]:
            if cell == tile_at(actor.position):
                return cell
            route = maze.boss_route(actor, cell, stay_near=True)
            if route and route[-1] == cell:
                return cell
        return None

    def search_mirror_territory(self, maze: "Maze", dt: float) -> None:
        goal = home = tile_at(maze.mirror.position)
        last = self.last_seen
        if last is not None and (self.knows_player or (
                maze.wielding.time < self.last_seen_at + BOSS_TRAIL_SECONDS
                and maze.in_mirror_territory(center(last)))):
            goal = last
            if not self.knows_player:
                key = (last, home, maze.wall_revision)
                if key != self.ambush_key:
                    self.ambush_key = key
                    self.ambush_post = self.find_ambush_post(maze, last)
                if self.ambush_post is not None:
                    goal = self.ambush_post
        else:
            self.ambush_key = self.ambush_post = None
        self.follow_route(maze, goal, dt)

    def defend_against_mimic(self, maze: "Maze", dt: float) -> bool:
        rival = self.rival
        if rival is None:
            return False
        if (not rival.active or rival.state == "fleeing" or maze.wielding.time >= self.rival_until
                or (self.boss and not maze.in_mirror_territory(rival.position))):
            self.rival = None
            self.rival_windup = 0.0
            return False
        self.cancel_windup(maze)
        offset = rival.position - self.position
        self.aim = offset.normalized() if offset.length else self.facing
        if self.attack_profile is None:
            if self.can_wield and (self.eye_form != "scribe" or self.boss):
                weapons = (item for item in self.gear_options(maze, 4 * TILE)
                           if item.spec.attack is not None and maze.wielding.can_grab(self, item, replace_held=True))
                item = min(weapons, key=lambda gear: self.distance_to(gear.position), default=None)
                if item is not None:
                    self.approach_item(maze, item, dt)
                    return True
            self.move(-offset, sprint=True)
            return True
        held = maze.item_actions.held_by(self)
        item = held.item if held is not None else None
        clear = maze.clear_path(self.position, rival.position)
        if not clear or offset.length > self.fighting_reach(item, 0.5):
            self.rival_windup = 0.0
            if clear and maze.clear_path(self.position, rival.position, self.radius):
                self.move(offset)
            else:
                self.follow_route(maze, tile_at(rival.position), dt)
            return True
        self.move(STOP)
        if self.attack_cooldown > 0:
            return True
        if self.rival_windup == 0:
            self.rival_windup = 0.25 if self.boss else 0.38
            if item is not None and item.spec.effect == "gravity":
                maze.item_actions.special(self)
        else:
            self.rival_windup = max(0.0, self.rival_windup - dt)
            if self.rival_windup == 0:
                maze.attack(self, charge=0.85, cursor=rival.position, target=rival)
        return True

    def update(self, maze: "Maze", dt: float) -> None:
        actor = self
        player = maze.player
        offset = player.position - actor.position
        visible = self.alert.visible
        self.following_eye_tip = False
        if self.boss and not maze.sandbox and not maze.gate_fight_started:
            actor.move(STOP)
            return
        if self.boss and self.eye_form == "scribe" and visible:
            maze.summon_for(self, dt)
        if actor.dodge_time > 0 or (self.alert.hostile and visible and self.try_learned_dodge(maze, offset)):
            self.cancel_windup(maze)
            actor.move(STOP)
            return

        if self.attempt is not None and (self.attempt.weapon.holder is actor or not self.alert.hostile):
            self.end_grip_contest(maze)
        if not self.alert.hostile:
            maze.wielding.release(actor)
        elif (held := maze.wielding.held.get(actor)) is not None and held.item.contender is maze.player:
            self.windup = 0.0
            actor.aim = (held.grip - actor.position).normalized()
            actor.move(actor.position - player.position)
            return

        if self.attempt is not None and self.continue_grip_contest(maze, dt):
            return

        if actor.kind == "scout" and self.relay_scout_report(maze, dt):
            return

        if self.windup > 0 and self.continue_windup(maze, dt):
            return

        if not self.alert.hostile:
            if self.reported_cell is not None and maze.wielding.time < self.report_expires:
                self.investigate(maze, self.reported_cell, dt)
            elif visible:
                actor.aim = offset.normalized() if offset.length else actor.aim
                actor.move(STOP)
            elif self.investigate_eye_tip(maze, dt):
                return
            elif self.alert.role == "guard" and self.post is not None:
                self.investigate(maze, self.post, dt)
            elif self.alert.role == "search":
                if self.knows_player and self.last_seen is not None:
                    self.follow_route(maze, self.last_seen, dt)
                else:
                    self.search(maze, dt)
            elif self.knows_player and self.last_seen is not None:
                self.follow_route(maze, self.last_seen, dt)
            elif self.heard is not None:
                self.investigate(maze, self.heard, dt)
            else:
                role = self.alert.role or enemy_policy.patrol_choice(maze, self)
                if role == "guard":
                    self.investigate(maze, self.home_cell, dt)
                else:
                    self.follow_route(maze, self.idle_goal(maze), dt)
            return

        self.update_combat(maze, dt, offset, visible)

    def update_combat(self, maze: "Maze", dt: float, offset: Vec2d, visible: bool) -> None:
        item = self.combat_hand(maze)
        if item is maze.key and self.carry_key(maze, dt):
            return
        if item is maze.key:
            item = None
        habits = self.combat_habits(maze)
        floor_read = 0.42 if maze.progress.level == 1 and not maze.sandbox and not maze.progress.hard_mode else 1.0
        read = min(1.0, habits.combat_read() * self.learning_weight * floor_read)
        if item is not None and (snare := maze.wielding.snares.get(item)) is not None and snare.victim is maze.player:
            if item.spec.effect == "catcher" and read > 0.22:
                maze.item_actions.special(self)
            elif item.spec.effect == "spear" and maze.wielding.time >= snare.started + 0.85:
                maze.item_actions.special(self)
            self.aim = offset.normalized()
            self.move(-self.aim + lateral_axis(self.aim) * 0.8)
            return

        clear = visible or (self.knows_player and not (self.boss or maze.eye_angry)
                            and maze.clear_path(self.position, maze.player.position))
        direct = clear and maze.clear_path(self.position, maze.player.position, self.radius + 1)
        reach = self.fighting_reach(item, read)
        moves = self.combat_directions(maze, offset, reach) if direct else {}
        options = list(moves) or ["pursue"]
        attack_ready = item is None or item.spec.effect != "gravity" or maze.item_actions.can_special(item)
        if (clear and self.attack_profile is not None and self.attack_cooldown == 0
                and self.swing_time == 0 and offset.length <= reach and attack_ready):
            options.append("attack")

        items = self.combat_items(maze, item)
        options.extend(items)
        if self.gear_search_goal is not None and not self.seen_weapons and self.can_wield:
            options.append("search_weapon")
        ranged = self.equipment is not None and self.equipment.mode in ("ranged", "magic")
        preferred = "back" if ranged and offset.length < min(92, 0.72 * reach) else "close"
        if not moves:
            preferred = "pursue"
        action = enemy_policy.choose_combat(maze, self, tuple(options), item, items, preferred, reach)
        if action == "attack":
            self.prepare_attack(maze, item, offset, read, habits)
        elif action in moves:
            self.route_goal = None
            self.aim = offset.normalized()
            self.move(moves[action])
        elif action == "contest":
            deadline = maze.wielding.time + 2.1
            self.trials["combat"].until = deadline
            self.attempt = GripContest(deadline, items[action])
            self.continue_grip_contest(maze, dt)
        elif action in items:
            target = items[action]
            if not self.approach_item(maze, target, dt):
                self.pursue_player(maze, dt, visible)
            elif target.holder is self:
                if target is maze.key:
                    self.key_take_until = maze.wielding.time + 3.2
                enemy_policy.finish_trial(maze, self, "combat")
        elif action == "search_weapon":
            self.follow_route(maze, self.gear_search_goal, dt)
            if not self.route:
                self.gear_search_goal = None
        else:
            self.pursue_player(maze, dt, visible)

    def pursue_player(self, maze: "Maze", dt: float, visible: bool) -> None:
        if not visible and self.boss:
            self.search_mirror_territory(maze, dt)
        elif not visible and maze.eye_angry:
            goal = tile_at(maze.player.position)
            room = maze.layout.gate_room
            allow_breach = (maze.gate_exposed_at is not None
                            and room.contains(goal) != room.contains(tile_at(self.position)))
            self.follow_route(maze, goal, dt, allow_breach=allow_breach)
        elif self.reported_cell is not None and maze.wielding.time < self.report_expires and not self.knows_player:
            self.investigate(maze, self.reported_cell, dt)
        elif self.last_seen is not None:
            self.follow_route(maze, self.last_seen, dt)
        else:
            self.move(STOP)

    def combat_hand(self, maze: "Maze") -> Gear | None:
        held = maze.wielding.held.get(self)
        if held is None:
            return None
        if held.item.spec.effect != "radar":
            return held.item
        if self.knows_player or maze.attack(self):
            maze.wielding.release(self)
            return None
        return held.item

    def gear_options(self, maze: "Maze", reach: float):
        actor = self
        for item in maze.available_gear(actor, reach):
            if self.boss and not maze.in_mirror_territory(item.position):
                continue
            offset = item.position - actor.position
            if not self.boss and offset.length > GRAB_RANGE and offset.normalized().dot(actor.facing) < SIGHT_DOT:
                continue
            yield item

    def combat_directions(self, maze: "Maze", offset: Vec2d, reach: float) -> dict[str, Vec2d]:
        toward = offset.normalized()
        side = lateral_axis(toward)
        close = toward if offset.length > self.radius + maze.player.radius + 5 else STOP
        choices = {"close": close, "left": side + close * 0.18,
                   "right": -side + close * 0.18}
        retreat_limit = min(SIGHT_DISTANCE - TILE, max(2 * TILE, reach + TILE))
        if offset.length < retreat_limit:
            choices["back"] = -toward
        return {action: direction for action, direction in choices.items()
                if direction.length and maze.clear_path(self.position, self.position + direction.normalized() * 24,
                                                        self.radius)}

    def scan_weapons(self, maze: "Maze") -> None:
        time = maze.wielding.time
        if time < self.gear_scan_at:
            return
        self.gear_scan_at = time + 0.27
        seen: dict[str, Gear] = {}
        for item in self.gear_options(maze, SIGHT_DISTANCE):
            radar = item.spec.effect == "radar" and self.equipment is None and not self.knows_player
            if item.spec is self.equipment or item.spec.attack is None and not radar:
                continue
            closest = seen.get(item.spec.name)
            if closest is None or self.distance_to(item.position) < self.distance_to(closest.position):
                seen[item.spec.name] = item
        self.seen_weapons = seen
        if seen:
            if (trial := self.trials.get("combat")) is not None and trial.action == "search_weapon":
                enemy_policy.finish_trial(maze, self, "combat")
            self.gear_search_goal = None
        elif self.gear_search_goal is None or self.distance_to(center(self.gear_search_goal)) < 22:
            self.gear_search_goal = self.weapon_search_room(maze)

    def combat_items(self, maze: "Maze", held: Gear | None) -> dict[str, Gear]:
        if not self.can_wield or (self.eye_form == "scribe" and not self.boss) or self.stamina <= 0:
            return {}
        self.scan_weapons(maze)
        now = maze.wielding.time
        if held is not None and (now < self.next_swap or self.swing_time > 0):
            return {}
        choices = {
            enemy_policy.TAKE_WEAPON + name: item for name, item in self.seen_weapons.items()
            if (held is None or (item.spec.attack is not None and item.spec is not held.spec))
            and maze.wielding.can_grab(self, item, replace_held=True)
        }
        grip = maze.item_actions.held_by(maze.player)
        if (now >= self.next_contest and grip is not None and grip.item.spec.attack is not None
                and self.alert.visible and self.distance_to(grip.item.position) <= 3 * TILE
                and maze.wielding.can_grab(self, grip.item, contest=True, replace_held=True)):
            choices["contest"] = grip.item
        key = maze.key
        if (not self.boss and now >= self.key_retry_at and maze.exit_gate.locked
                and self.distance_to(key.position) < 4 * TILE and maze.clear_path(self.position, key.position)
                and maze.wielding.can_grab(self, key, replace_held=True)):
            choices["secure_key"] = key
        return choices

    def weapon_search_room(self, maze: "Maze") -> Cell | None:
        """Pick a nearby room to check when no loose weapon is visible."""
        current = maze.layout.room_at(tile_at(self.position))
        candidates = [room.middle for room in maze.layout.rooms
                      if room is not current and (self.boss or room is not maze.layout.gate_room)
                      and self.distance_to(center(room.middle)) < 15 * TILE
                      and (not self.boss or maze.in_mirror_territory(center(room.middle)))]
        return min(candidates, key=lambda cell: self.distance_to(center(cell)), default=None)

    def approach_item(self, maze: "Maze", item: Gear, dt: float) -> bool:
        grip = item.point_on_axis(-0.7)
        reach = GRAB_RANGE + self.radius - 10
        if self.distance_to(grip) > reach or not maze.clear_path(self.position, grip):
            grip = item.point_query(self.position).point
        if self.distance_to(grip) <= reach and maze.clear_path(self.position, grip):
            if maze.wielding.grab(self, item, grip, replace_held=True):
                self.next_swap = maze.wielding.time + ITEM_SWITCH_COOLDOWN
                self.move(STOP)
                return True
        offset = item.position - self.position
        if maze.clear_path(self.position, item.position, self.radius + 1):
            self.aim = offset.normalized()
            self.move(offset)
            return True
        self.follow_route(maze, tile_at(item.position), dt)
        return bool(self.route)

    def carry_key(self, maze: "Maze", dt: float) -> bool:
        time = maze.wielding.time
        if time >= self.key_take_until or not maze.exit_gate.locked:
            maze.wielding.release(self)
            self.key_retry_at = time + 5.0
            return False
        anchor = center(self.last_seen) if self.last_seen else self.position
        goal = max(self.patrol or (self.home_cell,), key=lambda cell: anchor.get_distance(center(cell)))
        self.follow_route(maze, goal, dt)
        return True

    def end_grip_contest(self, maze: "Maze") -> None:
        attempt = self.attempt
        if attempt is None:
            return
        if attempt.weapon.holder is self:
            self.next_swap = maze.wielding.time + ITEM_SWITCH_COOLDOWN
            enemy_policy.finish_trial(maze, self, "combat")
        elif attempt.weapon.contender is self:
            maze.wielding.release(self)
        self.attempt = None
        self.next_contest = maze.wielding.time + 0.8

    def continue_grip_contest(self, maze: "Maze", dt: float) -> bool:
        actor, attempt = self, self.attempt
        item = attempt.weapon
        if item.holder is not maze.player or maze.wielding.time >= attempt.deadline:
            self.end_grip_contest(maze)
            return False
        if item.contender is not None and item.contender is not actor:
            self.end_grip_contest(maze)
            return False
        reach = GRAB_RANGE + actor.radius - 10
        grip = item.point_query(actor.position).point
        if item.contender is actor:
            grip = maze.wielding.held[actor].grip
        elif item.sharp_at(grip):
            handle = item.point_on_axis(-0.7)
            if handle.get_distance(actor.position) <= reach and maze.clear_path(actor.position, handle):
                grip = handle
        toward = grip - actor.position
        if item.contender is actor or (toward.length <= reach and maze.clear_path(actor.position, grip)
                                       and maze.wielding.grab(actor, item, grip, contest=True, replace_held=True)):
            actor.aim = toward.normalized()
            actor.move(actor.position - maze.player.position)
        else:
            self.follow_route(maze, tile_at(item.position), dt)
        return True

    def relay_scout_report(self, maze: "Maze", dt: float) -> bool:
        actor, time = self, maze.wielding.time
        if self.scout_ally is None:
            if not self.alert.hostile or time < self.next_report:
                return False
            if maze.player.attack_profile is None and actor.hp > actor.max_hp / 2:
                return False
            self.next_report = time + 4.0
            candidates = sorted(
                (guard for guard in maze.guards if guard.alive and not guard.boss
                 and guard.kind in ("armored", "guard", "elite")
                 and guard.position.get_distance(actor.position) <= 15 * TILE),
                key=lambda guard: guard.position.get_dist_sqrd(actor.position),
            )
            start = tile_at(actor.position)
            for ally in candidates:
                goal = tile_at(ally.position)
                if start == goal or find_path(start, goal, maze.current_tile):
                    self.scout_ally = ally
                    self.scout_cell = maze.previous_player_cell
                    self.scout_until = time + 7.0
                    break
            else:
                return False

        ally = self.scout_ally
        if not ally.alive or time >= self.scout_until:
            self.scout_ally = None
            return False
        goal = ally.position
        if actor.position.get_distance(goal) <= 38 and maze.clear_path(actor.position, goal):
            ally.reported_cell = self.scout_cell
            ally.report_expires = time + 7.0
            ally.alert.hear()
            self.scout_ally = None
            self.next_report = time + 6.0
            return False
        self.follow_route(maze, tile_at(goal), dt)
        return True

    def fighting_reach(self, item: Gear | None, read: float) -> float:
        attack = self.attack_profile
        if attack is None:
            return 0
        if item is None:
            return attack.reach * (0.88 + 0.08 * read)
        if item.spec.mode == "melee":
            return min(102, attack.reach + 8 + self.radius - 10 + 8 * read)
        if item.spec.mode == "shield":
            return 55
        return attack.reach * (0.63 + 0.16 * read)

    def prepare_attack(self, maze: "Maze", item: Gear | None, offset: Vec2d, read: float,
                       habits: PlayerHabits | None = None) -> None:
        actor = self
        habits = habits or self.combat_habits(maze)
        self.attack_item = item
        self.attack_read = read
        self.attack_special = False
        self.align_left = 0.14
        ranged = item is not None and item.spec.mode in ("ranged", "magic")
        fast = self.boss or self.eye_form == "elite"
        self.windup = (0.49 - 0.28 * read) if fast else (0.63 - 0.31 * read)
        if item is not None and maze.item_actions.can_special(item):
            effect = item.spec.effect
            if effect == "gravity":
                maze.item_actions.special(actor)
                self.windup += 0.25 + 0.13 * read
            elif self.should_use_special(maze, item, offset.length, read):
                self.attack_special = True
                self.windup += 0.13
                if effect == "bow":
                    self.attack_special = maze.item_actions.special(actor)
                    if self.attack_special:
                        self.windup += 0.63
        flight = 0.0
        if ranged and item.spec.effect not in ("beam", "flame") and item.spec.projectile_speed:
            flight = offset.length / item.spec.projectile_speed
        lingering = item is not None and item.spec.effect in ("flame", "gravity", "catcher", "grimoire", "seal")
        result_delay = self.windup + flight + (1.8 if lingering else 0.8)
        enemy_policy.wait_for_attack(maze, self, result_delay)
        if item is not None:
            enemy_policy.watch_weapon(maze, self, item, result_delay)
        mode = item.spec.mode if ranged else "melee"
        direction = enemy_policy.aim_offset(maze, self, offset, habits, self.windup, mode, flight)
        spread = (RANGED_SPREAD if ranged else MELEE_SPREAD) * (1 - 0.82 * read)
        if maze.progress.level == 1 and not maze.sandbox and not maze.progress.hard_mode:
            spread *= 1.3
        direction = direction.rotated(maze.rng.uniform(-spread, spread))
        self.attack_target = self.position + direction
        if self.attack_special and item.spec.effect == "sword":
            maze.item_actions.mark_sword(actor, self.attack_target + direction.normalized() * 24)
        if direction.length:
            actor.aim = direction.normalized()
        actor.move(STOP)

    def should_use_special(self, maze: "Maze", item: Gear, distance: float, read: float) -> bool:
        effect = item.spec.effect
        if effect == "shield":
            return True
        if effect == "catcher":
            return False
        if effect == "grimoire":
            return maze.item_actions.book_streaks.get(item, 0) >= 4
        if effect == "seal":
            return False  # Trained Eyes place a complete seal before closing it.
        if effect == "launcher":
            return self.experience > 0 and distance <= item.spec.attack.reach
        if effect in ("hammer", "sword", "pulse") and distance > 74:
            return False
        if effect == "sling":
            held = maze.item_actions.held_by(self)
            if held is not None and held.lock_target is not None:
                return False
        player_held = maze.item_actions.held_by(maze.player)
        style = player_held.item.spec.effect if player_held is not None else None
        habits = self.combat_habits(maze)
        base = 0.15 if self.boss else 0.07
        odds = base + self.learning_weight * (0.23 * read + 0.24 * habits.weapon_read(style))
        if effect == "sling":
            odds *= 0.38
        return maze.rng.random() < odds

    def try_learned_dodge(self, maze: "Maze", offset: Vec2d) -> bool:
        actor = self
        now = maze.wielding.time
        if (now < self.next_evasion_check or actor.dodge_cooldown > 0 or actor.restraint is not None
                or 0 < self.windup < 0.095):
            return False
        held = maze.item_actions.held_by(maze.player)
        if held is None:
            return False
        item = held.item
        attack = item.spec.attack
        if attack is None:
            return False
        threat_reach = attack.reach
        if item.spec.mode in ("melee", "shield"):
            threat_reach = HAND_REACH + item.spec.length * 0.85
        if offset.length > threat_reach + actor.radius:
            return False
        habits = self.combat_habits(maze)
        familiar = habits.weapon_read(item.spec.effect)
        tip = item.point_on_axis(1)
        toward = (actor.position - tip).normalized()
        if item.spec.mode in ("melee", "shield"):
            afoot = (-offset).normalized().dot(maze.player.aim) > 0.74
            incoming = item.body.velocity_at_world_point(tip).dot(toward)
            threatening = maze.wielding.active_swing(item) and (incoming > 46 or afoot and familiar > 0.35)
            if held.rush is not None:
                threatening = (-offset).normalized().dot(held.rush.direction) > 0.78
            if held.sword_destination is not None:
                lane = held.sword_destination - maze.player.position
                along = (-offset).dot(lane.normalized())
                threatening = threatening or (0 < along < lane.length and abs(offset.cross(lane.normalized())) < 27)
        else:
            winding_up = held.primary_down and (item.spec.draw_time > 0 or item.spec.effect == "flame")
            threatening = (item.axis.dot(toward) > 0.91 or held.lock_target is actor) and (
                winding_up or held.gravity_charging or held.bow_lance_charging or now - item.activated_at <= 0.16
            )
        if not threatening:
            return False
        self.next_evasion_check = now + (0.50 if self.boss else 0.72)
        side = lateral_axis(offset.normalized())
        choices = {"hold": STOP}
        for name, direction in (("left", side), ("right", -side)):
            landing = actor.position + direction * 55
            if self.boss and not maze.in_mirror_territory(landing):
                continue
            if maze.clear_path(actor.position, landing, actor.radius + 1):
                choices[name] = direction
        novice = maze.progress.level == 1 and not maze.sandbox and not maze.progress.hard_mode
        priors = {"hold": 0.22 if novice else 0.045,
                  "left": -0.12 * habits.bias("aim"), "right": 0.12 * habits.bias("aim")}
        action = enemy_policy.decide(maze, self, "dodge", tuple(choices), 0.57, priors=priors)
        return action != "hold" and actor.start_dodge(choices[action])

    def cancel_windup(self, maze: "Maze") -> None:
        if not self.windup:
            return
        self.windup = 0
        if (held := maze.item_actions.held_by(self)) is not None:
            held.charge = 0
            held.gravity_charging = False
            held.bow_lance_charging = False
            held.sword_destination = None

    def continue_windup(self, maze: "Maze", dt: float) -> bool:
        actor = self
        held = maze.item_actions.held_by(actor)
        item = held.item if held is not None else None
        if (not self.alert.hostile or not self.knows_player or actor.attack_profile is None
                or self.attack_item is not item):
            self.cancel_windup(maze)
            return False
        self.windup = max(0.0, self.windup - dt)
        offset = maze.player.position - actor.position
        toward = self.attack_target - actor.position
        if toward.length:
            actor.aim = toward.normalized()
        direction = STOP
        if item is not None and item.spec.mode in ("ranged", "magic"):
            if offset.length < min(70, 0.70 * self.fighting_reach(item, self.attack_read)):
                direction = -offset
            from_item = self.attack_target - item.position
            if from_item.length:
                item.steer(from_item.angle, held.joint.anchor_b, manual=True)
            if item.spec.draw_time and not self.attack_special:
                held.charge = min(0.46 + 0.54 * self.attack_read, held.charge + dt * 2.1)
            if self.windup == 0 and self.align_left > 0 and item.axis.dot(from_item.normalized()) < 0.95:
                self.windup = dt
                self.align_left -= dt
        elif item is not None and offset.length > actor.radius + maze.player.radius + 12:
            if maze.clear_path(actor.position, actor.position + offset.normalized() * 28, actor.radius + 1):
                direction = offset
        actor.move(direction)
        if self.windup == 0:
            charged = held.charge if held is not None else 1.0
            used_special = self.attack_special and (
                maze.item_actions.release_special(actor) if held is not None and held.bow_lance_charging
                else maze.item_actions.special(actor, self.attack_target)
            )
            if not used_special or item is not None and item.spec.effect == "whip":
                if item is not None and item.spec.effect == "seal":
                    item_effects.enemy_seal(maze.item_actions, actor, item, self.attack_target)
                else:
                    maze.attack(actor, charge=charged, cursor=self.attack_target)
            if held is not None and item.spec.draw_time:
                held.charge = 0
        return True

    def idle_goal(self, maze: "Maze") -> Cell:
        cell = tile_at(self.position)
        if not self.patrol:
            return cell
        goal = self.patrol[self.patrol_index]
        if self.position.get_distance(center(goal)) < 7:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol)
            goal = self.patrol[self.patrol_index]
        return goal

    def follow_route(self, maze: "Maze", goal: Cell, dt: float = 0.0, *, allow_breach: bool = False,
                     returning: bool = False) -> None:
        actor = self
        wide = actor.radius > TILE / 2
        current = tile_at(actor.position)
        mirror = maze.mirror.position if self.boss else None
        stay_near = self.boss and not returning
        territory_changed = self.boss and (
            self.route_mirror_position is None or mirror.get_distance(self.route_mirror_position) >= TILE / 4
            or self.route_returning != returning
            or (self.route and walk_distance(current, self.route[0]) > 1)
            or (stay_near and self.route and not maze.in_mirror_territory(maze.route_point(actor, self.route[0])))
        )
        if (goal != self.route_goal or self.route_wall_revision != maze.wall_revision
                or self.route_can_breach != allow_breach or territory_changed):
            if allow_breach and not wide:
                cells = maze.gate_breach_route(actor, goal)
            elif wide:
                cells = maze.boss_route(actor, goal, stay_near=stay_near, returning=returning)
            else:
                cells = find_path(current, goal, maze.current_tile)
            self.route = deque(cells)
            self.route_goal = goal
            self.route_wall_revision = maze.wall_revision
            self.route_can_breach = allow_breach
            self.route_mirror_position = mirror
            self.route_returning = returning

        while self.route and actor.position.get_distance(maze.route_point(actor, self.route[0])) < 6:
            self.route.popleft()
        next_cell = self.route[0] if self.route else current
        if allow_breach and maze.can_breach_gate_wall(next_cell) and maze.attack_gate_wall(self, next_cell):
            return
        if (door := maze.doors.get(next_cell)) is not None:
            if wide and not door.broken and (not door.is_open or not door.wide):
                if actor.position.get_distance(door.position) <= 1.4 * TILE:
                    maze.breach_door(door, dt)
                    actor.move(STOP)
                    return
            elif not door.is_open:
                maze.interact(actor, door)

        offset = maze.route_point(actor, next_cell) - actor.position
        if offset.length > 5:
            actor.aim = offset.normalized()
            actor.move(offset)
        else:
            actor.move(STOP)
