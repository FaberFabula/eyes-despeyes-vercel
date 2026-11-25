from math import ceil, cos, inf, pi, radians
from random import Random


from .projectiles import move_projectiles
from .merchant import arm_merchant, provoke_merchant, spawn_merchant
from . import enemy_policy, item_effects
from .physics import Physics, Shape, SweepHit, Vec2d
from .aggro import Noise, SEARCH_TIME, sound_distances
from .carving import WALL_HP, swing_outline, wall_contacts
from .item_actions import ItemActions
from .effects import DEATH_ANIMATION_SECONDS, Beam, Burn, Cue, Death, Fire, GravityWell, Hit, WeaponFlash
from .enemy import Enemy
from .physics import ACTOR, SOLID
from .item import (
    ALL_LAUNCHER,
    BLADE,
    BOW,
    CLAW_ATTACK,
    DEBUG_TOOL,
    EPIC_ITEMS,
    GATE_KEY,
    HEAL_PACK,
    LASER,
    LOOT,
    MAN_CATCHER,
    MIRROR,
    PULSE_GUN,
    RADAR,
    SCOUT_BITE,
    SCOUT_SCRATCH,
    SHIELD,
    SLINGSHOT,
    SPEAR,
    TORCH,
    WEAPONS,
    AttackProfile,
    Equipment,
)
from .entity import DOOR_HP, Chest, Door, XPOrb, solid_tile
from .actor import Actor
from .merchant import Merchant
from .layout import (
    COLS, LIT_TILES, ROWS, TILE, Cell, Room, center, find_path, generate_level, generate_sandbox,
    tile_at, walk_distance,
)
from .memory import lateral_axis
from .enemy_behaviors import (
    hit_disguise, is_disguised, reveal_disguise, scribe_sees_player, start_disguise,
    start_scribe, update_disguise, update_scribe,
)
from .mimic_body import (
    MIMIC_BASE_RADIUS, MimicCorpse, edible_gear, feed_on_actor, set_mimic_size, split_mimic,
    swallow_gear, swallow_mob, update_mimic_corpses,
)
from .player import PLAYER_DAMAGE_SCALE, create_player, prepare_movement
from .progression import (
    BOSS_XP, EYE_QUOTES, FLOOR_CLEAR_XP, FLOOR_HEAL_FRACTION, MERCHANT_PRICES, MIMIC_XP,
    MimicMemory, Progression, Victory,
)
from .item_debug import SandboxTools
from .watcher import damage_eye, eye_population_cap, place_eyes, update_eyes
from .item_physics import (
    CATCHER_COLLISION,
    COLLISION_ATTACK,
    GRAB_RANGE,
    GRIP_CUT,
    GUARD_SWING_TIME,
    HAND_CONE,
    HAND_REACH,
    MIN_HAND_REACH,
    SHIELD_RESISTANCE_PERCENT,
    LAUNCHER_COLLISION,
    LAUNCHER_IMPACT,
    Projectile,
    Wielding,
)
from .physics import GEAR_CATEGORY, SPIKE_CATEGORY
from .item import Gear

USE_DISTANCE = 49
OBSERVATION_INTERVAL = 0.25
SWING_HALF_ANGLE = radians(38)
VISION_HALF_ANGLE = HAND_CONE
VISION_DISTANCE = 6 * TILE
TORCH_RADIUS = 114
FIRE_ATTACK = AttackProfile(0, 1, 0, "heat")
BURN_DURATION = 3.0
BURN_INTERVAL = 0.75
COLLISION_DAMAGE_CAPS = {COLLISION_ATTACK: 2, CATCHER_COLLISION: 8, LAUNCHER_COLLISION: 12}
WEAPON_EFFECT_LABELS = {
    "Bow spike": BOW.name, "Bow lance": BOW.name, "Sword dash": BLADE.name,
    "Ricochet laser": LASER.name, "Shield rush": SHIELD.name, LAUNCHER_IMPACT: ALL_LAUNCHER.name,
}
ENEMY_STATS = {
    "claw": (115, 10, 10, True, CLAW_ATTACK),
    "scout": (151, 9, 8, True, SCOUT_SCRATCH),
    "armored": (74, 30, 16, False, None),
    "guard": (99, 19, 13, True, None),
    "elite": (119, 27, 14, False, None),
}
MOB_STEP_GAP = {
    "scout": 54, "claw": 56, "guard": 61, "armored": 58, "elite": 63,
    "scribe": 66, "mimic": 28,
}
MOB_SOUND_RANGE = {"alert": 9, "attack": 6, "step": 4, "report": 7}
ATTACKER_NAMES = {
    "scout": "Eye Scout", "claw": "Claw Eye", "guard": "Room Eye Guard", "armored": "Armored Eye",
    "elite": "Eye Elite", "scribe": "Eye Scribe", "mimic": "Mimic", "merchant": "Merchant",
}
EYE_LOCKDOWN_SECONDS = 5 * 60
QUICK_ESCAPE_SECONDS = 3 * 60
EYE_TIP_RADIUS = 8 * TILE
EYE_TIP_INTERVAL = 0.8
EYE_TIP_LINGER = 3.0
MIRROR_ESCAPE_SECONDS = 30.0
MIRROR_DECAY_PER_SECOND = 0.25
BOSS_MIRROR_RADIUS = 9 * TILE
BOSS_HIT_TRACK_SECONDS = 4.0
BREACH_SECONDS = 3.6
SUMMON_SECONDS = 4.2
GATE_BREACH_INTERVAL = 0.49
RADAR_BLINK_INTERVAL = 0.62
RADAR_BLINK_WIDTH = 0.28
RADAR_BLINKS = 3
EPIC_CHEST_CHANCE = 0.075
CHEST_REWARD_INTERVAL = 0.27
CHEST_GUARD_RADIUS = 3 * TILE
SANDBOX_XP = 20


class Maze:
    def __init__(self, seed: int | None = None, progress: Progression | None = None,
                 *, sandbox: bool = False):
        self.sandbox = sandbox
        self.debug = SandboxTools(self) if sandbox else None
        self.progress = progress if progress is not None else Progression(xp=SANDBOX_XP if sandbox else 0)
        self.physics = Physics()
        self.actors_by_shape: dict[Shape, Actor] = {}
        self.replacements: list[tuple[Equipment, Room]] = []
        self.next_resupply = 0.0
        self.discovered: dict[Cell, str] = {}
        self.wielding = Wielding(self.physics, self.actors_by_shape)
        self.item_actions = ItemActions(self)
        self.fires: dict[Cell, Fire] = {}
        self.burns: dict[Actor, Burn] = {}
        self.gravity_wells: list[GravityWell] = []
        self.hits: list[Hit] = []
        self.xp_drops: list[XPOrb] = []
        self.mimic_corpses: list[MimicCorpse] = []
        self.swallowed_gear: set[Gear] = set()
        self.flame_trails: list[tuple[Vec2d, Vec2d, float]] = []
        self.next_impact_cue = 0.0
        self.escaped = False
        self.victory: Victory | None = None
        self.death: Death | None = None
        self.escape_time = 0.0
        self.gate_exposed_at: float | None = None
        self.eye_anger_announced = False
        self.eye_tip: Cell | None = None
        self.eye_tip_origin = Vec2d(0, 0)
        self.eye_tip_time = -inf
        self.level_eyes = (False, False, False)
        self.bonus_eyes = (False, False)
        self.shop: Merchant | None = None
        self.cues: list[Cue] = []
        self.projectiles: list[Projectile] = []
        self.beams: list[Beam] = []
        self.weapon_flashes: list[WeaponFlash] = []
        self.radar_started = -10.0
        self.radar_blink = -1
        self.radar_path: list[Cell] = []
        self.radar_target: Vec2d | None = None
        self.habits = self.progress.habits
        self.next_aim_observation = 0.0
        self.scribes: list[Enemy] = []
        self.eye_spawn_charge = 0.0
        self.eye_spawn_retry = 0.0
        self.rng = Random(seed)
        self.layout = (generate_sandbox(len(LOOT)) if sandbox else
                       generate_level(self.rng, self.progress.level, mimic_minimum=len(self.progress.mimics)))
        self.player_cell = self.previous_player_cell = self.layout.player
        self.doors = {}
        self.walls = {}
        self.wall_damage: dict[Cell, int] = {}
        self.carved_walls: set[Cell] = set()
        self.wall_revision = 0
        self.guards = []
        self.gate_fight_started = False
        self.noises: list[Noise] = []
        self.step_distance = 0.0
        self.last_contact: Cell | None = None
        self.unseen_seconds = 0.0
        self.next_bump_noise = 0.0
        self.observe_in = OBSERVATION_INTERVAL

        self.build_board()
        starting_room = self.layout.room_at(self.layout.player)
        self.populate_loot(starting_room)
        self.populate_enemies()
        self.player = create_player(self)
        self.add_actor(self.player)
        if sandbox:
            tool = self.wielding.add(DEBUG_TOOL, self.player.position + Vec2d(36, 0))
            self.wielding.grab(self.player, tool, tool.position)
        self.place_loadout(starting_room)

    def build_board(self) -> None:
        self.physics.add_bounds(COLS * TILE, ROWS * TILE)

        for y, row in enumerate(self.layout.tiles):
            for x, symbol in enumerate(row):
                cell = (x, y)
                if symbol in "#X":
                    self.walls[cell] = solid_tile(self.physics, cell)
                elif symbol in "DJ":
                    jamb = self.layout.jambs[cell]
                    wide = symbol == "J"
                    door = Door(cell, solid_tile(self.physics, cell, jamb if wide else None),
                                locked=wide, jamb=jamb, wide=wide)
                    self.doors[cell] = door
                    if wide:
                        self.doors[jamb] = door

        self.exit_gate = self.doors[self.layout.gate]
        self.eyes = place_eyes(self)
        self.eyes_by_shape = {eye.shape: eye for eye in self.eyes}
        self.watching_eye = None
        self.actors: list[Actor] = []
        self.wall_cells = {shape: cell for cell, shape in self.walls.items()}
        self.doors_by_shape = {door.shape: door for door in self.doors.values()}

    def populate_loot(self, starting_room: Room) -> None:
        if self.sandbox:
            self.chests = [Chest(cell, featured is not None, featured)
                           for cell, featured in zip(self.layout.treasure, (None, *EPIC_ITEMS))]
        else:
            self.chests = [Chest(cell, not starting_room.contains(cell) and self.rng.random() < EPIC_CHEST_CHANCE)
                           for cell in self.layout.treasure]
        self.corpses = () if self.sandbox else (*self.layout.loot, *self.layout.gate_supply)
        for cell in self.layout.torches:
            self.wielding.add(TORCH, center(cell))
        if self.sandbox:
            for spec, cell in zip(LOOT, self.layout.loot):
                self.place_gear(spec, cell)
        else:
            self.wielding.add(self.rng.choice((BLADE, SLINGSHOT, PULSE_GUN)), center(self.layout.loot[0]))
            guaranteed = {}
            candidates = list(self.layout.loot[1:])
            self.rng.shuffle(candidates)
            for spec in (SPEAR, MAN_CATCHER, PULSE_GUN):
                cell = next(cell for cell in candidates if self.gear_angle(spec, cell) is not None)
                guaranteed[cell] = spec
                candidates.remove(cell)
            for index, cell in enumerate(self.layout.loot[1:]):
                pool = WEAPONS if index % 3 == 0 else LOOT
                spec = guaranteed.get(cell) or self.rng.choice(pool)
                if self.place_gear(spec, cell) is None:
                    self.wielding.add(SLINGSHOT, center(cell))
            for cell in self.layout.gate_supply:
                self.place_gear(self.rng.choice((BLADE, SLINGSHOT, PULSE_GUN)), cell)
        self.key = self.wielding.add(GATE_KEY, center(self.layout.key))
        self.mirror = self.wielding.add(MIRROR, center(self.layout.exit))

    def populate_enemies(self) -> None:
        for kind, patrol in self.layout.guards:
            if kind == "scribe":
                self.add_scribe(patrol[0], patrol)
                continue
            self.spawn_guard(kind, patrol[0], patrol)

        if not self.sandbox:
            self.spawn_bosses(*self.layout.bosses)

        self.targets = []
        for cell in self.layout.targets:
            target = Actor(self.physics, cell, speed=0, hp=80, organic=False, radius=13)
            self.targets.append(target)
            self.add_actor(target)

        self.merchants = []
        self.merchants_by_actor = {}
        for cell in self.layout.merchants:
            spawn_merchant(self, cell)

        self.mimics: list[Enemy] = []
        while len(self.progress.mimics) < len(self.layout.mimics):
            self.progress.mimics.append(MimicMemory(MIMIC_BASE_RADIUS))
        for cell, memory in zip(self.layout.mimics, self.progress.mimics):
            self.spawn_mimic(cell, memory=memory)
        if not self.sandbox:
            self.post_chest_guards()

    def place_loadout(self, starting_room: Room | None) -> None:
        if starting_room is not None:
            cells = sorted(starting_room.cells(), key=lambda cell: walk_distance(cell, self.layout.player))
            for spec in self.progress.loadout:
                for cell in cells:
                    if self.spawn_clear(cell, gap=max(24, spec.length * 0.67)) and self.place_gear(spec, cell):
                        break
        self.progress.loadout.clear()
        self.gear_budget = len(self.wielding.items) + 2

    def spawn_bosses(self, elite_cell: Cell, scribe_cell: Cell) -> None:
        self.spawn_guard("elite", elite_cell, boss=True)
        self.spawn_guard("scribe", scribe_cell, boss=True)

    def spawn_guard(self, kind: str, cell: Cell, patrol: tuple[Cell, ...] | None = None,
                    *, boss: bool = False, can_be_hostile: bool = True) -> Enemy:
        stats = {"elite": (107, 41, 21, False, None), "scribe": (88, 46, 22, False, None)} if boss else ENEMY_STATS
        speed, hp, radius, organic, innate = stats[kind]
        eye_form = kind if kind in ("elite", "scribe") else None
        actor = Enemy(self.physics, cell, speed, hp, innate, organic, radius, kind,
                      patrol=patrol or (cell,), eye_form=eye_form, boss=boss, can_be_hostile=can_be_hostile)
        if kind == "armored":
            actor.strength_multiplier = 1.3
            actor.grip_multiplier = 1.25
        if boss:
            actor.strength_multiplier = 1.45 if kind == "elite" else 1.4
            actor.grip_multiplier = 1.6 if kind == "elite" else 1.0
        self.add_guard(actor)
        if boss and kind == "elite":
            weapons = [spec for spec in WEAPONS if spec is not BLADE and spec.length < 50]
            actor.aim = actor.facing = (self.exit_gate.position - actor.position).normalized()
            item = self.wielding.add(self.rng.choice(weapons), actor.position + actor.aim * 39, actor.aim.angle)
            self.wielding.grab(actor, item, item.point_on_axis(-0.65))
        return actor

    def spawn_mimic(self, cell: Cell, *, can_be_hostile: bool = True, memory: MimicMemory | None = None) -> Enemy:
        memory = memory if memory is not None else MimicMemory(MIMIC_BASE_RADIUS)
        actor = Enemy(self.physics, cell, speed=175, hp=40, radius=memory.radius,
                      kind="mimic", eye_form="mimic", can_be_hostile=can_be_hostile)
        actor.mimic_memory = memory
        actor.duel_habits = memory.habits
        self.add_actor(actor)
        start_disguise(actor, self)
        set_mimic_size(actor, self, memory.radius, refill=True)
        self.mimics.append(actor)
        return actor

    def gear_angle(self, spec: Equipment, cell: Cell) -> float | None:
        position = center(cell)
        if max(spec.length, spec.width) < TILE - 4:
            return 0.0
        half_length = spec.length / 2 + 2
        for angle in (0.0, pi / 2):
            axis = Vec2d(half_length, 0).rotated(angle)
            if self.clear_path(position - axis, position + axis, spec.width / 2 + 2):
                return angle
        return None

    def place_gear(self, spec: Equipment, cell: Cell) -> Gear | None:
        angle = self.gear_angle(spec, cell)
        if angle is None:
            return None
        return self.wielding.add(spec, center(cell), angle)

    def post_chest_guards(self) -> None:
        speed, hp, radius, organic, attack = ENEMY_STATS["elite"]
        for chest in self.chests:
            if not chest.epic:
                continue
            room = self.layout.room_at(chest.cell)
            # Reuse a nearby regular Elite if one is already posted in this room.
            if any(not guard.boss and guard.eye_form == "elite"
                   and room.contains(tile_at(guard.position))
                   and chest.position.get_distance(guard.position) <= CHEST_GUARD_RADIUS
                   and self.clear_path(chest.position, guard.position, radius)
                   for guard in self.guards):
                continue
            cells = [cell for cell in room.cells() if cell not in (chest.cell, self.layout.player)
                     and self.clear_path(chest.position, center(cell), radius)]
            cells.sort(key=lambda cell: walk_distance(cell, chest.cell))
            spot = next((cell for cell in cells if self.spawn_clear(cell, gap=2 * radius)), cells[0])
            actor = Enemy(self.physics, spot, speed, hp, attack, organic, radius, "elite",
                          patrol=(spot, chest.cell), eye_form="elite")
            self.add_guard(actor)

    def update_chests(self) -> None:
        now = self.wielding.time
        for chest in self.chests:
            if not chest.rewards or now < chest.next_reward:
                continue
            reward = chest.rewards[0]
            angle = self.rng.uniform(-pi, pi)
            velocity = Vec2d(105, 0).rotated(angle)
            if isinstance(reward, int):
                self.xp_drops.append(XPOrb(chest.position, reward, velocity, now + 0.35))
            else:
                x, y = chest.cell
                cells = [(x + dx, y + dy) for dx, dy in
                         ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))]
                item = next((item for cell in cells
                             if self.clear_path(chest.position, center(cell))
                             and (item := self.place_gear(reward, cell)) is not None), None)
                if item is None:
                    chest.next_reward = now + 0.6
                    continue
                item.body.velocity = velocity * 1.45
                item.body.angular_velocity = self.rng.uniform(-8, 8)
            chest.rewards.pop(0)
            chest.next_reward = now + CHEST_REWARD_INTERVAL

    def update_xp_orbs(self, dt: float) -> None:
        player = self.player
        for orb in tuple(self.xp_drops):
            end = orb.position + orb.velocity * dt
            if (hit := self.hit_solid(orb.position, end, 2)) is not None:
                orb.velocity -= hit.normal * (1.45 * orb.velocity.dot(hit.normal))
            else:
                orb.position = end
            orb.velocity *= max(0, 1 - 5 * dt)
            if not player.active or self.wielding.time < orb.collect_at:
                continue
            offset = player.position - orb.position
            if offset.length <= 42 and self.clear_path(orb.position, player.position):
                if offset.length <= 17:
                    self.xp_drops.remove(orb)
                    self.progress.award_xp(orb.amount)
                else:
                    orb.velocity += offset.normalized() * (760 * dt)

    def replace_broken_items(self) -> None:
        for spec, position in self.wielding.broken:
            if spec is HEAL_PACK or spec in EPIC_ITEMS:
                continue
            cell = tile_at(position)
            room = self.layout.room_at(cell)
            if room is None:
                room = min(self.layout.rooms, key=lambda r: walk_distance(cell, r.middle))
            self.replacements.append((spec, room))
        had_breaks = bool(self.wielding.broken)
        self.wielding.broken.clear()
        if not self.replacements or (not had_breaks and self.wielding.time < self.next_resupply):
            return
        self.next_resupply = self.wielding.time + 1.0
        pending = []
        for spec, origin in self.replacements:
            rooms = [room for room in self.layout.rooms if room != origin
                     and (room != self.layout.gate_room or self.exit_gate.is_open)]
            self.rng.shuffle(rooms)
            cells = [
                cell for room in rooms for cell in self.rng.sample(room.cells(), len(room.cells()))
            ]
            for cell in cells:
                if self.spawn_clear(cell, gap=max(26, spec.length * 0.65)) and self.place_gear(spec, cell):
                    break
            else:
                pending.append((spec, origin))
        self.replacements = pending

    def add_actor(self, actor: Actor) -> None:
        self.actors.append(actor)
        self.actors_by_shape[actor.shape] = actor

    def scale_mob(self, actor: Actor) -> None:
        readiness = 1.0 if self.sandbox or self.progress.hard_mode else min(1.0, 0.68 + 0.12 * (self.progress.level - 1))
        actor.apply_upgrades(self.progress.enemy_bonus, readiness)

    def add_mob(self, actor: Actor) -> None:
        self.scale_mob(actor)
        self.add_actor(actor)

    def add_guard(self, guard: Enemy) -> None:
        self.add_mob(guard)
        self.guards.append(guard)

    def can_summon_eye_mob(self) -> bool:
        guards = sum(guard.alive and not guard.boss and guard.kind != "merchant" for guard in self.guards)
        scribes = sum(scribe.alive for scribe in self.scribes)
        return guards + scribes < eye_population_cap(self.progress.level)

    def guard_for(self, actor: Actor | None) -> Enemy | None:
        return actor if isinstance(actor, Enemy) and actor in self.guards else None

    @property
    def lockdown(self) -> bool:
        return not self.sandbox and self.wielding.time >= EYE_LOCKDOWN_SECONDS

    @property
    def eye_angry(self) -> bool:
        return self.lockdown or self.gate_exposed_at is not None

    def announce_eye_anger(self) -> None:
        if self.eye_angry and not self.eye_anger_announced:
            self.eye_anger_announced = True
            self.cues.append(Cue("eye_angry", 0.88, 0.0))

    def update_eye_tip(self) -> None:
        now = self.wielding.time
        if self.sandbox or self.eye_angry or now < self.eye_tip_time + EYE_TIP_INTERVAL or not self.watched:
            return
        cell = tile_at(self.player.position)
        room = self.layout.room_at(cell)
        self.eye_tip = room.middle if room is not None else cell
        self.eye_tip_origin = self.player.position
        self.eye_tip_time = now

    def eye_investigation(self, actor: Actor) -> Cell | None:
        if (self.wielding.time - self.eye_tip_time <= EYE_TIP_LINGER
                and actor.position.get_distance(self.eye_tip_origin) <= EYE_TIP_RADIUS):
            return self.eye_tip
        return None

    def eye_gaze(self, position: Vec2d) -> Vec2d:
        if self.watching_eye is not None and position == self.watching_eye.position:
            return self.player.position
        return position

    def spawn_clear(self, cell: Cell, gap: float = 24) -> bool:
        position = center(cell)
        return (self.current_tile(cell) not in "#DJG"
                and all(position.get_distance(actor.position) > actor.radius + gap / 2
                        for actor in self.actors if actor.active)
                and all(position.get_distance(item.position) > gap for item in self.wielding.items)
                and all(not eye.alive or position.get_distance(eye.position) > gap for eye in self.eyes)
                and all(cell != chest.cell for chest in self.chests))

    @property
    def near_mirror(self) -> bool:
        return (self.mirror.shape in self.wielding.by_shape
                and self.player.position.get_distance(self.mirror.position) <= 46
                and self.clear_path(self.player.position, self.mirror.position))

    def expose_gate_walls(self) -> None:
        if self.sandbox or self.gate_exposed_at is not None:
            return
        self.gate_exposed_at = self.wielding.time
        for cell in self.walls:
            if self.layout.reinforced(cell):
                self.wall_damage[cell] = max(2, self.wall_damage.get(cell, 0))
        self.wall_revision += 1
        self.wake_gate()
        self.make_cue("gate_collapse", self.mirror.position, 14)
        self.announce_eye_anger()

    def channel_mirror(self, dt: float, pressed: bool) -> None:
        if self.escaped or self.sandbox or not self.player.active:
            return
        if not pressed or not self.near_mirror:
            self.escape_time = max(0.0, self.escape_time - MIRROR_DECAY_PER_SECOND * dt)
            return
        self.expose_gate_walls()
        self.escape_time += dt
        if self.escape_time + 1e-8 < MIRROR_ESCAPE_SECONDS:
            return
        self.escaped = True
        if self.progress.hard_mode:
            self.settle_hard_learning()
        self.progress.award_xp(FLOOR_CLEAR_XP)
        earned = self.progress.enemy_xp
        upgrades = self.progress.improve_enemies()
        self.victory = Victory(enemy_upgrades=upgrades, enemy_upgrade_xp=earned - self.progress.enemy_xp,
                               quote=EYE_QUOTES.draw(reports=self.progress.scribe_reports))
        defeated = {guard.eye_form for guard in self.guards if guard.boss and not guard.alive}
        self.level_eyes = ("elite" in defeated, "scribe" in defeated, not self.lockdown)
        self.bonus_eyes = (any(mimic.defeated_form for mimic in self.mimics),
                           self.wielding.time < QUICK_ESCAPE_SECONDS)
        self.progress.score += sum(self.level_eyes) + sum(self.bonus_eyes)

    def settle_hard_learning(self) -> None:
        for actor in self.actors:
            if isinstance(actor, Enemy):
                enemy_policy.settle(self, actor, final=True)

    def buy_loadout(self, equipment: Equipment) -> bool:
        if self.victory is None or self.victory.phase != "results" or len(self.progress.loadout) >= 3:
            return False
        if not self.progress.buy_equipment(equipment):
            return False
        self.progress.loadout.append(equipment)
        return True

    @property
    def won(self) -> bool:
        return self.escaped

    @property
    def finished(self) -> bool:
        return self.won or not self.player.alive

    def advance_presentation(self, dt: float) -> None:
        if self.victory is not None:
            self.victory.elapsed += dt
        if (death := self.death) is None:
            return
        death.elapsed += dt
        if death.elapsed >= DEATH_ANIMATION_SECONDS or not death.velocity.length:
            return
        begin = death.position
        end = begin + death.velocity * dt
        hit = self.hit_solid(begin, end, self.player.radius)
        if hit is None:
            death.position = end
            damping = 7.4 if death.delivery == "projectile" else 3.9
            death.velocity *= max(0, 1 - damping * dt)
        else:
            death.position = begin.interpolate_to(end, max(0, hit.alpha - 0.01))
            death.velocity = Vec2d(0, 0)

    def hit_solid(self, start: Vec2d, end: Vec2d, radius: float = 0):
        return self.physics.first_hit(start, end, radius, SOLID)

    def clear_path(self, start: Vec2d, end: Vec2d, radius: float = 0) -> bool:
        return self.hit_solid(start, end, radius) is None

    def make_noise(self, origin: Vec2d, radius: int, fallback: Vec2d | None = None) -> None:
        if self.current_tile(tile_at(origin)) == "#" and fallback is not None:
            origin = fallback
        distances = sound_distances(self, origin, radius)
        if not distances:
            return
        self.noises.append(Noise(origin, radius))
        for guard in self.guards:
            if not guard.active or (guard.boss and not self.gate_fight_started):
                continue
            distance = distances.get(tile_at(guard.position))
            if distance is not None:
                guard.hear(tile_at(origin), radius - distance)

    def play_nearby_cue(self, name: str, origin: Vec2d, radius: int, gain: float = 1.0) -> None:
        offset = origin - self.player.position
        distance = offset.length / TILE
        if distance > radius:
            return
        if not self.clear_path(origin, self.player.position):
            around_wall = sound_distances(self, origin, radius).get(tile_at(self.player.position))
            if around_wall is None:
                return
            distance = max(distance, around_wall)
            gain *= 0.53
        volume = gain * (1 - distance / (radius + 1)) ** 1.4
        pan = max(-0.8, min(0.8, offset.x / (4 * TILE)))
        self.cues.append(Cue(name, volume, pan))

    def make_cue(self, kind: str, origin: Vec2d, radius: int) -> None:
        self.play_nearby_cue(kind, origin, radius)
        self.make_noise(origin, radius)

    def mob_sound(self, actor: Actor, action: str, *, cue: str | None = None) -> None:
        if actor.kind not in MOB_STEP_GAP:
            return
        if action == "alert":
            if actor.kind == "mimic" or self.wielding.time < actor.next_alert_sound:
                return
            actor.next_alert_sound = self.wielding.time + 2.0
        gain = 0.48 if action == "step" else 0.92
        prefix = actor.kind
        radius = MOB_SOUND_RANGE[action]
        if isinstance(actor, Enemy) and actor.boss:
            prefix += "_boss"
            if action == "step":
                gain, radius = 0.72, 6
        elif actor.kind == "armored" and action == "step":
            gain = 0.65
        if action == "step" and (mimic := actor if actor in self.mimics else None) is not None:
            cue = "mimic_step_alt" if mimic.alt_step else "mimic_step"
            mimic.alt_step = not mimic.alt_step
            gain = 0.65
        self.play_nearby_cue(cue or f"{prefix}_{action}", actor.position, radius, gain)

    def mob_footsteps(self, dt: float) -> None:
        for actor in self.actors:
            gap = MOB_STEP_GAP.get(actor.kind)
            if gap is None or not actor.active:
                continue
            if (mimic := actor if actor in self.mimics else None) is not None and mimic.state not in ("spider", "fleeing"):
                continue
            speed = min(actor.body.velocity.length, actor.desired_velocity.length)
            if speed < 20:
                continue
            actor.mob_step_distance += speed * dt
            if actor.mob_step_distance >= gap:
                actor.mob_step_distance -= gap
                self.mob_sound(actor, "step")

    def coordinate_search(self, dt: float) -> None:
        engaged = [
            guard for guard in self.guards
            if not guard.boss and guard.active and guard.alert.engaged
        ]
        if not engaged:
            self.last_contact = None
            self.unseen_seconds = 0.0
            return
        active = [guard for guard in engaged if guard.alert.hostile]
        if any(guard.knows_player for guard in active):
            self.last_contact = tile_at(self.player.position)
            self.unseen_seconds = 0.0
        else:
            self.unseen_seconds += dt
        if self.last_contact is None:
            return
        if not active and self.unseen_seconds >= SEARCH_TIME:
            for guard in engaged:
                guard.alert.stand_down()
            self.last_contact = None
            return
        x, y = self.last_contact
        posts = [
            cell for cell in ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
            if self.current_tile(cell) != "#"
        ] or [self.last_contact]
        post_index = 0
        for guard in engaged:
            if guard.alert.hostile:
                continue
            role = enemy_policy.patrol_choice(self, guard, contact=True)
            post = self.last_contact if role == "search" else posts[post_index % len(posts)]
            post_index += role == "guard"
            if guard.alert.role != role:
                guard.alert.assign(role)
                guard.heard = None
                guard.heard_for = 0.0
            if guard.post != post:
                guard.post = post
                guard.search_index = 0

    def current_tile(self, cell: Cell) -> str:
        if (door := self.doors.get(cell)) is not None and door.locked:
            return "#"
        return "." if cell in self.carved_walls else self.layout.tile(cell)

    def reinforced(self, cell: Cell) -> bool:
        return self.gate_exposed_at is None and self.layout.reinforced(cell)

    def can_breach_gate_wall(self, cell: Cell) -> bool:
        x, y = cell
        room = self.layout.gate_room
        return (self.gate_exposed_at is not None and cell in self.walls
                and 0 < x < COLS - 1 and 0 < y < ROWS - 1
                and room.left - 2 <= x <= room.right + 2 and room.top - 2 <= y <= room.bottom + 2)

    def gate_breach_route(self, actor: Actor, goal: Cell) -> list[Cell]:
        def tile(cell: Cell) -> str:
            return "." if self.can_breach_gate_wall(cell) else self.current_tile(cell)

        def cost(cell: Cell) -> float:
            if not self.can_breach_gate_wall(cell):
                return 1.0
            remaining = WALL_HP - self.wall_damage.get(cell, 0)
            damage = self.wall_breach_damage(actor)
            swings = (remaining + damage - 1) // damage
            return 1 + (0.36 + GATE_BREACH_INTERVAL * swings) * actor.speed / TILE

        return find_path(tile_at(actor.position), goal, tile, step_cost=cost)

    def wall_breach_damage(self, actor: Actor) -> int:
        attack = actor.attack_profile
        if attack is None:
            return 2
        return max(2, min(6, attack.damage_against(False, actor.strength_multiplier)))

    def attack_gate_wall(self, guard: Enemy, cell: Cell) -> bool:
        actor = guard
        shape = self.walls[cell]
        offset = center(cell) - actor.position
        if offset.length > actor.radius + TILE / 2 + 10:
            return False
        hit = self.hit_solid(actor.position, center(cell))
        if hit is None or hit.shape is not shape:
            return False
        actor.aim = offset.normalized()
        actor.move(Vec2d(0, 0))
        if self.wielding.time >= guard.next_breach_hit:
            guard.next_breach_hit = self.wielding.time + GATE_BREACH_INTERVAL
            self.mob_sound(actor, "attack")
            self.apply_hit(actor, shape, AttackProfile(0, self.wall_breach_damage(actor), 0, "arcane"),
                           position=hit.point)
        return True

    def route_point(self, actor: Actor, cell: Cell) -> Vec2d:
        point = center(cell)
        extra = actor.radius + 1 - TILE / 2
        if extra <= 0:
            return point
        x, y = cell
        door_here = self.doors.get(cell)

        def solid(neighbor: Cell) -> bool:
            if (door_here is not None and not door_here.locked and not door_here.wide
                    and door_here.jamb == neighbor and not self.reinforced(neighbor)):
                return False
            door = self.doors.get(neighbor)
            return self.current_tile(neighbor) == "#" or (door is not None and not door.is_open)

        return point + Vec2d(
            extra * (int(solid((x - 1, y))) - int(solid((x + 1, y)))),
            extra * (int(solid((x, y - 1))) - int(solid((x, y + 1)))),
        )

    def in_mirror_territory(self, position: Vec2d) -> bool:
        return position.get_dist_sqrd(self.mirror.position) <= BOSS_MIRROR_RADIUS ** 2

    def limit_to_mirror_territory(self, position: Vec2d) -> Vec2d:
        offset = position - self.mirror.position
        if offset.length <= BOSS_MIRROR_RADIUS:
            return position
        return self.mirror.position + offset.normalized() * (BOSS_MIRROR_RADIUS - 0.1)

    def boss_route(self, actor: Actor, goal: Cell, *, stay_near: bool = False,
                   returning: bool = False) -> list[Cell]:
        start = tile_at(actor.position)
        points = {start: actor.position}
        radius = actor.radius + 0.25

        def waypoint(cell: Cell) -> Vec2d:
            if cell not in points:
                points[cell] = self.route_point(actor, cell)
            return points[cell]

        def can_cross(begin: Cell, end: Cell) -> bool:
            source, destination = waypoint(begin), waypoint(end)
            if stay_near and not self.in_mirror_territory(destination):
                return False
            doors = (self.doors.get(begin), self.doors.get(end))
            for hit in self.physics.sweep(source, destination, radius, SOLID):
                shape = hit.shape
                # A boss can clear the door and its breakable jamb when it reaches them.
                if any(door is not None and not door.locked and (
                    shape is door.shape or (
                        door.jamb in self.walls and not self.reinforced(door.jamb)
                        and shape is self.walls[door.jamb]
                    )
                ) for door in doors):
                    continue
                if hit.alpha == 0 and shape.point_query(source).gradient.dot(destination - source) > 0:
                    continue
                return False
            return True

        def distance_to_territory(cell: Cell) -> float:
            # Leave room for the waypoint arrival tolerance so a returning boss crosses the boundary.
            return max(0, waypoint(cell).get_distance(self.mirror.position)
                       - (BOSS_MIRROR_RADIUS - TILE / 2)) / TILE

        return find_path(start, goal, self.current_tile, can_cross=can_cross, closest=True,
                         remaining_distance=distance_to_territory if returning else None)

    def breach_door(self, door: Door, dt: float) -> None:
        if door.broken:
            return
        door.breach += dt
        if door.breach >= BREACH_SECONDS * door.hp / DOOR_HP:
            self.break_door(door)

    def break_door(self, door: Door) -> None:
        if door.broken:
            return
        if not door.is_open:
            self.physics.remove_shape(door.shape)
        door.is_open = door.broken = True
        door.locked = False
        door.hp = 0
        del self.doors_by_shape[door.shape]
        if door.jamb in self.walls:
            self.remove_wall(door.jamb)
        self.wall_revision += 1
        if door is self.exit_gate:
            self.wake_gate()

    def wake_gate(self) -> None:
        if self.gate_fight_started:
            return
        self.gate_fight_started = True
        self.last_contact = tile_at(self.player.position)
        for guard in self.guards:
            if guard.boss and guard.alive:
                self.wake_guard(guard)

    def summon_for(self, guard: Enemy, dt: float) -> None:
        if not guard.alert.visible:
            return
        guard.summon_in -= dt
        if guard.summon_in > 0:
            return
        guard.summon_in = SUMMON_SECONDS
        actor = guard
        x, y = tile_at(actor.position)
        candidates = [
            (tx, ty) for tx in range(x - 2, x + 3) for ty in range(y - 2, y + 3)
            if (cell := (tx, ty)) != self.layout.exit and self.in_mirror_territory(center(cell))
            and self.clear_path(actor.position, center(cell))
            and self.spawn_clear(cell, gap=26)
        ]
        if not candidates:
            return
        cell = self.rng.choice(candidates)
        summoned = sum(ally.summoned and ally.alive for ally in self.guards)
        wants_gear = guard not in self.wielding.held
        spawn_weapon = wants_gear or guard.summon_weapon or summoned >= 2
        if spawn_weapon and len(self.wielding.items) < self.gear_budget:
            if wants_gear:
                candidates.sort(key=lambda square: guard.position.get_dist_sqrd(center(square)))
                cell = candidates[0]
            weapons = [spec for spec in WEAPONS if spec.length < 50]
            if wants_gear:
                weapons = [spec for spec in weapons if spec.attack.damage > 0]
            spec = self.rng.choice(weapons)
            if (item := self.place_gear(spec, cell)) is not None:
                grip = item.point_query(actor.position).point
                if wants_gear and actor.position.get_distance(grip) <= GRAB_RANGE + actor.radius - 10:
                    self.wielding.grab(actor, item, grip)
                self.mob_sound(actor, "attack")
        elif summoned < 2 and (self.sandbox or self.can_summon_eye_mob()):
            self.mob_sound(guard, "attack")
            minion = Enemy(self.physics, cell, speed=103, hp=12, radius=12, kind="guard",
                           patrol=(cell,), summoned=True)
            self.wake_guard(minion)
            self.add_guard(minion)
        guard.summon_weapon = not guard.summon_weapon

    def wake_guard(self, guard: Enemy) -> None:
        if not guard.can_be_hostile:
            return
        if not guard.alert.hostile:
            self.mob_sound(guard, "alert")
        guard.alert.become_hostile()
        guard.last_seen = tile_at(self.player.position)
        guard.last_seen_at = self.wielding.time
        toward = self.player.position - guard.position
        if toward.length:
            guard.aim = guard.facing = toward.normalized()

    def notice_mimic(self, guard: Enemy, mimic: Enemy) -> None:
        if not guard.can_be_hostile:
            return
        if guard.rival is not mimic:
            self.mob_sound(guard, "alert")
        guard.rival = mimic
        guard.rival_until = self.wielding.time + 5.5
        if (merchant := self.merchants_by_actor.get(guard)) is not None and guard not in self.guards:
            provoke_merchant(self, merchant, mimic)

    @property
    def radar_visible(self) -> bool:
        elapsed = self.wielding.time - self.radar_started
        return (0 <= elapsed < RADAR_BLINK_INTERVAL * RADAR_BLINKS
                and elapsed % RADAR_BLINK_INTERVAL < RADAR_BLINK_WIDTH)

    def update_radar(self) -> None:
        if not self.radar_visible:
            return
        blink = int((self.wielding.time - self.radar_started) / RADAR_BLINK_INTERVAL)
        if blink == self.radar_blink:
            return
        self.radar_blink = blink
        self.radar_target = self.exit_gate.position
        start = tile_at(self.player.position)
        target = self.layout.gate
        self.radar_path = [start, *find_path(
            start, target, lambda cell: "." if cell == target else self.current_tile(cell)
        )]

    def start_radar(self) -> bool:
        if self.wielding.time - self.radar_started < RADAR_BLINK_INTERVAL * RADAR_BLINKS:
            return False
        self.radar_started = self.wielding.time
        self.radar_blink = -1
        self.update_radar()
        return True

    def track_with_radar(self, guard: Enemy) -> None:
        guard.radar_until = self.wielding.time + RADAR_BLINK_INTERVAL * RADAR_BLINKS

    @property
    def local_vision_radius(self) -> float:
        return 24.0

    def player_sees(self, position: Vec2d) -> bool:
        offset = position - self.player.position
        distance = offset.length
        within_view = (
            distance <= self.local_vision_radius
            or (distance <= 12 * TILE and self.position_in_light(position))
            or (distance <= VISION_DISTANCE and offset.normalized().dot(self.player.aim) >= cos(VISION_HALF_ANGLE))
        )
        return within_view and self.clear_path(self.player.position, position)

    def add_scribe(self, cell: Cell, patrol: tuple[Cell, ...], *, can_be_hostile: bool = True) -> Enemy:
        actor = Enemy(self.physics, cell, speed=244, hp=8, radius=9, kind="scribe",
                      patrol=patrol, eye_form="scribe", can_be_hostile=can_be_hostile)
        self.add_mob(actor)
        start_scribe(actor)
        self.scribes.append(actor)
        return actor

    def update_scribes(self, dt: float) -> None:
        for scribe in self.scribes[:]:
            actor = scribe
            if actor.active:
                if update_scribe(scribe, self, dt):
                    self.habits.merge(scribe.observations)
                    for note in scribe.tactical_notes:
                        self.progress.policy.learn(*note)
                    self.progress.scribe_reports += 1
                    self.mob_sound(actor, "report")
            elif not actor.alive:
                self.scribes.remove(scribe)
                self.actors.remove(actor)
                del self.actors_by_shape[actor.shape]

    def next_level(self) -> "Maze":
        if not self.escaped:
            return self
        self.progress.elapsed_floors += self.wielding.time
        self.progress.mimics = [mimic.mimic_memory for mimic in self.mimics if mimic.alive]
        self.progress.level += 1
        next_floor = Maze(progress=self.progress)
        next_floor.player.hp = self.player.hp + ceil(next_floor.player.max_hp * FLOOR_HEAL_FRACTION)
        return next_floor

    def call_armored_guards(self, scribe: Actor) -> None:
        self.mob_sound(scribe, "alert")
        position = tile_at(scribe.position)
        distances = sound_distances(self, scribe.position, 15)
        for guard in self.guards:
            actor = guard
            distance = distances.get(tile_at(actor.position))
            if actor.kind in ("armored", "guard") and actor.active and distance is not None:
                guard.hear(position, max(2, 16 - distance))

    def damage_walls(self, actor: Actor, attack: AttackProfile) -> None:
        origin = actor.position
        outline = swing_outline(origin, actor.aim, attack.reach, SWING_HALF_ANGLE)
        struck = []
        for cell, contact in wall_contacts(outline, COLS, ROWS):
            shape = self.walls.get(cell)
            if shape is None and (door := self.doors.get(cell)) is not None and not door.is_open:
                shape = door.shape
            if shape is None:
                continue
            hit = self.hit_solid(origin, contact)
            if hit is not None and hit.shape is shape and shape not in struck:
                struck.append(shape)

        # Resolve every contact against the original walls before removing any.
        for shape in struck:
            self.apply_hit(actor, shape, attack)

    def note_hit(self, shape: Shape, position: Vec2d, damage: int, kind: str,
                 *, cue: str | None = None) -> None:
        if damage <= 0:
            return
        now = self.wielding.time
        self.hits.append(Hit(position, shape, now, kind, damage))
        offset = position - self.player.position
        if offset.length <= 8 * TILE and (cue is not None or now >= self.next_impact_cue):
            pan = max(-0.8, min(0.8, offset.x / (4 * TILE)))
            name = cue or ("hit_flesh" if kind == "flesh" else "hit_solid")
            self.cues.append(Cue(name, 1.0 - 0.65 * offset.length / (8 * TILE), pan))
            self.next_impact_cue = now + 0.07

    def apply_hit(self, source: Actor | None, shape: Shape, attack: AttackProfile,
                  force: float = 1.0, *, position: Vec2d | None = None,
                  shove: Vec2d | None = None, delivery: str = "melee", weapon: str | None = None,
                  weapon_item: Gear | None = None, unavoidable: bool = False) -> int:
        if attack.damage <= 0:
            return 0
        if source is self.player and shape is not source.shape:
            force *= PLAYER_DAMAGE_SCALE
        damage = attack.damage_against(False, force)
        if (eye := self.eyes_by_shape.get(shape)) is not None:
            return damage_eye(self, eye, damage)
        if (spike := self.wielding.spikes.get(shape)) is not None:
            contact = shape.body.position if position is None else position
            self.note_hit(shape, contact, damage, "metal")
            spike.hp -= damage
            if spike.hp <= 0:
                self.wielding.release_spike(spike)
            return damage
        if (item := self.wielding.by_shape.get(shape)) is not None:
            if attack in COLLISION_DAMAGE_CAPS and self.wielding.time < item.physics_safe_until:
                return 0
            if item.durability is not None and not (item.spec is TORCH and attack.kind == "heat"):
                kind = "stone" if item.spec is TORCH else "metal"
                self.note_hit(item.shape, item.position if position is None else position, damage, kind)
                self.wielding.wear(item, damage)
                return damage
            return 0
        if (door := self.doors_by_shape.get(shape)) is not None and not door.is_open:
            self.note_hit(shape, door.position if position is None else position, damage, "stone")
            door.hp -= damage
            if door.hp <= 0:
                self.break_door(door)
            return damage
        if (victim := self.actors_by_shape.get(shape)) is not None:
            if (not victim.active or (victim is source and attack not in COLLISION_DAMAGE_CAPS and not unavoidable)):
                return 0
            if victim.kind == "mimic" and self.wielding.time < victim.split_safe_until:
                return 0
            damage = attack.damage_against(victim.organic, force)
            if attack in COLLISION_DAMAGE_CAPS:
                if self.wielding.time < max(victim.collision_hurt_until, victim.physics_safe_until):
                    return 0
                victim.collision_hurt_until = self.wielding.time + 0.52
                damage = min(COLLISION_DAMAGE_CAPS[attack], damage)
            held = self.wielding.held.get(victim)
            if (not unavoidable and held is not None and held.item.holder is victim
                    and held.item.spec.mode == "shield"):
                # Carry fractional damage between hits so even 1 HP hits receive the resistance.
                damage, victim.shield_damage_remainder = divmod(
                    damage * (100 - SHIELD_RESISTANCE_PERCENT) + victim.shield_damage_remainder, 100
                )
            damage = min(victim.hp, damage)
            if shove is not None:
                self.knock_actor(source, victim, shove, position, delivery=delivery, weapon=weapon,
                                 weapon_item=weapon_item)
            self.note_hit(shape, victim.position if position is None else position, damage,
                          "flesh" if victim.organic else "stone",
                          cue="grip_cut" if attack is GRIP_CUT else None)
            victim.hp -= damage
            merchant = self.merchants_by_actor.get(victim)
            if victim is self.player:
                self.progress.damage_taken += damage
            elif source is self.player:
                self.progress.damage_dealt += damage
            if merchant is not None and victim.can_be_hostile and self.wielding.time >= merchant.next_help:
                merchant.next_help = self.wielding.time + 2.0
                self.make_cue("help", merchant.position, 12)
            if merchant is not None and source is self.player and victim.alive:
                provoke_merchant(self, merchant, source)
            if damage and victim.alive and victim.kind == "mimic":
                reveal_disguise(victim, self)
                if source is not None and source is not victim and source.active and victim.state != "fleeing":
                    victim.prey = source
                    victim.hunt_at = self.wielding.time + 0.8
            if (damage and isinstance(source, Enemy) and source is not victim
                    and (victim is self.player or source.kind == "mimic")):
                source.tactical_damage += damage
                source.seconds_without_hit = 0.0
                if victim is self.player:
                    source.alert.damage += damage
                name = weapon_item.spec.name if weapon_item is not None else WEAPON_EFFECT_LABELS.get(weapon, weapon)
                if name is not None:
                    source.weapon_damage[name] = source.weapon_damage.get(name, 0) + damage
            if damage and isinstance(victim, Enemy) and victim.alive and victim.kind != "mimic" and source is not None and source.kind == "mimic":
                self.notice_mimic(victim, source)
            guard = self.guard_for(victim)
            if guard is not None and guard.boss and damage:
                # Any real hit wakes the chamber, including distant shots and lingering magic.
                self.wake_gate()
                if victim.alive:
                    guard.wounded_until = self.wielding.time + BOSS_HIT_TRACK_SECONDS
                    self.wake_guard(guard)
            elif merchant is None and source is self.player and guard is not None:
                guard.wounded_until = self.wielding.time + 1.1
                if victim.alive and self.clear_path(source.position, victim.position):
                    self.wake_guard(guard)
            if not victim.alive:
                self.defeat_actor(victim, source, attack, position, shove, delivery, weapon, weapon_item)
            return damage

        cell = self.wall_cells.get(shape)
        if cell is None:
            return 0
        if self.reinforced(cell):
            self.note_hit(shape, center(cell) if position is None else position, damage, "metal")
            return 0
        self.note_hit(shape, center(cell) if position is None else position, damage, "stone")
        accumulated = damage + self.wall_damage.get(cell, 0)
        if accumulated < WALL_HP:
            self.wall_damage[cell] = accumulated
        else:
            self.remove_wall(cell)
        return damage

    def drop_defeat_xp(self, origin: Vec2d, xp: int) -> None:
        if self.sandbox:
            return
        for remaining in range(xp, 0, -3):
            amount = min(3, remaining)
            direction = self.rng.uniform(-pi, pi)
            velocity = Vec2d(68 if xp > 3 else 0, 0).rotated(direction)
            self.xp_drops.append(XPOrb(origin, amount, velocity, self.wielding.time))
        self.progress.enemies_defeated += 1

    def defeat_actor(self, victim: Actor, source: Actor | None, attack: AttackProfile,
                     position: Vec2d | None, shove: Vec2d | None, delivery: str, weapon: str | None,
                     weapon_item: Gear | None) -> None:
        merchant = self.merchants_by_actor.get(victim)
        if self.shop is not None and self.shop is merchant:
            self.shop = None
        if isinstance(victim, Enemy):
            enemy_policy.settle(self, victim, final=True)
            victim.end_grip_contest(self)
        held = self.wielding.held.get(victim)
        loose_after = held.item if held is not None and held.item.holder is victim else None
        self.item_actions.end_sword(victim)
        self.wielding.release_victim(victim)
        self.wielding.release(victim)
        self.burns.pop(victim, None)
        self.wielding.momentum_sources.pop(victim.shape, None)

        eater = (source if source is not None and source.kind == "mimic" and source is not victim
                 and source.active and not source.preview_attack else None)
        already_defeated = victim.kind == "mimic" and victim.defeated_form
        if weapon_item is not None and source is not victim and not already_defeated:
            item_effects.credit_kill(self.item_actions, weapon_item)
        if victim is self.player:
            if eater is not None:
                feed_on_actor(eater, self, victim)
                enemy_policy.settle(self, eater, final=True)
            self.record_death(source, attack, position, shove, delivery, weapon)
        elif victim.kind == "mimic":
            if not already_defeated:
                self.drop_defeat_xp(victim.position, MIMIC_XP)
                victim.defeated_form = True
            if split_mimic(victim, self, source):
                return
        elif eater is not None and eater.distance_to(victim.position) <= 6 * TILE:
            if (loose_after is not None and loose_after.holder is None
                    and loose_after.physics_safe_until <= self.wielding.time and edible_gear(self, loose_after)):
                swallow_gear(eater, self, loose_after)
            swallow_mob(eater, self, victim)
            return
        elif merchant is None:
            self.drop_defeat_xp(victim.position, BOSS_XP if isinstance(victim, Enemy) and victim.boss else 2)
        self.physics.remove_actor(victim.shape)

    def record_death(self, source: Actor | None, attack: AttackProfile, position: Vec2d | None,
                     shove: Vec2d | None, delivery: str, weapon: str | None) -> None:
        player = self.player
        attacker = ATTACKER_NAMES.get(source.kind) if source is not None else None
        if source is self.player:
            attacker = "Your own weapon"
        if weapon is None:
            if attack is GRIP_CUT:
                weapon = "Sharp grip"
            elif attack is COLLISION_ATTACK:
                weapon = "Collision"
            else:
                weapon = attacker or "Impact"
        direction = shove if shove is not None else player.external_velocity
        if not direction.length and position is not None:
            direction = player.position - position
        if not direction.length and source is not None:
            direction = player.position - source.position
        direction = direction.normalized() if direction.length else -player.aim
        speed = 0 if delivery == "magic" else 175 if delivery == "projectile" else (
            max(360, min(780, player.external_velocity.length))
        )
        origin = source.position if source is not None and source is not player else None
        self.death = Death(weapon, delivery, attacker, player.position, direction * speed, direction, origin,
                           quote=EYE_QUOTES.draw(defeated=True))

    def knock_actor(self, source: Actor | None, victim: Actor, shove: Vec2d,
                    position: Vec2d | None = None, *, delivery: str = "melee", weapon: str | None = None,
                    weapon_item: Gear | None = None) -> None:
        contact = victim.position if position is None else position
        victim.knock_back(shove, contact - victim.position)
        if source is not None:
            self.wielding.attribute_motion(victim.shape, source, 1.1, delivery=delivery, weapon=weapon,
                                           weapon_item=weapon_item)

    def remove_wall(self, cell: Cell) -> None:
        if cell not in self.walls or self.reinforced(cell):
            return
        shape = self.walls.pop(cell)
        self.physics.remove_shape(shape)
        del self.wall_cells[shape]
        self.wall_damage.pop(cell, None)
        self.carved_walls.add(cell)
        self.wall_revision += 1

    def grab_candidate(self, actor: Actor, cursor: Vec2d) -> tuple[Gear, Vec2d] | None:
        candidates = []
        for item in self.wielding.items:
            if item is self.mirror and actor is self.player and self.near_mirror:
                continue
            contest = actor is self.player and item.spec.attack is not None
            if (not self.wielding.can_grab(actor, item, contest=contest)
                    or item.position.get_distance(cursor) > max(item.spec.length, item.spec.width) / 2 + 5):
                continue
            query = item.point_query(cursor)
            if query.distance > 4:
                continue
            grip = cursor if query.distance <= 0 else query.point
            if grip.get_distance(actor.position) <= GRAB_RANGE and self.clear_path(actor.position, grip):
                candidates.append((query.distance, item, grip))
        if not candidates:
            return None
        _, item, grip = min(candidates, key=lambda candidate: candidate[0])
        return item, grip

    def grab_at(self, actor: Actor, cursor: Vec2d) -> bool:
        if not actor.active or not actor.can_wield or actor in self.wielding.held:
            return False
        if actor is self.player:
            if (mimic := self.disguised_gear_at(cursor)) is not None:
                reveal_disguise(mimic, self)
                return True
        candidate = self.grab_candidate(actor, cursor)
        if candidate is None:
            return False
        item, grip = candidate
        grabbed = self.wielding.grab(actor, item, grip, contest=actor is self.player)
        if grabbed and item is self.mirror and actor is self.player:
            self.expose_gate_walls()
        return grabbed

    def disguised_gear_at(self, cursor: Vec2d) -> Enemy | None:
        return min((mimic for mimic in self.mimics if hit_disguise(mimic, cursor)
                    and self.can_use(self.player, mimic)),
                   key=lambda mimic: cursor.get_dist_sqrd(mimic.position), default=None)

    def player_interact(self, cursor: Vec2d) -> bool:
        if self.grab_at(self.player, cursor):
            return True
        if not self.near_mirror:
            return self.interact(self.player)
        return False

    def available_gear(self, actor: Actor, reach: float = inf) -> list[Gear]:
        origin = actor.position
        return [
            item for item in self.wielding.items
            if item.holder is None and origin.get_dist_sqrd(item.position) <= reach * reach
            and self.clear_path(origin, item.position)
        ]

    def enemies_of(self, actor: Actor, *, disguised: bool = False) -> list[Actor]:
        if actor is not self.player:
            return [self.player] if self.player.active else []
        candidates = [guard for guard in self.guards] + [scribe for scribe in self.scribes] + self.targets
        candidates.extend(mimic for mimic in self.mimics if disguised or not is_disguised(mimic))
        return [enemy for enemy in dict.fromkeys(candidates) if enemy.active]

    def closest_enemy(self, actor: Actor) -> Actor | None:
        return min(self.enemies_of(actor),
                   key=lambda enemy: actor.position.get_dist_sqrd(enemy.position), default=None)

    def ignite(self, position: Vec2d, owner: Actor, power: float, *, weapon: str = "Fire") -> None:
        cell = tile_at(position)
        if self.current_tile(cell) in "#DJ":
            return
        now = self.wielding.time
        if fire := self.fires.get(cell):
            fire.expires = now + 3.4
            if power >= fire.power:
                fire.owner, fire.power, fire.weapon = owner, power, weapon
        else:
            self.fires[cell] = Fire(position, owner, now + 3.4, power=power, weapon=weapon)

    def spread_fire(self, origin: Vec2d, owner: Actor, radius: float, power: float, *, weapon: str = "Fire") -> None:
        self.ignite(origin, owner, power, weapon=weapon)
        x, y = tile_at(origin)
        span = int(radius / TILE) + 1
        for col in range(max(0, x - span), min(COLS, x + span + 1)):
            for row in range(max(0, y - span), min(ROWS, y + span + 1)):
                if (col, row) == (x, y):
                    continue
                point = center((col, row))
                if origin.get_dist_sqrd(point) <= radius * radius and self.clear_path(origin, point):
                    self.ignite(point, owner, power, weapon=weapon)

    def ignite_actor(self, victim: Actor, owner: Actor, power: float, *, weapon: str = "Fire") -> None:
        if not victim.active:
            return
        now = self.wielding.time
        if (burn := self.burns.get(victim)) is not None and burn.expires >= now:
            # New exposure extends the same burn without adding extra damage ticks.
            burn.expires = now + BURN_DURATION
            if power >= burn.power:
                burn.owner, burn.power, burn.weapon = owner, power, weapon
        else:
            self.burns[victim] = Burn(owner, now + BURN_DURATION, now + BURN_INTERVAL, power, weapon)

    def update_effects(self, dt: float) -> None:
        now = self.wielding.time
        self.fires = {cell: fire for cell, fire in self.fires.items() if fire.expires > now}
        self.gravity_wells = [well for well in self.gravity_wells if well.expires > now]
        self.flame_trails = [trail for trail in self.flame_trails if trail[2] > now]
        self.weapon_flashes = [flash for flash in self.weapon_flashes if now - flash.started < 0.34]
        self.hits = [hit for hit in self.hits if now - hit.started < 0.36]
        if self.fires:
            for actor in self.actors:
                if not actor.active:
                    continue
                x, y = tile_at(actor.position)
                for col in range(x - 2, x + 3):
                    for row in range(y - 2, y + 3):
                        if ((fire := self.fires.get((col, row))) is not None
                                and fire.position.get_dist_sqrd(actor.position) <= (17 + actor.radius) ** 2
                                and self.clear_path(fire.position, actor.position)):
                            self.ignite_actor(actor, fire.owner, fire.power, weapon=fire.weapon)
        for actor, burn in tuple(self.burns.items()):
            if not actor.active or now > burn.expires + 1e-6:
                del self.burns[actor]
            elif now + 1e-6 >= burn.next_tick:
                burn.next_tick += BURN_INTERVAL
                source = None if actor is burn.owner else burn.owner
                self.apply_hit(source, actor.shape, FIRE_ATTACK, burn.power, delivery="magic", weapon=burn.weapon)
        if not self.gravity_wells:
            return
        bodies = [*(actor for actor in self.actors if actor.active), *self.wielding.items]
        for well in self.gravity_wells:
            for target in bodies:
                if not self.physics.contains(target.shape):
                    continue
                offset = well.position - target.position
                if not 7 < offset.length < well.radius or not self.clear_path(well.position, target.position):
                    continue
                acceleration = 2480 * well.strength * (1 - 0.50 * offset.length / well.radius) / target.body.mass ** 0.5
                delta = offset.normalized() * acceleration * dt
                self.wielding.attribute_motion(target.shape, well.owner, 0.75,
                                              delivery="magic", weapon=well.weapon)
                if isinstance(target, Actor):
                    target.push(delta)
                else:
                    velocity = target.body.velocity + delta
                    target.body.velocity = velocity.normalized() * min(810, velocity.length)

    def lights(self):
        for fire in self.fires.values():
            yield fire.position, fire.radius, True
        for item in self.wielding.items:
            if not item.light_on:
                continue
            flame = item.body.local_to_world((0, -item.spec.width * 0.33))
            resting = (item.holder is None and item.body.velocity.length_squared < 0.04
                       and abs(item.body.angular_velocity) < 0.05)
            yield flame, TORCH_RADIUS, resting

    def position_in_light(self, position: Vec2d) -> bool:
        return (
            self.layout.tile(tile_at(position)) in LIT_TILES
            or any(position.get_distance(origin) <= radius and self.clear_path(origin, position)
                   for origin, radius, _ in self.lights())
        )

    @property
    def player_in_light(self) -> bool:
        return self.position_in_light(self.player.position)

    @property
    def watched(self) -> bool:
        return self.watching_eye is not None and self.player.active

    def can_use(self, actor: Actor, target: Door | Chest | Merchant | Enemy) -> bool:
        if not actor.active or actor.position.get_distance(target.position) > USE_DISTANCE:
            return False
        hit = self.hit_solid(actor.position, target.position)
        if isinstance(target, Door):
            return not target.broken and (hit is None or hit.shape is target.shape)
        if hit is not None or actor is not self.player:
            return False
        if isinstance(target, Merchant):
            return target.actor.active and target.actor not in self.guards
        if isinstance(target, Enemy) and target.kind == "mimic":
            return is_disguised(target)
        return target in self.chests and not target.opened

    def nearby_target(self, actor: Actor) -> Door | Chest | Merchant | Enemy | None:
        candidates = [*self.chests, *self.doors_by_shape.values()]
        if actor is self.player:
            candidates.extend(self.merchants)
            candidates.extend(mimic for mimic in self.mimics
                              if mimic.form in ("merchant", "chest", "door", "gate"))
        return min(
            (target for target in candidates if self.can_use(actor, target)),
            key=lambda target: actor.position.get_distance(target.position),
            default=None,
        )

    def interact(self, actor: Actor, target: Door | Chest | Merchant | Enemy | None = None) -> bool:
        if target is None:
            target = self.nearby_target(actor)
        if target is None or not self.can_use(actor, target):
            return False
        if isinstance(target, Door):
            if target.locked:
                held = self.wielding.held.get(actor)
                if (held is None or held.item.holder is not actor or held.item.spec is not GATE_KEY
                        or not target.toggle(self.physics, self.actors)):
                    return False
                target.locked = False
                target.opened_at = self.wielding.time
                self.wielding.remove(held.item)
                self.wall_revision += 1
                self.wake_gate()
                return True
            if target.is_open and any(target.shape.bb.intersects(shape.bb)
                                      for item in self.wielding.items for shape in item.shapes):
                return False
            changed = target.toggle(self.physics, self.actors)
            if changed:
                target.opened_at = self.wielding.time
                self.wall_revision += 1
            return changed
        if isinstance(target, Merchant):
            self.shop = target
            return True
        if isinstance(target, Enemy) and target.kind == "mimic":
            reveal_disguise(target, self)
            return True
        actor.treasures += 1
        target.open(self.wielding.time, self.rng)
        return True

    def observe(self, kind: str, direction: Vec2d, style: str | None = None) -> None:
        player = self.player
        now = self.wielding.time
        if not player.active or not direction.length or (kind == "aim" and now < self.next_aim_observation):
            return
        if kind == "aim":
            self.next_aim_observation = now + OBSERVATION_INTERVAL
        watched = self.watched
        witnesses = [scribe for scribe in self.scribes if scribe_sees_player(scribe, self)]
        observers = [*self.guards, *(mimic for mimic in self.mimics if mimic.state == "spider")]
        threats = [guard for guard in observers if guard.active and guard.alert.hostile
                   and guard.alert.visible and player.position.get_distance(guard.position) <= 6 * TILE
                   and self.clear_path(player.position, guard.position)]
        eyes = [guard for guard in threats if guard.duel_habits is not None]
        if not watched and not witnesses and not eyes:
            return
        if kind == "aim" and not threats:
            return
        threat = min(threats, key=lambda guard: player.position.get_dist_sqrd(guard.position), default=None)
        if watched:
            observer = threat.position if threat is not None else self.watching_eye.position
            toward_player = (player.position - observer).normalized()
            self.habits.record(kind, direction, lateral_axis(toward_player), style)
        for scribe in witnesses:
            observer = threat.position if threat is not None else scribe.position
            side = lateral_axis((player.position - observer).normalized())
            scribe.observations.record(kind, direction, side, style)
        for guard in eyes:
            side = lateral_axis((player.position - guard.position).normalized())
            guard.duel_habits.record(kind, direction, side, style)

    def dodge(self, direction: Vec2d) -> None:
        if self.player.start_dodge(direction):
            if self.player.stamina == 0:
                self.wielding.release(self.player)
            self.observe("dodge", self.player.dodge_direction)

    def shield_between(self, start: Vec2d, victim: Actor) -> Gear | None:
        held = self.wielding.held.get(victim)
        if (held is not None and held.item.holder is victim and held.item.spec.mode == "shield"
                and held.item.shape.segment_query(start, victim.position, 2) is not None):
            return held.item
        return None

    def attack(self, actor: Actor, *, charge: float = 1.0, cursor: Vec2d | None = None,
               target: Actor | None = None) -> bool:
        if not actor.active or not actor.can_be_hostile or actor.attack_cooldown > 0:
            return False
        if (held := self.item_actions.held_by(actor)) is not None:
            return self.item_actions.attack_held(actor, held, charge, cursor)
        attack = actor.innate_attack
        foe = target if target is not None else self.player
        if actor.kind == "scout" and actor.distance_to(foe.position) <= SCOUT_BITE.reach:
            attack = SCOUT_BITE
        if attack is None:
            return False
        self.mob_sound(actor, "attack")
        actor.attack_cooldown = attack.recovery
        actor.swing_time = 0.14
        actor.swing_direction = actor.aim
        actor.swing_reach = attack.reach

        if actor is self.player:
            self.observe("aim", actor.aim)
            opponents = [other for other in self.actors if other is not actor]
        else:
            opponents = [foe]

        targets = []
        for opponent in opponents:
            offset = opponent.position - actor.position
            if (
                opponent.active
                and offset.length <= attack.reach
                and offset.normalized().dot(actor.aim) >= cos(SWING_HALF_ANGLE)
                and self.clear_path(actor.position, opponent.position)
            ):
                targets.append(opponent)
        if targets:
            victim = min(targets, key=lambda target: target.position.get_distance(actor.position))
            blocker = self.shield_between(actor.position, victim)
            if blocker is not None:
                self.apply_hit(actor, blocker.shape, attack, actor.strength_multiplier)
            else:
                direction = (victim.position - actor.position).normalized()
                shove = direction * 295
                self.apply_hit(actor, victim.shape, attack, actor.strength_multiplier, shove=shove)
        if attack.damage > 0:
            self.damage_walls(actor, attack)
        return True

    def step(self, dt: float, movement: Vec2d, cursor: Vec2d | None = None,
             sprint: bool = False, interact_held: bool = False) -> None:
        if self.finished or self.shop is not None or self.debug is not None and self.debug.is_open:
            return
        previous_position = self.player.position
        self.begin_tick(dt)
        prepare_movement(self, movement, sprint, dt)
        self.prepare_player_items(dt, cursor)
        self.update_world_effects(dt)
        self.prepare_enemies(dt)
        self.step_physics(dt)
        self.resolve_contacts(previous_position, dt)
        self.update_floor(dt, interact_held)
        self.record_movement(dt)

    def begin_tick(self, dt: float) -> None:
        self.wielding.time += dt
        self.announce_eye_anger()
        self.beams = [beam for beam in self.beams if beam.expires > self.wielding.time]

    def prepare_player_items(self, dt: float, cursor: Vec2d | None) -> None:
        self.update_radar()
        if cursor is not None:
            self.wielding.update_hand(self.player, cursor, dt, player_control=True)
        self.item_actions.update(self.player, dt)

    def update_world_effects(self, dt: float) -> None:
        update_eyes(self, dt)
        self.update_effects(dt)
        if self.debug is not None:
            self.debug.effects.update(dt)
        self.update_eye_tip()
        for noise in self.noises:
            noise.remaining -= dt
        self.noises = [noise for noise in self.noises if noise.remaining > 0]

    def prepare_enemies(self, dt: float) -> None:
        for guard in self.guards:
            if guard.active and self.player.active:
                guard.sense(self, dt)
                enemy_policy.settle(self, guard)
            elif guard.active:
                guard.move(Vec2d(0, 0))
        self.coordinate_search(dt)
        for guard in tuple(self.guards):
            if guard.active:
                self.prepare_guard(guard, dt)
        for mimic in self.mimics:
            if mimic.active and self.player.active:
                enemy_policy.settle(self, mimic)
                update_disguise(mimic, self, dt)

    def prepare_guard(self, guard: Enemy, dt: float) -> None:
        actor = guard
        returning = guard.boss and guard.return_to_mirror(self, dt)
        if not self.player.active and not returning:
            return
        actor.tick_timers(dt)
        actor.update_stamina(dt)
        if (guard.alert.hostile or guard.rival is not None) and (merchant := self.merchants_by_actor.get(actor)) is not None:
            arm_merchant(self, merchant, guard)
        if actor in self.item_actions.dashes:
            self.item_actions.update(actor, dt)
            return
        held = self.item_actions.held_by(actor)
        if not returning and (held is None or held.rush is None):
            if not guard.defend_against_mimic(self, dt):
                guard.update(self, dt)
        elif not returning and guard.knows_player:
            actor.aim = (self.player.position - actor.position).normalized()
        actor.facing = actor.aim
        direction = actor.facing
        reach = HAND_REACH + actor.radius - 10
        held = self.item_actions.held_by(actor)
        if held is not None and held.item.spec.mode in ("ranged", "magic") and (guard.rival or guard.knows_player):
            target = guard.rival if guard.rival is not None else self.player
            muzzle_length = held.item.spec.length / 2 - held.joint.anchor_b.x
            distance = actor.distance_to(target.position)
            reach = max(MIN_HAND_REACH, min(reach, distance - muzzle_length - target.radius - 6))
        if actor.equipment and actor.equipment.mode in ("melee", "shield"):
            winding = guard.windup > 0 or guard.rival_windup > 0
            phase = 1 - actor.swing_time / GUARD_SWING_TIME if actor.swing_time > 0 else 0
            if winding:
                reach *= 0.43
            elif actor.swing_time > 0:
                reach *= min(1.0, 0.43 + 3.6 * phase)
            if actor.equipment.effect in ("spear", "catcher"):
                direction = actor.aim
            elif winding or actor.swing_time > 0:
                direction = actor.facing.rotated(-0.82 + 1.64 * phase)
        self.wielding.update_hand(actor, actor.position + direction * reach, dt)
        self.item_actions.update(actor, dt)
        if guard.boss and not returning:
            next_position = actor.position + actor.desired_velocity * dt
            limit = self.limit_to_mirror_territory(next_position)
            if limit != next_position:
                actor.desired_velocity = (limit - actor.position) / dt
            if actor.restraint is None:
                # Pymunk integrates the prior velocity before applying the next movement command.
                # Limit that voluntary step while retaining the external knockback.
                movement = actor.body.velocity - actor.external_velocity
                next_position = actor.position + movement * dt
                limit = self.limit_to_mirror_territory(next_position)
                if limit != next_position:
                    actor.body.velocity = (limit - actor.position) / dt + actor.external_velocity

    def step_physics(self, dt: float) -> None:
        self.wielding.apply_floor_drag(dt)
        self.wielding.approach_speeds.clear()
        self.physics.step(dt)

    def resolve_contacts(self, previous_position: Vec2d, dt: float) -> None:
        """Apply hits after the collision callbacks finish."""
        if self.wielding.mirror_touched:
            self.expose_gate_walls()
        self.mob_footsteps(dt)
        current_cell = tile_at(self.player.position)
        if current_cell != self.player_cell:
            self.previous_player_cell, self.player_cell = self.player_cell, current_cell
        self.wielding.resolve_grips(dt)
        for actor in self.wielding.jostled:
            self.wielding.release(actor)
            if actor is self.player and self.wielding.time >= self.next_bump_noise:
                self.make_noise(self.player.position, 6)
                self.next_bump_noise = self.wielding.time + 0.4
        self.wielding.jostled.clear()
        if self.player.sprinting or self.player.dodge_time > 0:
            self.step_distance += self.player.position.get_distance(previous_position)
            if self.step_distance >= 32:
                self.step_distance -= 32
                self.make_noise(self.player.position, 4)
        else:
            self.step_distance = 0.0
        for source, origin, radius in self.wielding.contact_noise:
            if source is self.player:
                self.make_noise(origin, radius, source.position)
        self.wielding.contact_noise.clear()
        self.resolve_item_impacts()

    def resolve_item_impacts(self) -> None:
        for impact in self.wielding.impacts:
            self.apply_hit(impact.source, impact.target, impact.attack, impact.force,
                           position=impact.position, shove=impact.shove,
                           delivery=impact.delivery, weapon=impact.weapon, weapon_item=impact.weapon_item)
            if impact.snaring_gear is not None and (victim := self.actors_by_shape.get(impact.target)) is not None:
                if victim.kind != "mimic" or self.wielding.time >= victim.split_safe_until:
                    self.wielding.hook(impact.snaring_gear, victim, impact.position)
            if impact.source is self.player and impact.attack not in (COLLISION_ATTACK, CATCHER_COLLISION, GRIP_CUT):
                held = self.item_actions.held_by(self.player)
                style = held.item.spec.effect if held is not None else None
                self.observe("aim", self.player.aim, style)
        self.wielding.impacts.clear()
        for item, wear in self.wielding.wear_events:
            if self.wielding.time >= item.physics_safe_until:
                self.wielding.wear(item, wear)
        self.wielding.wear_events.clear()
        self.wielding.clear_old_contacts()

    def update_floor(self, dt: float, interact_held: bool) -> None:
        move_projectiles(self, dt)
        item_effects.update_item_effects(self.item_actions, dt)
        self.update_chests()
        self.update_xp_orbs(dt)
        update_mimic_corpses(self)
        self.replace_broken_items()
        if self.player.active:
            self.update_scribes(dt)
            if self.layout.gate_room.contains(tile_at(self.player.position)):
                self.wake_gate()
        self.channel_mirror(dt, interact_held)

    def record_movement(self, dt: float) -> None:
        self.observe_in -= dt
        if self.observe_in <= 0:
            self.observe_in += OBSERVATION_INTERVAL
            if self.player.dodge_time == 0:
                self.observe("move", self.player.desired_velocity)
