"""Health, stamina, movement, and knockback."""

from collections.abc import Callable
from math import atan2, cos, sin

from .entity import Entity
from .item import AttackProfile, Equipment
from .layout import Cell, center
from .physics import Body, Physics, Vec2d


class Actor(Entity):
    def __init__(self, physics: Physics, cell: Cell, speed: float, hp: int,
                 innate_attack: AttackProfile | None = None, organic: bool = True,
                 radius: float = 10, kind: str = "guard", *, can_be_hostile: bool = True):
        self.radius = radius
        self.kind = kind
        self.can_be_hostile = can_be_hostile
        self.can_wield = kind not in ("scout", "claw", "mimic")
        self.restraint: float | None = None
        self.tumble_angle = 0.0
        self.tumble_speed = 0.0
        self.desired_velocity = Vec2d(0, 0)
        self.external_velocity = Vec2d(0, 0)
        self.body, self.shape = physics.add_actor(center(cell), radius, self.apply_velocity)
        self.speed = speed
        self.max_hp = hp
        self.hp = hp
        self.max_stamina = 100.0
        self.stamina = self.max_stamina
        self.stamina_delay = 0.0
        self.sprint_exhausted = False
        self.sprinting = False
        self.mob_step_distance = 0.0
        self.next_alert_sound = 0.0
        self.light_until = 0.0
        self.grip_multiplier = 1.0
        self.strength_multiplier = 1.0
        self.dexterity_multiplier = 1.0
        self.intelligence_multiplier = 1.0
        self.experience = 4
        self.captured = False
        self.organic = organic
        self.equipment: Equipment | None = None
        self.innate_attack = innate_attack
        self.treasures = 0
        self.aim = Vec2d(1, 0)
        self.facing = self.aim
        self.attack_cooldown = 0.0
        self.collision_hurt_until = 0.0
        self.physics_safe_until = 0.0
        self.shield_damage_remainder = 0
        self.swing_time = 0.0
        self.swing_direction = self.aim
        self.swing_reach = 0.0
        self.dodge_cooldown = 0.0
        self.dodge_time = 0.0
        self.dodge_direction = self.aim

    @property
    def hp(self) -> int:
        return self._hp

    @hp.setter
    def hp(self, value: int) -> None:
        self._hp = max(0, min(self.max_hp, value))

    @property
    def max_hp(self) -> int:
        return self._max_hp

    @max_hp.setter
    def max_hp(self, value: int) -> None:
        self._max_hp = max(1, value)
        if hasattr(self, "_hp"):
            self._hp = min(self._hp, self._max_hp)

    @property
    def position(self) -> Vec2d:
        return self.body.position

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def active(self) -> bool:
        return self.alive and not self.captured

    def weapon_power(self, spec: Equipment) -> float:
        if spec.mode == "ranged":
            return self.dexterity_multiplier
        if spec.mode == "magic":
            return self.intelligence_multiplier
        return 1.0

    def apply_upgrades(self, bonus: Callable[[str], float], readiness: float = 1.0) -> None:
        self.speed *= readiness * (1 + bonus("Speed"))
        self.max_hp = round(self.max_hp * readiness * (1 + bonus("Health")))
        self.hp = self.max_hp
        self.max_stamina *= 1 + bonus("Stamina")
        self.stamina = self.max_stamina
        self.grip_multiplier *= 1 + bonus("Grip")
        self.strength_multiplier *= readiness * (1 + bonus("Strength"))
        self.dexterity_multiplier *= 1 + bonus("Dexterity")
        self.intelligence_multiplier *= 1 + bonus("Intelligence")

    @property
    def attack_profile(self) -> AttackProfile | None:
        if self.equipment is not None and self.equipment.attack is not None:
            return self.equipment.attack
        return self.innate_attack

    def tick_timers(self, dt: float) -> None:
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.swing_time = max(0.0, self.swing_time - dt)
        self.dodge_cooldown = max(0.0, self.dodge_cooldown - dt)
        self.dodge_time = max(0.0, self.dodge_time - dt)

    def start_dodge(self, direction: Vec2d) -> bool:
        if not self.active or self.dodge_cooldown > 0 or self.stamina < 18:
            return False
        self.dodge_direction = direction.normalized() if direction.length else self.aim
        self.dodge_time = 0.15
        self.dodge_cooldown = 0.95
        self.spend_stamina(18)
        return True

    def move(self, direction: Vec2d, sprint: bool = False) -> None:
        if not sprint:
            self.sprint_exhausted = False
        self.sprinting = (
            sprint and not self.sprint_exhausted and self.stamina > 0
            and direction.length > 0 and self.dodge_time <= 0
        )
        if self.dodge_time > 0:
            self.desired_velocity = self.dodge_direction * 360
        else:
            self.desired_velocity = direction.normalized() * self.speed * (1.65 if self.sprinting else 1.0)

    def spend_stamina(self, amount: float) -> None:
        self.stamina = max(0.0, self.stamina - amount)
        self.stamina_delay = 0.55

    def update_stamina(self, dt: float) -> None:
        if self.sprinting:
            self.spend_stamina(26 * dt)
            self.sprint_exhausted = self.stamina == 0
            return
        self.stamina_delay = max(0.0, self.stamina_delay - dt)
        if self.stamina_delay == 0:
            self.stamina = min(self.max_stamina, self.stamina + 23 * dt)

    def push(self, velocity: Vec2d) -> None:
        self.external_velocity += velocity
        if self.external_velocity.length > 920:
            self.external_velocity = self.external_velocity.normalized() * 920

    @property
    def inertia(self) -> float:
        return max(0.62, self.body.mass ** 0.42 * (0.78 + 0.011 * self.max_hp)
                   * self.strength_multiplier ** 0.35)

    def knock_back(self, force: Vec2d, offset: Vec2d) -> None:
        velocity = force / self.inertia
        self.push(velocity)
        torque = offset.cross(velocity)
        turn = 1 if torque > 0 else -1 if torque < 0 else (1 if force.x >= 0 else -1)
        self.tumble_speed += turn * min(32, velocity.length / 23)

    def absorb_collision(self, normal: Vec2d) -> None:
        speed = self.external_velocity.dot(normal)
        if speed > 0:
            self.external_velocity -= normal * speed

    def apply_velocity(self, body: Body, _gravity, _damping, dt: float) -> None:
        self.external_velocity *= max(0.0, 1 - 2.5 * dt)
        mobility = self.restraint if self.restraint is not None else 1.0
        velocity = self.desired_velocity * mobility + self.external_velocity
        if self.restraint is not None:
            body.velocity = body.velocity.interpolate_to(velocity, min(1.0, 9.0 * dt))
        else:
            body.velocity = velocity
        self.tumble_angle += self.tumble_speed * dt
        self.tumble_angle = atan2(sin(self.tumble_angle), cos(self.tumble_angle))
        self.tumble_angle *= max(0.0, 1 - 2.2 * dt)
        self.tumble_speed *= max(0.0, 1 - 3.9 * dt)
