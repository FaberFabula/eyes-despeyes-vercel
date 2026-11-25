from collections import Counter, deque

from .physics import Vec2d

from .item import EPIC_ITEMS, WEAPONS

OBSERVATION_PRIORS = {"move": 6, "dodge": 3, "aim": 4}
RECENT_SAMPLES = 24


def lateral_axis(toward_player: Vec2d) -> Vec2d:
    """Perpendicular to the guard-to-player line in screen coordinates."""
    return Vec2d(-toward_player.y, toward_player.x)


class PlayerHabits:
    """Learn movement, aim lanes, and repeated weapon choices from recent observations."""

    def __init__(self):
        self.counts = Counter()
        self.recent = {kind: deque(maxlen=RECENT_SAMPLES) for kind in OBSERVATION_PRIORS}

    def record(self, kind: str, direction: Vec2d, side: Vec2d, style: str | None = None) -> None:
        lateral = direction.normalized().dot(side)
        bucket = 1 if lateral > 0.30 else -1 if lateral < -0.30 else 0
        self.recent[kind].append((bucket, direction, style))
        self.counts[kind] += 1

    def merge(self, observations: "PlayerHabits") -> None:
        self.counts.update(observations.counts)
        for kind, samples in observations.recent.items():
            self.recent[kind].extend(samples)

    def usual_movement(self) -> Vec2d:
        samples = self.recent["move"]
        return sum((direction.normalized() for _, direction, _ in samples), Vec2d(0, 0)) / (
            OBSERVATION_PRIORS["move"] + len(samples)
        )

    def predicted_velocity(self) -> Vec2d:
        samples = self.recent["move"]
        return sum((direction for _, direction, _ in samples), Vec2d(0, 0)) / (
            OBSERVATION_PRIORS["move"] + len(samples)
        )

    def count(self, kind: str) -> int:
        return self.counts[kind]

    def bias(self, kind: str) -> float:
        samples = self.recent[kind]
        return sum(bucket for bucket, _, _ in samples) / (OBSERVATION_PRIORS[kind] + len(samples))

    def weapon_read(self, style: str | None) -> float:
        if style is None:
            return 0.0
        samples = self.recent["aim"]
        return sum(sample_style == style for _, _, sample_style in samples) / (3 + len(samples))

    def combat_read(self) -> float:
        samples = sum(len(group) for group in self.recent.values())
        confidence = samples / (6 + samples)
        return min(1.0, 0.30 * confidence + 0.40 * self.evasion_read()
                   + 0.33 * self.usual_movement().length + 0.65 * abs(self.bias("dodge")))

    def evasion_read(self) -> float:
        """Estimate how consistently the player targets one attack lane."""
        counts = Counter(bucket for bucket, _, _ in self.recent["aim"])
        samples = counts.total()
        peak = max(counts.values(), default=0)
        return (3 * peak - samples) / (2 * (OBSERVATION_PRIORS["aim"] + samples))

    def enemy_upgrade_weights(self) -> dict[str, float]:
        """Estimate which enemy stats would help against the tactics we've seen."""
        aims = Counter(style for _, _, style in self.recent["aim"])
        aim_samples = aims.total() + OBSERVATION_PRIORS["aim"]
        weapons = (*WEAPONS, *EPIC_ITEMS)
        melee = sum(aims[weapon.effect] for weapon in weapons if weapon.mode == "melee") / aim_samples
        ranged = sum(aims[weapon.effect] for weapon in weapons if weapon.mode == "ranged") / aim_samples
        magic = sum(aims[weapon.effect] for weapon in weapons if weapon.mode == "magic") / aim_samples
        heavy = sum(aims[style] for style in ("hammer", "pulse", "gravity", "catcher")) / aim_samples
        shield = aims["shield"] / aim_samples
        moves = self.recent["move"]
        fast_movement = sum(velocity.length > 220 for _, velocity, _ in moves) / (
            len(moves) + OBSERVATION_PRIORS["move"]
        )
        dodges = len(self.recent["dodge"])
        evasive = dodges / (dodges + OBSERVATION_PRIORS["dodge"])
        return {
            "Speed": 1 + 3 * (ranged + magic) + 3 * evasive + 2 * fast_movement,
            "Grip": 1 + 2 * melee,
            "Health": 1 + 7 * heavy + ranged + magic,
            "Strength": 1 + 3 * melee + 6 * shield,
            "Stamina": 0.75 + 2 * (fast_movement + evasive + melee),
            "Dexterity": 0.55 + 3 * evasive + 2 * melee,
            "Intelligence": 0.55 + 3 * shield + 2 * magic,
        }
