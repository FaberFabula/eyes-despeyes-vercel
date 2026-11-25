"""Shared action values for the Eye's tactical decisions."""

from collections.abc import Callable
from dataclasses import dataclass
from math import log1p, sqrt
from random import Random

Context = tuple[str, ...]


@dataclass
class Estimate:
    mean: float = 0.0
    count: int = 0

    def learn(self, reward: float, rate_floor: float = 0.16) -> None:
        self.count += 1
        rate = max(rate_floor, 1 / self.count)
        self.mean += rate * (reward - self.mean)


class TacticalModel:
    """Share evidence across mobs, with a separate correction for each context."""

    def __init__(self, learning_rate: float = 0.16,
                 on_learn: Callable[[str, Context, str, float], None] | None = None):
        self.learning_rate = learning_rate
        self.on_learn = on_learn
        self.shared: dict[tuple[str, str], Estimate] = {}
        self.local: dict[tuple[str, Context, str], Estimate] = {}
        self.observations = 0

    def estimate(self, task: str, context: Context, action: str) -> tuple[float, float]:
        shared = self.shared.get((task, action), Estimate())
        local = self.local.get((task, context, action), Estimate())
        weight = local.count / (local.count + 4)
        pooled = shared.mean * shared.count / (shared.count + 3)
        return (1 - weight) * pooled + weight * local.mean, local.count + 0.25 * shared.count

    def choose(self, task: str, context: Context, actions: tuple[str, ...], rng: Random,
               *, confidence: float = 1.0, priors: dict[str, float] | None = None) -> str:
        estimates = {action: self.estimate(task, context, action) for action in actions}
        total = sum(count for _, count in estimates.values())
        exploration = 0.18 * sqrt(log1p(total + 1))
        priors = priors or {}
        return max(actions, key=lambda action: confidence * estimates[action][0]
                   + exploration / sqrt(1 + estimates[action][1])
                   + priors.get(action, 0) + rng.uniform(-0.025, 0.025))

    def learn(self, task: str, context: Context, action: str, reward: float) -> None:
        reward = max(-1.0, min(1.0, reward))
        self.shared.setdefault((task, action), Estimate()).learn(reward, self.learning_rate)
        self.local.setdefault((task, context, action), Estimate()).learn(reward, self.learning_rate)
        self.observations += 1
        if self.on_learn is not None:
            self.on_learn(task, context, action, reward)

    def score(self, task: str, action: str) -> float:
        entry = self.shared.get((task, action), Estimate())
        return entry.mean * entry.count / (entry.count + 4)

    def pressure(self, task: str, *actions: str) -> float:
        choices = [self.score(group, action) for group, action in self.shared
                   if group == task and (not actions or action.partition(":")[0] in actions)]
        return max(0.0, -min(choices, default=0.0))

    def upgrade_weights(self) -> dict[str, float]:
        """Weight stats that address weaknesses or support successful tactics."""
        gear_pressure = self.pressure("combat", "take", "search_weapon", "contest")
        movement_pressure = self.pressure("combat", "close", "left", "right", "back", "pursue")
        return {
            "Speed": 4 * self.pressure("patrol") + 3 * movement_pressure,
            "Grip": 4 * gear_pressure + 3 * max(0, self.score("combat", "contest")),
            "Health": 6 * self.pressure("dodge"),
            "Strength": 3 * max(0, self.score("combat", "close")) + 2 * gear_pressure,
            "Stamina": 2 * (self.pressure("dodge", "left", "right") + self.pressure("combat", "contest")),
            "Dexterity": 4 * self.pressure("aim_ranged"),
            "Intelligence": 4 * self.pressure("aim_magic"),
        }
