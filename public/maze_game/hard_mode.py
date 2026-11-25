"""Turn pooled tactical outcomes into a prior for a fresh Hard Mode run."""

from collections.abc import Callable
from json import JSONDecodeError, dumps, loads

from .learning import Context, Estimate, TacticalModel
from .progression import Progression

Report = Callable[[str, str], None]
Prior = str | Callable[[], str]
MAX_PRIOR_LOCAL = 12
MAX_PRIOR_SHARED = 20


def seed_policy(model: TacticalModel, data: str) -> None:
    try:
        rows = loads(data)
    except (TypeError, JSONDecodeError):
        return
    if not isinstance(rows, list):
        return
    shared_totals: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 5:
            continue
        task, context, action, count, reward = row
        if (not isinstance(task, str) or not isinstance(action, str)
                or not isinstance(context, list) or not all(isinstance(part, str) for part in context)
                or not isinstance(count, int) or count <= 0
                or not isinstance(reward, int) or abs(reward) > 1000 * count):
            continue
        model.local[(task, tuple(context), action)] = Estimate(reward / (1000 * count), min(count, MAX_PRIOR_LOCAL))
        key = task, action
        prior_count, prior_reward = shared_totals.get(key, (0, 0))
        shared_totals[key] = prior_count + count, prior_reward + reward
    for key, (count, reward) in shared_totals.items():
        model.shared[key] = Estimate(reward / (1000 * count), min(count, MAX_PRIOR_SHARED))


def hard_progression(prior_rows: str, report: Report | None) -> Progression:
    def record(task: str, context: Context, action: str, reward: float) -> None:
        report("sample", dumps((task, context, action, round(1000 * reward))))

    model = TacticalModel(on_learn=record if report is not None else None)
    seed_policy(model, prior_rows)
    return Progression(policy=model, hard_mode=True)
