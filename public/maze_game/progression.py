from dataclasses import dataclass, field
from random import Random

from .item import BLADE, EPIC_ITEMS, HEAL_PACK, LASER, LOOT, RADAR, TORCH, WAND, Equipment
from .learning import TacticalModel
from .memory import PlayerHabits

UPGRADES = ("Speed", "Grip", "Health", "Strength", "Stamina", "Dexterity", "Intelligence")
STAT_GAIN = 0.12
FLOOR_HEAL_FRACTION = 0.50
FLOOR_CLEAR_XP = 4
HARD_ENEMY_XP_PERCENT = 115
BOSS_XP = 12
MIMIC_XP = 10
MERCHANT_ITEMS = ((RADAR, 2), (BLADE, 2), (LASER, 3), (WAND, 3), (TORCH, 1))
NORMAL_PRICES = dict((*MERCHANT_ITEMS, *((item, 3) for item in LOOT if item not in dict(MERCHANT_ITEMS))))
SHOP_ITEMS = tuple((item, max(5, 3 * cost)) for item, cost in NORMAL_PRICES.items())
MERCHANT_PRICES = dict((*NORMAL_PRICES.items(), (HEAL_PACK, 2), *((item, 20) for item in EPIC_ITEMS)))
MIRROR_BLINK_AT = 1.18
BLACKOUT_SECONDS = 0.40

EYE_PHRASES = (
    "You can run, and you CAN hide, but...nevermind...",
    "Allergies are bad this season",
    "What's in my eye? Or...me. Since eye'm.",
    "Why does it seem like this dust is fighting back?",
    "I remember how you held that sword.",
    "The walls told me they were tired.",
    "Next time, turn left. I'd like to see what happens.",
    "I blinked. You used the whole moment.",
    "There are doors behind my eyelids, too.",
    "That is the last time I trust a mirror.",
)

DEFEAT_PHRASES = (
    "You made a very convincing floor tile.",
    "I almost closed my eyes. Almost.",
    "That was the part where you were supposed to duck.",
    "The maze will keep your outline.",
    "The wall would like to apologize. I wouldn't.",
    "I was just getting used to your footsteps.",
    "I had a better view than you did.",
    "The floor has been waiting to meet you.",
    "Tomorrow, we'll move the stain.",
    "I wonder if the mirror will remember you.",
)


class EyeQuotes:
    """Give each outcome one quote and shuffle independently of dungeon generation."""

    def __init__(self):
        self.rng = Random()
        self.decks: dict[bool, list[str]] = {False: [], True: []}
        self.previous: dict[bool, str] = {}

    def draw(self, *, defeated: bool = False, reports: int = 0) -> str:
        deck = self.decks[defeated]
        chorus = "We saw that"
        eligible = [quote for quote in deck if reports or quote != chorus]
        if not eligible:
            deck[:] = DEFEAT_PHRASES if defeated else (*EYE_PHRASES, chorus)
            self.rng.shuffle(deck)
            eligible = [quote for quote in deck if (reports or quote != chorus)
                        and quote != self.previous.get(defeated)]
        quote = eligible[-1]
        deck.remove(quote)
        self.previous[defeated] = quote
        return quote


EYE_QUOTES = EyeQuotes()


def upgrade_cost(rank: int) -> int:
    return 3 + rank * 2


@dataclass
class MimicAppearance:
    extra_eyes: int = 0
    fragments: list[Equipment] = field(default_factory=list)


@dataclass
class MimicMemory:
    radius: float
    habits: PlayerHabits = field(default_factory=PlayerHabits)
    policy: TacticalModel = field(default_factory=lambda: TacticalModel(learning_rate=0.38))
    appearance: MimicAppearance = field(default_factory=MimicAppearance)


@dataclass
class Victory:
    elapsed: float = 0.0
    results: bool = False
    enemy_upgrades: dict[str, int] = field(default_factory=dict)
    enemy_upgrade_xp: int = 0
    quote: str = ""

    @property
    def phase(self) -> str:
        if self.results:
            return "results"
        if self.elapsed < MIRROR_BLINK_AT:
            return "mirror"
        if self.elapsed < MIRROR_BLINK_AT + BLACKOUT_SECONDS:
            return "blackout"
        return "eye"


@dataclass
class Progression:
    level: int = 1
    xp: int = 0
    ranks: dict[str, int] = field(default_factory=dict)
    habits: PlayerHabits = field(default_factory=PlayerHabits)
    policy: TacticalModel = field(default_factory=TacticalModel)
    scribe_reports: int = 0
    score: int = 0
    elapsed_floors: float = 0.0
    enemies_defeated: int = 0
    damage_dealt: int = 0
    damage_taken: int = 0
    loadout: list[Equipment] = field(default_factory=list)
    enemy_xp: int = 0
    enemy_ranks: dict[str, int] = field(default_factory=dict)
    mimics: list[MimicMemory] = field(default_factory=list)
    hard_mode: bool = False
    enemy_xp_fraction: int = 0

    def award_xp(self, amount: int) -> None:
        self.xp += amount
        if self.hard_mode:
            amount, self.enemy_xp_fraction = divmod(
                amount * HARD_ENEMY_XP_PERCENT + self.enemy_xp_fraction, 100
            )
        self.enemy_xp += amount

    def rank(self, upgrade: str) -> int:
        return self.ranks.get(upgrade, 0)

    def bonus(self, upgrade: str) -> float:
        return self.rank(upgrade) * STAT_GAIN

    def enemy_bonus(self, stat: str) -> float:
        return self.enemy_ranks.get(stat, 0) * STAT_GAIN

    def improve_enemies(self) -> dict[str, int]:
        """Buy enemy upgrades with the same prices and gains as the player."""
        weights = self.habits.enemy_upgrade_weights()
        for stat, pressure in self.policy.upgrade_weights().items():
            weights[stat] += pressure
        bought: dict[str, int] = {}
        while True:
            prices = {stat: upgrade_cost(self.enemy_ranks.get(stat, 0)) for stat in UPGRADES}
            affordable = [stat for stat in UPGRADES if self.enemy_xp >= prices[stat]]
            if not affordable:
                return bought
            stat = max(affordable, key=lambda name: weights[name] / (1 + bought.get(name, 0)))
            self.enemy_xp -= prices[stat]
            self.enemy_ranks[stat] = self.enemy_ranks.get(stat, 0) + 1
            bought[stat] = bought.get(stat, 0) + 1

    def cost(self, upgrade: str) -> int:
        return upgrade_cost(self.rank(upgrade))

    def buy(self, upgrade: str) -> bool:
        if upgrade not in UPGRADES or self.xp < self.cost(upgrade):
            return False
        self.xp -= self.cost(upgrade)
        self.ranks[upgrade] = self.rank(upgrade) + 1
        return True

    def buy_equipment(self, spec: Equipment) -> bool:
        price = next((price for item, price in SHOP_ITEMS if item is spec), None)
        if price is None or self.xp < price:
            return False
        self.xp -= price
        return True
