from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
from random import Random

from .physics import Vec2d

TILE = 36
ZONE_COLUMNS, ZONE_ROWS = 4, 3
ZONE_WIDTH, ZONE_HEIGHT = 15, 16
CORRIDOR_TILES = 5
HALL_START = ZONE_HEIGHT - CORRIDOR_TILES
COLS = ZONE_COLUMNS * ZONE_WIDTH + 1
ROWS = ZONE_ROWS * ZONE_HEIGHT + 1
Cell = tuple[int, int]
LIT_TILES = "LO"
BASIC_MOBS = ("scout", "claw")
ROOM_GUARDS = ("guard", "armored")
EYE_MOBS = ("elite", "scribe")
# Each floor sets the weights for basic mobs and room guards, plus a cap for stronger Eyes.
FLOOR_MOB_POOLS = (
    (7, 3, 0), (6, 4, 0), (6, 4, 1), (5, 5, 1), (5, 5, 1),
    (5, 5, 2), (4, 6, 2), (4, 6, 2), (4, 6, 3), (3, 7, 3),
)
MIMIC_COUNT_WEIGHTS = ((9, 1, 0), (7, 3, 0), (5, 3, 2))


@dataclass(frozen=True)
class Room:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def middle(self) -> Cell:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    def cells(self) -> list[Cell]:
        return [(x, y) for y in range(self.top, self.bottom + 1)
                for x in range(self.left, self.right + 1)]

    def contains(self, cell: Cell) -> bool:
        x, y = cell
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass(frozen=True)
class LevelLayout:
    tiles: tuple[str, ...]
    gate_room: Room
    rooms: tuple[Room, ...]
    gate: Cell
    exit: Cell
    eye: Cell
    player: Cell
    key: Cell
    treasure: tuple[Cell, ...]
    loot: tuple[Cell, ...]
    gate_supply: tuple[Cell, ...]
    guards: tuple[tuple[str, tuple[Cell, ...]], ...]
    bosses: tuple[Cell, ...]
    jambs: dict[Cell, Cell]
    merchants: tuple[Cell, ...]
    mimics: tuple[Cell, ...]
    torches: tuple[Cell, ...]
    targets: tuple[Cell, ...] = ()

    def tile(self, cell: Cell) -> str:
        x, y = cell
        if 0 <= x < COLS and 0 <= y < ROWS:
            symbol = self.tiles[y][x]
            return "#" if symbol == "X" else symbol
        return "#"

    def room_at(self, cell: Cell) -> Room | None:
        return next((room for room in self.rooms if room.contains(cell)), None)

    def reinforced(self, cell: Cell) -> bool:
        x, y = cell
        return self.tiles[y][x] == "X"


def mob_roster(rng: Random, floor: int) -> list[str]:
    basic_weight, guard_weight, eye_limit = FLOOR_MOB_POOLS[floor - 1]
    room_guards = ROOM_GUARDS[:1] if floor == 1 else ROOM_GUARDS
    patrols = rng.randint(3, 4)
    eyes = rng.randint(0, min(eye_limit, patrols - 2))
    pools = rng.choices((BASIC_MOBS, room_guards), weights=(basic_weight, guard_weight),
                        k=patrols - eyes - 2)
    kinds = [rng.choice(BASIC_MOBS), rng.choice(room_guards)]
    kinds.extend(rng.choice(pool) for pool in pools)
    kinds.extend(rng.choices(EYE_MOBS, k=eyes))
    rng.shuffle(kinds)
    return kinds


def generate_level(rng: Random, level: int = 1, *, mimic_minimum: int = 0) -> LevelLayout:
    """Connect the rooms with wide corridors and draw a roster from floor-specific pools."""
    grid = [["#"] * COLS for _ in range(ROWS)]
    last_col, last_row = ZONE_COLUMNS - 1, ZONE_ROWS - 1
    corners = ((0, 0), (last_col, 0), (0, last_row), (last_col, last_row))
    gate_zone = rng.choice(corners)
    start_zone = (last_col - gate_zone[0], last_row - gate_zone[1])

    def make_room(zone: Cell) -> Room:
        column, row = zone
        gate = zone == gate_zone
        width = rng.randint(9, 10) if gate else rng.randint(7, 10)
        height = rng.randint(8, 9) if gate else rng.randint(6, 8)
        left_edge = (2 if column == 0 else 4) if gate else 3
        left = column * ZONE_WIDTH + rng.randint(left_edge, left_edge + 10 - width)
        top = row * ZONE_HEIGHT + rng.randint(1 if gate else 2, HALL_START - height - 1)
        return Room(left, top, left + width - 1, top + height - 1)

    def carve(left: int, top: int, width: int, height: int) -> None:
        for y in range(top, top + height):
            grid[y][left:left + width] = ["."] * width

    zones = [start_zone, *((x, y) for y in range(ZONE_ROWS) for x in range(ZONE_COLUMNS)
                           if (x, y) not in (gate_zone, start_zone))]
    gate_room = make_room(gate_zone)
    rooms = {zone: make_room(zone) for zone in zones}
    # Protect the gate chamber before carving the rooms or hall network.
    all_rooms = {gate_zone: gate_room, **rooms}
    for y in range(gate_room.top - 1, gate_room.bottom + 2):
        for x in range(gate_room.left - 1, gate_room.right + 2):
            grid[y][x] = "X"

    lit_zone = rng.choice(zones[1:])
    other_lit_zone = rng.choice([zone for zone in zones[1:] if zone != lit_zone])
    lit_zones = (lit_zone, other_lit_zone)
    player, eye = rooms[start_zone].middle, rooms[lit_zone].middle
    eyes = (eye, rooms[other_lit_zone].middle)
    exit_cell = (gate_room.middle[0], gate_room.top + 1)
    bosses = ((gate_room.left + 2, gate_room.top + 2),
              (gate_room.right - 2, gate_room.top + 2))
    reserved = {player, exit_cell, *eyes, *bosses}
    for zone, room in all_rooms.items():
        for x, y in room.cells():
            grid[y][x] = "L" if zone in lit_zones else "."
    for x, y in eyes:
        grid[y][x] = "O"

    loot, treasure = [], []
    for zone, room in rooms.items():
        cells = [cell for cell in room.cells() if cell not in reserved]
        gear_cells = [] if zone == start_zone else rng.sample(cells, 2)
        chest = rng.choice([cell for cell in cells if cell not in gear_cells])
        loot.extend(gear_cells)
        treasure.append(chest)
        reserved.update((*gear_cells, chest))

    halls = [row * ZONE_HEIGHT + HALL_START for row in range(ZONE_ROWS)]
    half_corridor = CORRIDOR_TILES // 2
    for hall in halls:
        carve(1, hall, COLS - 2, CORRIDOR_TILES)
    for earlier, later in zip(halls, halls[1:]):
        for column in rng.sample(range(1, ZONE_COLUMNS), 2):
            left = column * ZONE_WIDTH - half_corridor
            carve(left, earlier, CORRIDOR_TILES, later - earlier + CORRIDOR_TILES)

    jambs: dict[Cell, Cell] = {}
    for zone, room in all_rooms.items():
        door_y = room.bottom + 1
        approach = rng.randint(room.left, room.right - CORRIDOR_TILES + 1)
        x = approach + half_corridor
        companion_x = x + rng.choice((-1, 1))
        carve(approach, door_y + 1, CORRIDOR_TILES, halls[zone[1]] - door_y)
        door = (x, door_y)
        jambs[door] = (companion_x, door_y)
        grid[door_y][x] = "J" if zone == gate_zone else "D"
        if zone == gate_zone:
            gate = door
            grid[door_y][companion_x] = "G"
        if zone == start_zone:
            first_corpse = (x, halls[zone[1]] + 1)

    for row in range(ZONE_ROWS):
        for column in range(ZONE_COLUMNS - 1):
            left, right = rooms.get((column, row)), rooms.get((column + 1, row))
            if left is None or right is None or rng.random() >= 0.55:
                continue
            top, bottom = max(left.top, right.top), min(left.bottom, right.bottom)
            if bottom - top + 1 >= CORRIDOR_TILES:
                y = rng.randint(top, bottom - CORRIDOR_TILES + 1)
                carve(left.right + 1, y, right.left - left.right - 1, CORRIDOR_TILES)

    # Put a first weapon down the hall from spawn. Space extra bodies along
    # the edge to leave the center clear for movement.
    loot.insert(0, first_corpse)
    reserved.add(first_corpse)
    for hall in halls:
        edge = hall + CORRIDOR_TILES - 1
        for start in range(2, COLS - 7, 24):
            cells = [(x, edge) for x in range(start, start + 7)
                     if (x, edge) not in reserved and walk_distance((x, edge), first_corpse) >= 3]
            if cells:
                corpse = rng.choice(cells)
                loot.append(corpse)
                reserved.add(corpse)

    outside = [(x, y) for y, row in enumerate(grid) for x, symbol in enumerate(row)
               if symbol in ".L" and not gate_room.contains((x, y)) and (x, y) not in reserved]
    gate_supply = tuple(sorted(outside, key=lambda cell: walk_distance(cell, gate))[:1])
    reserved.update(gate_supply)
    key_cells = [cell for zone in lit_zones for cell in rooms[zone].cells() if cell not in reserved]
    key = rng.choice(key_cells)
    reserved.add(key)

    floor = min(len(FLOOR_MOB_POOLS), max(1, level))
    kinds = mob_roster(rng, floor)
    patrol_zones = zones[1:]
    rng.shuffle(patrol_zones)
    if level == 1:
        patrol_zones.sort(key=lambda zone: walk_distance(zone, start_zone) <= 1)
    shared_rooms = max(0, len(kinds) - len(patrol_zones))
    patrol_zones.extend(rng.sample(patrol_zones, shared_rooms))
    guards = []
    for kind, zone in zip(kinds, patrol_zones):
        patrol = tuple(rng.sample([cell for cell in rooms[zone].cells() if cell not in reserved], 3))
        guards.append((kind, patrol))
        reserved.add(patrol[0])
    free_room_cells = [cell for zone, room in rooms.items() if zone != start_zone
                       for cell in room.cells() if cell not in reserved]
    merchants = tuple(rng.sample(free_room_cells, 2))
    reserved.update(merchants)
    tier = min((floor - 1) // 3, len(MIMIC_COUNT_WEIGHTS) - 1)
    mimic_weights = (1, 0, 0) if floor == 1 else MIMIC_COUNT_WEIGHTS[tier]
    mimic_count = max(mimic_minimum, rng.choices((1, 2, 3), weights=mimic_weights)[0])
    mimic_rooms = rng.sample(zones[1:], mimic_count)
    mimics = tuple(rng.choice([cell for cell in rooms[zone].cells() if cell not in reserved])
                   for zone in mimic_rooms)
    reserved.update(mimics)

    torches = []
    for zone, room in all_rooms.items():
        cells = [cell for cell in room.cells() if cell not in reserved]
        if zone == start_zone:
            cells.sort(key=lambda cell: (abs(walk_distance(cell, player) - 2), cell))
            chosen = [cells[0]]
        else:
            edges = [cell for cell in cells if cell[0] in (room.left, room.right)]
            chosen = rng.sample(edges, 1)
        torches.extend(chosen)
        reserved.update(chosen)
    for hall in halls:
        torches.extend((x, hall) for x in range(5, COLS - 3, 12)
                       if (x, hall) not in reserved and (x, hall + 1) not in reserved)

    return LevelLayout(
        tuple("".join(row) for row in grid), gate_room, (gate_room, *rooms.values()),
        gate, exit_cell, eye, player, key, tuple(treasure), tuple(loot), gate_supply,
        tuple(guards), bosses, jambs, merchants, mimics, tuple(torches),
    )


def generate_sandbox(gear_count: int) -> LevelLayout:
    """An isolated practice hall with a door, target dummies, and room to wield gear."""
    x, y = COLS // 2, ROWS // 2
    hall = Room(x - 10, y - 6, x + 8, y + 5)
    gate_room = Room(x + 10, y - 3, x + 15, y + 3)
    supply_room = Room(x - 18, y - 3, x - 12, y + 3)
    gate = (x + 9, y)
    jamb = (x + 9, y + 1)
    player = (x, y + 2)
    eye = (x + 2, y - 4)
    key = (x + 3, y - 5)
    grid = [["#"] * COLS for _ in range(ROWS)]
    for tx, ty in hall.cells():
        grid[ty][tx] = "L" if x - 4 <= tx <= x + 3 else "."
    grid[eye[1]][eye[0]] = "O"
    for tx, ty in gate_room.cells():
        grid[ty][tx] = "."
    grid[gate[1]][gate[0]] = "J"
    grid[jamb[1]][jamb[0]] = "G"
    for tx, ty in supply_room.cells():
        grid[ty][tx] = "L"
    half_corridor = CORRIDOR_TILES // 2
    for ty in range(y - half_corridor, y + half_corridor + 1):
        grid[ty][x - 11] = "."

    targets = ((x - 5, y), (x + 6, y - 2), (x + 6, y + 2))
    shelf = sorted(
        ((tx, ty) for tx in range(x - 6, x + 7, 3) for ty in range(y - 6, player[1], 2)
         if all(walk_distance((tx, ty), target) > 1 for target in (*targets, eye, key))),
        key=lambda cell: (walk_distance(cell, player), cell[1], cell[0]),
    )
    return LevelLayout(
        tiles=tuple("".join(row) for row in grid),
        gate_room=gate_room, rooms=(hall, gate_room, supply_room), gate=gate, exit=(x + 13, y),
        eye=eye, player=player, key=key,
        treasure=((x - 6, y + 4), (x - 3, y + 4), (x, y + 4), (x + 3, y + 4), (x + 6, y + 4)),
        loot=tuple(shelf[:gear_count]), gate_supply=(),
        guards=(("armored", ((x - 15, y), (x - 16, y + 2))),), bosses=(),
        jambs={gate: jamb}, merchants=(), mimics=(),
        torches=((x - 2, y + 3), (x + 4, y + 1), (x - 7, y - 1)),
        targets=targets,
    )


def tile_at(position) -> Cell:
    x, y = position
    return int(x // TILE), int(y // TILE)


def center(cell: Cell) -> Vec2d:
    x, y = cell
    return Vec2d((x + 0.5) * TILE, (y + 0.5) * TILE)


def walk_distance(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_path(start: Cell, goal: Cell, tile_at_cell, *,
              can_cross: Callable[[Cell, Cell], bool] | None = None,
              closest: bool = False, step_cost: Callable[[Cell], float] | None = None,
              remaining_distance: Callable[[Cell], float] | None = None) -> list[Cell]:
    """A* with optional clearance checks and zero distance as the goal condition."""
    if tile_at_cell(start) == "#" or (not closest and remaining_distance is None and tile_at_cell(goal) == "#"):
        return []
    def estimate(cell: Cell) -> float:
        return walk_distance(cell, goal) if remaining_distance is None else remaining_distance(cell)

    frontier = [(estimate(start), 0, start)]
    distance = {start: 0}
    previous = {}
    reached = start
    while frontier:
        _, steps, cell = heappop(frontier)
        if steps != distance[cell]:
            continue
        remaining = estimate(cell)
        if remaining == 0:
            reached = cell
            break
        if closest and remaining < estimate(reached):
            reached = cell

        x, y = cell
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if tile_at_cell(neighbor) == "#":
                continue
            cost = steps + (1 if step_cost is None else step_cost(neighbor))
            if cost >= distance.get(neighbor, float("inf")):
                continue
            if can_cross is not None and not can_cross(cell, neighbor):
                continue
            distance[neighbor] = cost
            previous[neighbor] = cell
            heappush(frontier, (cost + estimate(neighbor), cost, neighbor))
    route = []
    while reached != start:
        route.append(reached)
        reached = previous[reached]
    return list(reversed(route))
