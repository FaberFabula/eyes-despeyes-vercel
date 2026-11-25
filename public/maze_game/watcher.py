"""Room Eyes watch, summon, and lose sight when broken."""

from typing import TYPE_CHECKING

from .entity import RoomEye
from .layout import BASIC_MOBS, EYE_MOBS, FLOOR_MOB_POOLS, ROOM_GUARDS, center, tile_at

if TYPE_CHECKING:
    from .world import Maze

EYE_HP = 12
SUMMON_INTERVAL = 25.0
DEFENSE_INTERVAL = 8.0


def eye_population_cap(level: int) -> int:
    return 9 + (level - 1) // 2


def place_eyes(maze: "Maze") -> list[RoomEye]:
    eyes = []
    for y, row in enumerate(maze.layout.tiles):
        for x, tile in enumerate(row):
            if tile == "O" and (room := maze.layout.room_at((x, y))) is not None:
                shape = maze.physics.add_fixed_target(center((x, y)), 11)
                eyes.append(RoomEye((x, y), room, shape, EYE_HP))
    return eyes


def eye_in_player_room(maze: "Maze") -> RoomEye | None:
    if not maze.player.active:
        return None
    position = maze.player.position
    cell = tile_at(position)
    return next((eye for eye in maze.eyes if eye.alive and eye.room.contains(cell)), None)


def update_eyes(maze: "Maze", dt: float) -> None:
    defending = eye_in_player_room(maze)
    maze.watching_eye = (defending if defending is not None
                         and maze.clear_path(defending.position, maze.player.position) else None)
    if maze.sandbox or not maze.player.active:
        return
    interval = DEFENSE_INTERVAL if defending is not None else SUMMON_INTERVAL
    maze.eye_spawn_charge = min(1.0, maze.eye_spawn_charge + dt / interval)
    if (maze.eye_spawn_charge + 1e-9 < 1 or maze.wielding.time < maze.eye_spawn_retry
            or not maze.can_summon_eye_mob()):
        return
    eyes = [eye for eye in maze.eyes if eye.alive and eye is not defending]
    maze.rng.shuffle(eyes)
    candidates = [defending, *eyes] if defending is not None else eyes
    if any(summon_eye_guard(maze, eye, defending=eye is defending) for eye in candidates):
        maze.eye_spawn_charge = 0.0
    else:
        maze.eye_spawn_retry = maze.wielding.time + 1


def summon_eye_guard(maze: "Maze", eye: RoomEye, *, defending: bool = False) -> bool:
    cells = [cell for cell in eye.room.cells() if cell != eye.cell
             and maze.player.distance_to(center(cell)) >= 72 and maze.spawn_clear(cell)]
    if not cells:
        return False
    cell = maze.rng.choice(cells)
    floor = min(maze.progress.level, len(FLOOR_MOB_POOLS))
    basic, guard_weight, elite = FLOOR_MOB_POOLS[floor - 1]
    guards = ROOM_GUARDS[:1] if floor == 1 else ROOM_GUARDS
    pool = maze.rng.choices((BASIC_MOBS, guards, EYE_MOBS), weights=(basic, guard_weight, elite))[0]
    kind = maze.rng.choice(pool)
    if kind == "scribe":
        guard = maze.add_scribe(cell, (cell, eye.cell))
    else:
        guard = maze.spawn_guard(kind, cell, (cell, eye.cell))
    if defending:
        guard.hear(tile_at(maze.player.position), 9)
    maze.mob_sound(guard, "alert")
    return True


def damage_eye(maze: "Maze", eye: RoomEye, damage: int) -> int:
    dealt = min(eye.hp, damage)
    eye.hp -= dealt
    maze.note_hit(eye.shape, eye.position, dealt, "stone")
    if not eye.alive:
        maze.physics.remove_shape(eye.shape)
        del maze.eyes_by_shape[eye.shape]
        if maze.watching_eye is eye:
            maze.watching_eye = None
            maze.eye_tip = None
    return dealt
