"""Create the player and apply movement and aim."""

from typing import TYPE_CHECKING

from .actor import Actor
from .physics import Vec2d

if TYPE_CHECKING:
    from .world import Maze

PLAYER_SPEED = 153
PLAYER_HEALTH = 28
PLAYER_DAMAGE_SCALE = 0.75


def create_player(maze: "Maze") -> Actor:
    player = Actor(maze.physics, maze.layout.player, speed=PLAYER_SPEED, hp=PLAYER_HEALTH, kind="player")
    player.apply_upgrades(maze.progress.bonus)
    return player


def aim_at(player: Actor, offset: Vec2d) -> Vec2d:
    if offset.length:
        player.aim = offset.normalized()
    return player.position + offset


def prepare_movement(maze: "Maze", direction: Vec2d, sprint: bool, dt: float) -> None:
    player = maze.player
    player.tick_timers(dt)
    player.facing = player.aim
    held = maze.item_actions.held_by(player)
    if player.active:
        player.move(direction, sprint and (held is None or held.rush is None))
    player.update_stamina(dt)
