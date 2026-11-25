"""Translate keys and the mouse into game actions."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygame

from .menu import buy_at_station, debug_menu_input, next_floor_button
from .merchant import buy_from_merchant
from .physics import Vec2d
from .player import aim_at

if TYPE_CHECKING:
    from .session import GameSession

ROTATE_PER_PIXEL = 0.027
SHOP_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5)


@dataclass(frozen=True)
class Frame:
    elapsed: float
    movement: Vec2d
    sprint: bool
    pointer: tuple[float, float]
    external_pause: bool


def movement_input() -> tuple[Vec2d, bool]:
    keys = pygame.key.get_pressed()
    direction = Vec2d(keys[pygame.K_d] - keys[pygame.K_a], keys[pygame.K_s] - keys[pygame.K_w])
    return direction, bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)


def rotate_grip_from_browser(game: "GameSession", pointer: tuple[float, float]) -> None:
    if game.pointer_source is not None and game.previous_pointer is not None and game.locked_aim is not None:
        dx = pointer[0] - game.previous_pointer[0]
        game.maze.wielding.rotate(game.maze.player, dx * ROTATE_PER_PIXEL)
    game.previous_pointer = pointer


def pointer_for(game: "GameSession", frame: Frame, event: pygame.event.Event) -> tuple[float, float]:
    return event.pos if game.pointer_source is None and hasattr(event, "pos") else frame.pointer


def handle_event(game: "GameSession", frame: Frame, event: pygame.event.Event) -> bool:
    if event.type == pygame.QUIT:
        return False
    maze = game.maze
    key = event.key if event.type == pygame.KEYDOWN else None
    if maze.debug is not None and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        if event.button in (1, 3):
            if event.type == pygame.MOUSEBUTTONDOWN:
                maze.debug.mouse_buttons.add(event.button)
            else:
                maze.debug.mouse_buttons.discard(event.button)
    click = pointer_for(game, frame, event) if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 else None
    if game.show_main_menu or game.paused or key == pygame.K_ESCAPE:
        return handle_navigation(game, frame, key, click)
    if event.type == pygame.WINDOWFOCUSLOST:
        game.clear_input()
        return True
    if event.type == pygame.KEYUP and event.key == pygame.K_e:
        game.interact_held = False
    if event.type == pygame.KEYUP and event.key == pygame.K_f:
        maze.wielding.stop_rotation(maze.player)
        game.locked_aim = None
    if game.debug_open:
        if event.type == pygame.MOUSEBUTTONDOWN and maze.debug.mouse_buttons == {1, 3}:
            maze.debug.open("effects")
        else:
            debug_menu_input(game.view, maze, event, pointer_for(game, frame, event))
        game.accumulator = 0.0
        game.clear_grip_input()
    elif event.type == pygame.MOUSEWHEEL:
        game.view.adjust_zoom(event.y)
    elif key in (pygame.K_EQUALS, pygame.K_MINUS):
        game.view.adjust_zoom(1 if key == pygame.K_EQUALS else -1)
    elif key == pygame.K_m:
        game.view.map_expanded = not game.view.map_expanded
    elif maze.won:
        handle_victory(game, key, click)
    elif maze.shop is not None:
        if key == pygame.K_e:
            maze.shop = None
        elif key in SHOP_KEYS:
            buy_from_merchant(maze, SHOP_KEYS.index(key))
    elif not maze.finished:
        handle_player(game, frame, event, key)
    return True


def handle_navigation(game: "GameSession", frame: Frame, key: int | None, click) -> bool:
    primary_rect, secondary_rect = game.view.menu_rects
    primary = click is not None and primary_rect.collidepoint(click)
    secondary = click is not None and secondary_rect.collidepoint(click)
    if game.show_main_menu:
        if key == pygame.K_ESCAPE:
            return False
        if click is not None and game.view.hard_mode_rect.collidepoint(click):
            game.restart(sandbox=False, hard_mode=True)
        elif primary or secondary or key in (pygame.K_RETURN, pygame.K_s):
            game.restart(sandbox=secondary or key == pygame.K_s, hard_mode=False)
    elif key == pygame.K_ESCAPE and not frame.external_pause:
        if game.paused:
            game.paused = False
            game.clear_input()
        else:
            game.pause()
        game.accumulator = 0.0
    elif game.paused and key != pygame.K_ESCAPE:
        if click is not None and game.view.pause_restart_rect.collidepoint(click):
            game.restart()
        elif primary or key == pygame.K_RETURN:
            game.paused = False
        elif secondary or key == pygame.K_HOME:
            if game.in_browser:
                return False
            game.show_main_menu, game.paused = True, False
    return True


def handle_victory(game: "GameSession", key: int | None, click) -> None:
    maze = game.maze
    phase = maze.victory.phase
    if phase == "eye" and key == pygame.K_RETURN:
        maze.victory.results = True
    elif phase == "results":
        if key == pygame.K_RETURN or (click is not None and next_floor_button(game.view).collidepoint(click)):
            game.next_level()
        elif click is not None:
            buy_at_station(game.view, maze, click)


def handle_player(game: "GameSession", frame: Frame, event: pygame.event.Event, key: int | None) -> None:
    maze = game.maze
    player = maze.player
    held = maze.item_actions.held_by(player)
    if key == pygame.K_e:
        game.interact_held = not maze.player_interact(aim_at(player, game.view.mouse_offset(frame.pointer)))
        if maze.shop is not None:
            maze.item_actions.clear_input(player)
    elif key == pygame.K_SPACE:
        maze.dodge(frame.movement)
    elif key == pygame.K_q:
        maze.wielding.release(player)
        game.locked_aim = None
    elif key == pygame.K_f and held is not None:
        game.locked_aim = game.view.mouse_offset(frame.pointer)
        maze.wielding.start_rotation(player)
    elif event.type == pygame.MOUSEMOTION and game.locked_aim is not None and game.pointer_source is None:
        maze.wielding.rotate(player, event.rel[0] * ROTATE_PER_PIXEL)
    elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
        offset = game.view.mouse_offset(pointer_for(game, frame, event))
        aim_at(player, game.locked_aim if game.locked_aim is not None else offset)
        target = player.position + offset
        if (maze.debug is not None and maze.debug.mouse_buttons == {1, 3}
                and held is not None and held.item.spec.effect == "debug"):
            maze.debug.open("effects", target)
        elif event.button == 1:
            maze.item_actions.press_primary(player, target)
        else:
            maze.item_actions.special(player, target)
    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            maze.item_actions.release_primary(player)
        elif event.button == 3:
            maze.item_actions.release_special(player)
