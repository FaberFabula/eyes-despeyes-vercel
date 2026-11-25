"""Connect frame input, simulation, drawing, and sound."""

import pygame

from .menu import draw_main_menu, draw_pause
from . import controls
from .audio import SoundCues
from .hard_mode import Prior, Report, hard_progression
from .player import aim_at
from .settings import DESKTOP_FPS, MAX_FRAME_TIME, PHYSICS_STEP, TITLE
from .view import WINDOW_SIZE, MazeView
from .world import Maze

BROWSER_MAX_STEPS = 6


class GameSession:
    def __init__(self, *, sandbox, in_browser, play_sound, pointer, size, pause,
                 hard_mode: bool = False, learning_prior: Prior = "[]", report_learning: Report | None = None):
        pygame.display.init()
        pygame.font.init()
        self.size_source = size
        self.pointer_source = pointer
        self.pause_source = pause
        self.in_browser = in_browser
        self.sandbox = sandbox
        self.hard_mode = hard_mode and not sandbox
        self.learning_prior = learning_prior
        self.report_learning = report_learning
        initial_size = tuple(size()) if size is not None else WINDOW_SIZE
        self.screen = pygame.display.set_mode(initial_size)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.view = MazeView(initial_size)
        self.sounds = SoundCues(play_sound)
        self.maze = self.new_maze()
        self.floor_reported = False
        self.record_run_event("start")
        self.accumulator = 0.0
        self.paused = False
        self.show_main_menu = False
        self.interact_held = False
        self.locked_aim = None
        self.previous_pointer = None

    @property
    def debug_open(self) -> bool:
        return self.maze.debug is not None and self.maze.debug.is_open

    def new_maze(self) -> Maze:
        if not self.hard_mode:
            return Maze(sandbox=self.sandbox)
        prior = self.learning_prior() if callable(self.learning_prior) else self.learning_prior
        progress = hard_progression(prior, self.report_learning)
        return Maze(progress=progress, sandbox=self.sandbox)

    def record_run_event(self, event: str) -> None:
        if self.hard_mode and self.report_learning is not None:
            if event in ("defeat", "restart", "leave"):
                self.maze.settle_hard_learning()
            self.report_learning(event, str(self.maze.progress.level))

    def aim_offset(self, pointer):
        return self.locked_aim if self.locked_aim is not None else self.view.mouse_offset(pointer)

    def read_frame(self) -> controls.Frame:
        elapsed = min(self.clock.tick(0 if self.in_browser else DESKTOP_FPS) / 1000, MAX_FRAME_TIME)
        self.accumulator += elapsed
        if self.in_browser:
            # Drop excess backlog after a stall so rendering gets another turn.
            self.accumulator = min(self.accumulator, BROWSER_MAX_STEPS * PHYSICS_STEP)
        self.resize_window()
        external_pause = self.pause_source is not None and self.pause_source()
        if external_pause and not self.paused:
            self.pause()
        pointer = tuple(self.pointer_source()) if self.pointer_source is not None else pygame.mouse.get_pos()
        controls.rotate_grip_from_browser(self, pointer)
        if not (self.paused or self.show_main_menu or self.maze.finished or self.debug_open):
            aim_at(self.maze.player, self.aim_offset(pointer))
        movement, sprint = controls.movement_input()
        return controls.Frame(elapsed, movement, sprint, pointer, external_pause)

    def resize_window(self) -> None:
        if self.size_source is not None:
            size = tuple(self.size_source())
            if size != self.view.screen_size:
                self.screen = pygame.display.set_mode(size)
                self.view.set_screen_size(size)

    def handle_input(self, frame: controls.Frame) -> bool:
        for event in pygame.event.get():
            if not controls.handle_event(self, frame, event):
                return False
        return True

    def advance(self, frame: controls.Frame) -> None:
        maze = self.maze
        if maze.player not in maze.wielding.held:
            self.locked_aim = None
        if self.show_main_menu or self.paused or self.debug_open:
            self.accumulator = 0.0
            return
        offset = self.aim_offset(frame.pointer)
        while self.accumulator >= PHYSICS_STEP and not maze.finished:
            aim_at(maze.player, offset)
            cursor = self.view.hand_cursor(maze.player, offset)
            maze.step(PHYSICS_STEP, frame.movement, cursor, frame.sprint, self.interact_held)
            self.accumulator -= PHYSICS_STEP
        if maze.finished:
            if not self.floor_reported:
                self.record_run_event("floor_clear" if maze.won else "defeat")
                self.floor_reported = True
            maze.advance_presentation(frame.elapsed)
            self.accumulator = 0.0
        self.sounds.play(maze.cues)

    def draw(self, frame: controls.Frame) -> None:
        if self.show_main_menu:
            draw_main_menu(self.view, self.screen, frame.pointer)
        elif self.paused:
            draw_pause(self.view, self.screen, frame.pointer)
        else:
            cursor = self.maze.player.position + self.aim_offset(frame.pointer)
            self.view.draw(self.screen, self.maze, cursor, screen_pointer=frame.pointer)
        pygame.display.flip()

    def clear_grip_input(self) -> None:
        self.interact_held = False
        self.locked_aim = None

    def clear_input(self) -> None:
        self.maze.item_actions.clear_input(self.maze.player)
        self.clear_grip_input()
        if self.maze.debug is not None:
            self.maze.debug.mouse_buttons.clear()

    def pause(self) -> None:
        self.paused = True
        self.view.capture_game(self.screen)
        self.clear_input()

    def restart(self, *, sandbox: bool | None = None, hard_mode: bool | None = None) -> None:
        self.record_run_event("restart")
        if sandbox is not None:
            self.sandbox = sandbox
        if hard_mode is not None:
            self.hard_mode = hard_mode
        self.hard_mode = self.hard_mode and not self.sandbox
        self.maze = self.new_maze()
        self.floor_reported = False
        self.record_run_event("start")
        self.paused = self.show_main_menu = False
        self.accumulator = 0.0
        self.clear_grip_input()

    def next_level(self) -> None:
        self.maze = self.maze.next_level()
        self.floor_reported = False
        self.record_run_event("floor_start")
        self.accumulator = 0.0
        self.clear_grip_input()

    def close(self) -> None:
        self.record_run_event("leave")
        pygame.quit()
