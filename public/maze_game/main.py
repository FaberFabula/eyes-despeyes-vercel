"""Run the game on desktop or in the browser."""

import asyncio
from collections.abc import Awaitable, Callable

from .session import GameSession
from .hard_mode import Prior, Report


async def run(next_frame: Callable[[], Awaitable[None]] | None = None, *, sandbox: bool = False,
              hard_mode: bool = False, learning_prior: Prior = "[]", report_learning: Report | None = None,
              play_sound: Callable[[str, float, float], None] | None = None,
              browser_pointer: Callable[[], tuple[float, float]] | None = None,
              browser_size: Callable[[], tuple[int, int]] | None = None,
              browser_pause: Callable[[], bool] | None = None) -> None:
    game = GameSession(sandbox=sandbox, in_browser=next_frame is not None, play_sound=play_sound,
                       pointer=browser_pointer, size=browser_size, pause=browser_pause,
                       hard_mode=hard_mode, learning_prior=learning_prior, report_learning=report_learning)
    try:
        while True:
            frame = game.read_frame()
            if not game.handle_input(frame):
                break
            game.advance(frame)
            game.draw(frame)
            if next_frame is not None:
                await next_frame()
    finally:
        game.close()


def main() -> None:
    asyncio.run(run())
