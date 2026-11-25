"""Play positional game cues through Pygame or the browser's audio mixer."""

from collections.abc import Callable
from pathlib import Path

import pygame

from .effects import Cue

SAMPLE_RATE = 22050
ASSET_DIRECTORY = Path(__file__).with_name("assets") / "audio"


class SoundCues:
    def __init__(self, play_sample: Callable[[str, float, float], None] | None = None) -> None:
        self.play_sample = play_sample
        self.effects: dict[str, pygame.mixer.Sound] = {}
        if play_sample is None:
            try:
                pygame.mixer.init(SAMPLE_RATE, -16, 2, 512)
                self.effects = {path.stem: pygame.mixer.Sound(str(path))
                                for path in ASSET_DIRECTORY.glob("*.wav")}
            except pygame.error:
                pass

    def play(self, cues: list[Cue]) -> None:
        nearby = {}
        for cue in cues:
            key = (cue.name, round(cue.pan))
            previous = nearby.get(key)
            if previous is None or cue.volume > previous.volume:
                nearby[key] = cue
        for cue in sorted(nearby.values(), key=lambda entry: -entry.volume)[:6]:
            if self.play_sample is not None:
                self.play_sample(cue.name, cue.volume, cue.pan)
            elif (sound := self.effects.get(cue.name)) is not None:
                channel = sound.play()
                if channel is not None:
                    channel.set_volume(cue.volume * (1 - max(0, cue.pan)),
                                       cue.volume * (1 + min(0, cue.pan)))
        cues.clear()
