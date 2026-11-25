"""Load the reusable PNG animation sheets and their frame rectangles."""

import json
from functools import lru_cache
from pathlib import Path

import pygame


class Sprites:
    def __init__(self) -> None:
        folder = Path(__file__).with_name("assets")
        self.data = json.loads((folder / "sprites.json").read_text())
        filenames = {entry["sheet"] for entry in self.data.values()}
        self.sheets = {name: pygame.image.load(folder / name).convert_alpha() for name in filenames}

    def natural_width(self, name: str, height: int) -> int:
        width, natural_height = self.data[name]["bounds"]
        return max(1, round(height * width / natural_height))

    def sprite(self, name: str, size: tuple[int, int], state: str = "idle",
               age: float = 0, *, loop: bool = True) -> pygame.Surface:
        category = self.data[name]["states"][state]
        count = len(category["rects"])
        frame = max(0, int(age * category["fps"]))
        index = frame % count if loop else min(frame, count - 1)
        return self.frame(name, state, index, size)

    @lru_cache(maxsize=512)
    def frame(self, name: str, state: str, index: int, size: tuple[int, int]) -> pygame.Surface:
        entry = self.data[name]
        rect = entry["states"][state]["rects"][index]
        frame = self.sheets[entry["sheet"]].subsurface(rect)
        width, height = entry["bounds"]
        padded_size = (round(size[0] * rect[2] / width), round(size[1] * rect[3] / height))
        return pygame.transform.scale(frame, padded_size)

    @lru_cache(maxsize=64)
    def fragment(self, name: str) -> pygame.Surface:
        """Crop a dense quarter of the idle sprite."""
        sprite = self.sprite(name, tuple(self.data[name]["bounds"]))
        bounds = sprite.get_bounding_rect(min_alpha=96)
        patch = pygame.Rect(0, 0, max(1, bounds.width // 2), max(1, bounds.height // 2))
        texture = pygame.mask.from_surface(sprite, 95)
        area = pygame.mask.Mask(patch.size, fill=True)
        corners = [(x, y) for x in (bounds.left, bounds.right - patch.width)
                   for y in (bounds.top, bounds.bottom - patch.height)]
        patch.topleft = max(corners, key=lambda corner: texture.overlap_area(area, corner))
        return sprite.subsurface(patch)
