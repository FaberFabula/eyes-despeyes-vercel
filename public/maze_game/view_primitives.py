"""Drawing helpers shared by the maze and its menus."""

from math import cos, pi, sin, sqrt
import pygame
from .physics import Vec2d
from .item import Equipment


def point(vector: Vec2d) -> tuple[int, int]:
    return round(vector.x), round(vector.y)


def closed_eye(surface: pygame.Surface, position: Vec2d, width: int, color: tuple[int, int, int]) -> None:
    box = pygame.Rect(0, 0, width, width // 2)
    box.center = point(position)
    pygame.draw.arc(surface, color, box, pi, 2 * pi, 2)


def gear_sprite_name(spec: Equipment) -> str:
    asset_names = {"sword": "blade", "whip": "cat_o_nine_tails", "launcher": "incoming_tax"}
    return asset_names.get(spec.effect, spec.name.lower().replace(" ", "_"))


def scribe_eye_position(index: int, midpoint: Vec2d) -> Vec2d:
    angle = (index + 1) * pi * (3 - sqrt(5))
    radius = 0.53 + 0.34 * (((index + 1) * (sqrt(2) - 1)) % 1)
    return midpoint + Vec2d(cos(angle) * min(410, midpoint.x - 36), sin(angle) * 242) * radius


def draw_eye(surface: pygame.Surface, position: Vec2d, width: int, gaze: Vec2d,
             *, bloodshot: bool = False) -> None:
    rect = pygame.Rect(0, 0, width, int(width * 0.52))
    rect.center = point(position)
    if width < 10:
        pygame.draw.ellipse(surface, (111, 26, 41), rect.inflate(2, 2))
        pygame.draw.ellipse(surface, (228, 119, 116), rect)
        pygame.draw.circle(surface, (35, 11, 22), rect.center, max(0, width // 5))
        return
    pygame.draw.ellipse(surface, (104, 37, 51), rect.inflate(6, 6))
    pygame.draw.ellipse(surface, (234, 120, 118) if bloodshot else (205, 154, 137), rect)
    offset = gaze - position
    iris = position + offset.normalized() * min(width * 0.15, offset.length)
    pygame.draw.circle(surface, (110, 36, 48), point(iris), max(2, int(width * 0.20)))
    pygame.draw.circle(surface, (15, 9, 15), point(iris), max(1, int(width * 0.12)))
    glint = iris + Vec2d(-width * 0.038, -width * 0.045)
    pygame.draw.circle(surface, (244, 210, 182), point(glint), max(1, int(width * 0.025)))
    if bloodshot:
        for side in (-1, 1):
            base = position + Vec2d(side * width * 0.42, width * 0.04)
            branch = base + Vec2d(-side * width * 0.10, -width * 0.05)
            veins = (base, branch, branch + Vec2d(-side * width * 0.08, width * 0.10))
            pygame.draw.lines(surface, (121, 23, 42), False, [point(vertex) for vertex in veins], 1)
