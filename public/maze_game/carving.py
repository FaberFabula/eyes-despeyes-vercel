"""Melee contact geometry for destructible maze walls."""

from .physics import Vec2d
from shapely import Polygon, box

from .layout import TILE, Cell

WALL_HP = 8


def swing_outline(origin: Vec2d, aim: Vec2d, reach: float, half_angle: float) -> list[Vec2d]:
    return [origin] + [
        origin + aim.rotated(half_angle * (index / 4 - 1)) * reach
        for index in range(9)
    ]


def wall_contacts(outline: list[Vec2d], cols: int, rows: int):
    """Yield a contact inside each map tile overlapped by the swing."""
    swing = Polygon([tuple(point) for point in outline])
    left, top, right, bottom = swing.bounds
    for y in range(max(0, int(top // TILE)), min(rows, int(bottom // TILE) + 1)):
        for x in range(max(0, int(left // TILE)), min(cols, int(right // TILE) + 1)):
            tile_area = box(x * TILE, y * TILE, (x + 1) * TILE, (y + 1) * TILE)
            overlap = swing.intersection(tile_area)
            if overlap.area > 0:
                point = overlap.representative_point()
                cell: Cell = (x, y)
                yield cell, Vec2d(point.x, point.y)
