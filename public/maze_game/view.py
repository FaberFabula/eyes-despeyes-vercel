"""Draw a player-centered slice of the maze and its light mask."""

from math import ceil, cos, degrees, hypot, pi, sin, sqrt

import pygame
from .menu import draw_debug_menu, draw_end, draw_eye_report, draw_merchant_shop
from .watcher import EYE_HP
from .physics import Vec2d

from .view_primitives import closed_eye, draw_eye, gear_sprite_name, point
from .carving import swing_outline
from .effects import DEATH_ANIMATION_SECONDS, Death
from .enemy import Enemy
from .entity import DOOR_HP, Chest, Door
from .actor import Actor
from .item import Equipment
from .merchant import Merchant
from .item_effects import DRAIN_RADIUS, EPIC_PURPLE, GRIMOIRE_COOLDOWN, GRIMOIRE_PAGES, SEAL_LIMIT
from .layout import COLS, LIT_TILES, ROWS, TILE, Cell, center
from .enemy_behaviors import disguise_eye_width, disguise_position, is_disguised, tendril_tip
from .mimic_body import MIMIC_BASE_RADIUS, MIMIC_BURST_DELAY, MimicCorpse
from .progression import MIRROR_BLINK_AT, MimicAppearance
from .sprite_motion import mimic_legs
from .sprites import Sprites
from .item_debug import EFFECT_PAGE_SIZE, GOLD, SUMMON_ITEMS, SUMMON_MOBS, SandboxTools
from .item_debug_effects import EFFECT_OPTIONS
from .item_physics import GEAR_COLORS, HAND_REACH, HeldGear, MIN_HAND_REACH
from .item import Gear
from .world import ATTACKER_NAMES, MIRROR_ESCAPE_SECONDS, SWING_HALF_ANGLE, VISION_DISTANCE, VISION_HALF_ANGLE, Maze

WINDOW_SIZE = (828, 700)
ZOOM_LEVELS = (1.05, 1.25, 1.50, 1.75, 2.0, 2.3)
CONE_SPAN = 2 * VISION_HALF_ANGLE
MAP_FLOOR = (88, 90, 103)
MAP_WALL = (44, 43, 54)
MAP_REINFORCED = (99, 116, 132)
MAP_LIGHT = (120, 107, 89)
MAP_DOOR = (139, 89, 81)
MAP_TERRAIN_COLORS = {"#": MAP_WALL, "X": MAP_REINFORCED, "D": MAP_DOOR}
MAP_LOOT = (249, 204, 77)
RADAR_BEARING = (92, 209, 220)
RADAR_ROUTE = (172, 248, 205)
MIMIC_DISGUISES = {
    "chest": ("chest", (25, 20)), "merchant": ("merchant", (23, 31)),
}


def draw_dashes(surface: pygame.Surface, color: tuple[int, int, int],
                start: tuple[int, int], end: tuple[int, int], dash: int, gap: int) -> None:
    origin = Vec2d(*start)
    vector = Vec2d(*end) - origin
    direction = vector.normalized()
    for distance in range(0, round(vector.length), dash + gap):
        pygame.draw.line(surface, color, point(origin + direction * distance),
                         point(origin + direction * min(vector.length, distance + dash)), 1)


def make_vignette(size: tuple[int, int], color: tuple[int, int, int]) -> pygame.Surface:
    screen = pygame.Surface(size, pygame.SRCALPHA)
    screen.fill((*color, 137))
    for index in range(32):
        scale = 1.38 - index * 0.024
        oval = pygame.Rect(0, 0, *(round(dimension * scale) for dimension in size))
        oval.center = screen.get_rect().center
        pygame.draw.ellipse(screen, (*color, max(0, 124 - index * 4)), oval)
    return screen


def fan(maze: Maze, origin: Vec2d, middle: float, arc: float, distance: float, samples: int):
    points = [origin] if arc < 2 * pi else []
    for index in range(samples + 1):
        angle = middle - arc / 2 + arc * index / samples
        direction = Vec2d(cos(angle), sin(angle))
        end = origin + distance * direction
        hit = maze.hit_solid(origin, end)
        points.append(end if hit is None else hit.point + direction * 4)
    return points


class MazeView:
    def __init__(self, size: tuple[int, int] = WINDOW_SIZE):
        self.screen_size = size
        self.zoom_index = 3
        self.origin = Vec2d(0, 0)
        self.sprites = Sprites()
        self.disguise_cache = {}
        self.open_chests = {name: self.open_chest_sprite(name) for name in ("chest", "epic_chest")}
        self.reinforced_wall = self.sprites.sprite("wall", (TILE, TILE)).copy()
        self.reinforced_wall.fill((118, 162, 198), special_flags=pygame.BLEND_RGB_MULT)
        panel = pygame.Rect(0, 0, TILE, TILE)
        panel.center = self.reinforced_wall.get_rect().center
        pygame.draw.rect(self.reinforced_wall, (75, 98, 116), panel.inflate(-2, -2), 2)
        for x in (panel.left + 5, panel.right - 6):
            for y in (panel.top + 5, panel.bottom - 6):
                pygame.draw.circle(self.reinforced_wall, (172, 156, 119), (x, y), 2)
        self.corpse_sprites = []
        for kind, angle in (("skeleton", -69), ("zombie", 77)):
            sprite = self.sprites.sprite(kind, (18, 30)).copy()
            sprite.fill((146, 134, 143, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self.corpse_sprites.append(pygame.transform.rotate(sprite, angle))
        self.cached_lights: dict[tuple[Vec2d, float], list[Vec2d]] = {}
        self.vision_origin: Vec2d | None = None
        self.radial_vision: dict[tuple[float, int], list[Vec2d]] = {}
        self.light_revision = -1
        self.level = None
        self.now = 0.0
        self.hit_times = {}
        self.map_expanded = False
        self.resize_view()
        # Cache a radial gradient. A wall-clipped cone masks it during normal play.
        reach = VISION_DISTANCE + 4
        self.cone_falloff = pygame.Surface((reach * 2 + 1, reach * 2 + 1))
        for radius in range(reach, 0, -1):
            brightness = round(253 - 116 * (radius / reach) ** 1.2)
            pygame.draw.circle(self.cone_falloff, (brightness,) * 3, (reach, reach), radius)
        self.vignette = make_vignette(size, (166, 5, 34))
        self.pause_backdrop: pygame.Surface | None = None
        self.font = pygame.font.Font(None, 24)
        self.small = pygame.font.Font(None, 19)
        self.large = pygame.font.Font(None, 52)

    def set_screen_size(self, size: tuple[int, int]) -> None:
        self.screen_size = size
        self.resize_view()
        self.vignette = make_vignette(size, (166, 5, 34))
        if self.pause_backdrop is not None:
            self.pause_backdrop = pygame.transform.scale(self.pause_backdrop, size)

    @property
    def menu_rects(self) -> tuple[pygame.Rect, pygame.Rect]:
        x, y = self.screen_size[0] // 2, self.screen_size[1] // 2 + 181
        return pygame.Rect(x - 212, y, 200, 46), pygame.Rect(x + 12, y, 200, 46)

    @property
    def pause_restart_rect(self) -> pygame.Rect:
        return pygame.Rect(self.screen_size[0] // 2 - 100, self.screen_size[1] // 2 + 235, 200, 40)

    @property
    def hard_mode_rect(self) -> pygame.Rect:
        return self.pause_restart_rect

    def resize_view(self, zoom: float | None = None) -> None:
        zoom = ZOOM_LEVELS[self.zoom_index] if zoom is None else zoom
        field_size = tuple(round(dimension / zoom) for dimension in self.screen_size)
        if field_size == getattr(self, "field_size", None):
            return
        self.field_size = field_size
        self.radial_vision.clear()
        self.scene = pygame.Surface(self.field_size).convert()
        self.visible = pygame.Surface(self.field_size)
        self.light = pygame.Surface(self.field_size)

    def adjust_zoom(self, amount: int) -> None:
        index = min(len(ZOOM_LEVELS) - 1, max(0, self.zoom_index + amount))
        if index != self.zoom_index:
            self.zoom_index = index
            self.resize_view()

    def mouse_offset(self, screen_position: tuple[float, float]) -> Vec2d:
        return Vec2d(*((position / screen_size - 0.5) * world_size
                       for position, screen_size, world_size in zip(screen_position, self.screen_size, self.field_size)))

    def hand_cursor(self, actor: Actor, offset: Vec2d) -> Vec2d:
        # Measure the fraction of the ray to the viewport edge independently of zoom.
        fraction = min(1.0, 2 * max(abs(offset.x) / self.field_size[0],
                                    abs(offset.y) / self.field_size[1]))
        reach = HAND_REACH + actor.radius - 10
        distance = MIN_HAND_REACH + (reach - MIN_HAND_REACH) * fraction
        direction = offset.normalized() if offset.length else actor.aim
        return actor.position + direction * distance

    def at(self, position: Vec2d) -> tuple[int, int]:
        return point(position - self.origin)

    def in_view(self, position: Vec2d, padding: int = 0) -> bool:
        x, y = position - self.origin
        return -padding <= x < self.field_size[0] + padding and -padding <= y < self.field_size[1] + padding

    def place(self, sprite: pygame.Surface, position: Vec2d) -> None:
        self.scene.blit(sprite, sprite.get_rect(center=self.at(position)))

    def animation(self, shape, fallback: str = "idle", age: float | None = None):
        elapsed = self.now - self.hit_times.get(shape, float("-inf"))
        if elapsed < 0.16:
            return "hit", elapsed, False
        return fallback, self.now if age is None else age, age is None

    def view_tiles(self):
        row_start = max(0, int(self.origin.y // TILE))
        row_stop = min(ROWS, int((self.origin.y + self.field_size[1]) // TILE) + 1)
        col_start = max(0, int(self.origin.x // TILE))
        col_stop = min(COLS, int((self.origin.x + self.field_size[0]) // TILE) + 1)
        return ((x, y) for y in range(row_start, row_stop) for x in range(col_start, col_stop))

    def draw_terrain(self, maze: Maze) -> list[pygame.Rect]:
        self.scene.fill((10, 11, 16))
        lit_tiles = []
        doors = set()
        for cell in self.view_tiles():
            x, y = cell
            location = center(cell)
            symbol = maze.layout.tiles[y][x]
            if (wall := maze.walls.get(cell)) is not None:
                if maze.reinforced(cell):
                    self.place(self.reinforced_wall, location)
                    continue
                state, age, loop = self.animation(wall, "damaged" if cell in maze.wall_damage else "idle")
                self.place(self.sprites.sprite("wall", (TILE, TILE), state, age, loop=loop), location)
                if symbol == "X" and maze.gate_exposed_at is not None:
                    elapsed = self.now - maze.gate_exposed_at
                    if elapsed < 0.67:
                        px, py = self.at(location)
                        for fragment in range(5):
                            dx = fragment % 3 * 11 - 15 + round((fragment - 2) * elapsed * 15)
                            dy = fragment // 3 * 13 - 14 + round(elapsed * (36 + 13 * fragment))
                            color = (max(36, 132 - round(124 * elapsed)), 105, 121)
                            pygame.draw.rect(self.scene, color, (px + dx, py + dy, 8, 5))
                continue
            self.place(self.sprites.sprite("floor", (TILE, TILE)), location)
            if cell in maze.carved_walls:
                px, py = self.at(location)
                pygame.draw.polygon(self.scene, (73, 72, 85),
                                    ((px - 12, py + 9), (px - 8, py + 6), (px - 5, py + 11)))
                pygame.draw.polygon(self.scene, (65, 65, 78),
                                    ((px + 7, py - 10), (px + 11, py - 12), (px + 13, py - 7)))
            if symbol in LIT_TILES:
                rect = pygame.Rect(*self.at(Vec2d(x * TILE, y * TILE)), TILE, TILE)
                lit_tiles.append(rect)
            door = maze.doors.get(cell)
            if door is not None and not door.broken:
                doors.add(door)
        for door in doors:
            self.place(self.door_sprite(maze, door), door.position)
        return lit_tiles

    def door_sprite(self, maze: Maze, door: Door) -> pygame.Surface:
        name = "gate" if door is maze.exit_gate else "door"
        width = height = TILE
        if door.wide and door.jamb is not None:
            width *= 1 + (door.cell[0] != door.jamb[0])
            height *= 1 + (door.cell[1] != door.jamb[1])
        if door.is_open:
            return self.sprites.sprite(name, (width, height), "open", self.now - door.opened_at, loop=False)
        state, age, loop = self.animation(door.shape, "damaged" if door.hp < DOOR_HP else "idle")
        return self.sprites.sprite(name, (width, height), state, age, loop=loop)

    def open_chest_sprite(self, name: str) -> pygame.Surface:
        sprite = self.sprites.sprite(name, (25, 20)).copy()
        bounds = sprite.get_bounding_rect()
        mouth = pygame.Rect(bounds.x + 2, bounds.y + 6, bounds.width - 4, 5)
        lid = sprite.subsurface((bounds.x, bounds.y, bounds.width, 9)).copy()
        sprite.fill((0, 0, 0, 0), (bounds.x, bounds.y, bounds.width, 9))
        pygame.draw.rect(sprite, (43, 24, 51) if name == "epic_chest" else (32, 24, 21), mouth)
        sprite.blit(lid, (bounds.x, bounds.y - 4))
        return sprite

    def chest_sprite(self, chest: Chest) -> pygame.Surface:
        name = "epic_chest" if chest.epic else "chest"
        return self.open_chests[name] if chest.opened else self.sprites.sprite(name, (25, 20))

    def apply_visibility(self, maze: Maze, lit_tiles: list[pygame.Rect]) -> None:
        if maze.layout is not self.level or maze.wall_revision != self.light_revision:
            self.cached_lights.clear()
            self.radial_vision.clear()
            self.light_revision = maze.wall_revision
            if maze.layout is not self.level:
                self.disguise_cache.clear()
            self.level = maze.layout
        self.light.fill(0)
        self.visible.fill(0)
        static_keys = set()
        sources = []
        for position, radius, static in maze.lights():
            key = (position, radius)
            if static:
                static_keys.add(key)
            if not self.in_view(position, round(radius)):
                continue
            vertices = self.cached_lights.get(key) if static else None
            if vertices is None:
                vertices = fan(maze, position, 0, 2 * pi, radius, 44)
                if static:
                    self.cached_lights[key] = vertices
            sources.append((position, vertices))
            pygame.draw.polygon(self.light, (175, 152, 125), [self.at(p) for p in vertices])
        for key in self.cached_lights.keys() - static_keys:
            del self.cached_lights[key]
        for position, vertices in sources:
            points = [self.at(position + (vertex - position) * 0.69) for vertex in vertices]
            pygame.draw.polygon(self.light, (240, 221, 193), points)
        for rect in lit_tiles:
            self.light.fill((221, 203, 184), rect)

        player = maze.player
        position = player.position
        if position != self.vision_origin:
            self.vision_origin = position
            self.radial_vision.clear()
        white = (255, 255, 255)
        def reveal(distance: float, samples: int) -> None:
            key = (distance, samples)
            vertices = self.radial_vision.get(key)
            if vertices is None:
                vertices = fan(maze, position, 0, 2 * pi, distance, samples)
                self.radial_vision[key] = vertices
            pygame.draw.polygon(self.visible, white, [self.at(p) for p in vertices])

        if sources or lit_tiles:
            reveal(hypot(*self.field_size), 144)
            self.visible.blit(self.light, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        self.light.fill(0)
        cone = fan(maze, player.position, player.aim.angle, CONE_SPAN, VISION_DISTANCE, 50)
        pygame.draw.polygon(self.light, white, [self.at(vertex) for vertex in cone])
        self.light.blit(self.cone_falloff, self.cone_falloff.get_rect(center=self.at(player.position)),
                        special_flags=pygame.BLEND_RGB_MULT)
        self.visible.blit(self.light, (0, 0), special_flags=pygame.BLEND_RGB_MAX)
        reveal(maze.local_vision_radius, 30)
        self.reveal_door_faces(maze)
        self.scene.blit(self.visible, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

    def reveal_door_faces(self, maze: Maze) -> None:
        # Reveal a closed door's front from the adjoining floor. Mimicked doors use the same rule.
        bounds = self.visible.get_rect()
        fakes = {cell: mimic.wall_face for mimic in maze.mimics if is_disguised(mimic)
                 for cell in mimic.wall_cells}
        for cell in self.view_tiles():
            door = maze.doors.get(cell)
            if cell in fakes:
                faces = (-fakes[cell],)
            elif door is not None and not door.is_open and not door.broken:
                faces = (Vec2d(1, 0), Vec2d(-1, 0), Vec2d(0, 1), Vec2d(0, -1))
            else:
                continue
            location = center(cell)
            color = max((self.visible.get_at(sample)[:3] for direction in faces
                         if bounds.collidepoint(sample := self.at(location + direction * (TILE / 2 + 3)))),
                        default=(0, 0, 0))
            if max(color) > 0:
                tile = pygame.Rect(0, 0, TILE, TILE)
                tile.center = self.at(location)
                self.visible.fill(color, tile)

    def record_visible_terrain(self, maze: Maze) -> None:
        bounds = self.visible.get_rect()
        near_edge = TILE // 2 - 2
        samples = ((0, 0), (-near_edge, 0), (near_edge, 0), (0, -near_edge), (0, near_edge))
        fakes = {cell for mimic in maze.mimics if is_disguised(mimic)
                 and maze.clear_path(maze.player.position, mimic.position) for cell in mimic.wall_cells}
        for cell in self.view_tiles():
            door = maze.doors.get(cell)
            if cell in fakes:
                tile = "D"
            elif door is not None:
                tile = "." if door.is_open else "D"
            else:
                tile = "X" if maze.reinforced(cell) else maze.current_tile(cell)
            if maze.discovered.get(cell) == tile:
                continue
            x, y = self.at(center(cell))
            for dx, dy in samples:
                sample = (x + dx, y + dy)
                if bounds.collidepoint(sample) and self.visible.get_at(sample).r > 0:
                    maze.discovered[cell] = tile
                    break

    def draw_minimap(self, screen: pygame.Surface, maze: Maze) -> None:
        map_width, map_height = (399, 301) if self.map_expanded else (171, 129)
        box = pygame.Rect(0, 0, map_width + 16, map_height + 32)
        box.bottomright = self.screen_size[0] - 12, self.screen_size[1] - 12
        pygame.draw.rect(screen, (18, 17, 24), box, border_radius=5)
        pygame.draw.rect(screen, (88, 76, 84), box, 1, border_radius=5)
        self.label(screen, "Minimap", (box.x + 8, box.y + 5))
        self.label(screen, "M", (box.right - 8, box.y + 5), right=True)
        board = pygame.Rect(box.x + 8, box.y + 24, map_width, map_height)
        map_center = Vec2d(*board.center)
        position = maze.player.position
        if self.map_expanded:
            # Keep both the player centered and every tile inside the larger map.
            scale = min(
                (board.width / 2 - 7) / max(position.x, COLS * TILE - position.x),
                (board.height / 2 - 7) / max(position.y, ROWS * TILE - position.y),
            )
        else:
            scale = 3 / TILE
        origin = map_center - position * scale

        def map_point(location: Vec2d) -> tuple[int, int]:
            return point(origin + location * scale)

        clip = screen.get_clip()
        screen.set_clip(board)
        pygame.draw.rect(screen, (9, 9, 14), board)
        tile_scale = TILE * scale
        for (col, row), tile in maze.discovered.items():
            left = round(origin.x + col * tile_scale)
            top = round(origin.y + row * tile_scale)
            right = round(origin.x + (col + 1) * tile_scale)
            bottom = round(origin.y + (row + 1) * tile_scale)
            if right <= board.left or left >= board.right or bottom <= board.top or top >= board.bottom:
                continue
            color = MAP_TERRAIN_COLORS.get(tile, MAP_LIGHT if tile in LIT_TILES else MAP_FLOOR)
            pygame.draw.rect(screen, color, (left, top, right - left, bottom - top))

        player = board.center
        if maze.radar_visible and maze.radar_target is not None:
            draw_dashes(screen, RADAR_BEARING, player, map_point(maze.radar_target), 4, 3)
            if len(maze.radar_path) > 1:
                path = [map_point(center(cell)) for cell in maze.radar_path]
                pygame.draw.lines(screen, RADAR_ROUTE, False, path, 1 + self.map_expanded)

        gear_dots = [(item.position, item.spec.effect) for item in maze.wielding.items
                     if item.holder is None and item is not maze.mirror]
        gear_dots.extend((mimic.position, mimic.form) for mimic in maze.mimics
                         if is_disguised(mimic) and mimic.form in ("item", "key"))
        for location, effect in gear_dots:
            pos = map_point(location)
            radius = 3 if self.map_expanded else 2
            if effect == "key":
                if not board.inflate(-14, -14).collidepoint(pos):
                    bearing = Vec2d(*pos) - map_center
                    fraction = min(
                        (board.width / 2 - 7) / max(1, abs(bearing.x)),
                        (board.height / 2 - 7) / max(1, abs(bearing.y)),
                    )
                    pos = point(map_center + bearing * fraction)
                pygame.draw.circle(screen, (27, 21, 20), pos, radius + 3)
                pygame.draw.circle(screen, (255, 232, 117), pos, radius + 2, 1)
            pygame.draw.circle(screen, MAP_LOOT, pos, radius)
        pygame.draw.circle(screen, (13, 22, 22), player, 5)
        pygame.draw.circle(screen, (199, 252, 232), player, 3)
        screen.set_clip(clip)

    def actor_sprite(self, actor: Actor, acting: bool = False) -> pygame.Surface:
        height = round(actor.radius * 2.85)
        name = actor.kind
        state = "action" if acting else "walk" if actor.body.velocity.length > 17 else "idle"
        state, age, loop = self.animation(actor.shape, state)
        sprite = self.sprites.sprite(name, (self.sprites.natural_width(name, height), height),
                                     state, age, loop=loop)
        if actor.aim.x < -0.4:
            sprite = pygame.transform.flip(sprite, True, False)
        if abs(actor.tumble_angle) >= 0.04:
            sprite = pygame.transform.rotate(sprite, -degrees(actor.tumble_angle))
        return sprite

    def draw_actor(self, actor: Actor, acting: bool = False) -> None:
        if not actor.active or not self.in_view(actor.position, 48):
            return
        if actor.swing_time > 0 and actor.swing_reach:
            arc = swing_outline(actor.position, actor.swing_direction, actor.swing_reach, SWING_HALF_ANGLE)
            pygame.draw.polygon(self.scene, (203, 109, 99), [self.at(p) for p in arc], 1)
        self.place(self.actor_sprite(actor, acting), actor.position - Vec2d(0, actor.radius * 0.16))

    def draw_mimic(self, maze: Maze, mimic: Enemy) -> None:
        if not mimic.active or not self.in_view(mimic.position, 135):
            return
        if is_disguised(mimic):
            if not mimic.wall_cells or maze.clear_path(maze.player.position, mimic.position):
                if mimic.form == "item" and mimic.disguise_item.effect == "whip":
                    direction = Vec2d(1, 0).rotated(-mimic.tilt * mimic.imperfection * pi / 180)
                    root = mimic.position + direction * (mimic.disguise_item.length / 2)
                    tip = root + direction * 48 + direction.perpendicular() * 6
                    self.draw_whip_tail(root, tip, -direction.perpendicular() * 58, (217, 113, 129))
                self.place(*self.disguise_sprite(mimic))
            return
        phase = self.now * (26 if mimic.desired_velocity.length > 1 else 7)
        for leg in mimic_legs(mimic.legs, phase, mimic.radius * 1.17):
            root, knee, tip = [self.at(mimic.position + Vec2d(*p).rotated(mimic.tumble_angle)) for p in leg]
            pygame.draw.line(self.scene, (93, 55, 55), root, knee, 5)
            pygame.draw.line(self.scene, (155, 96, 82), knee, tip, 2)
            pygame.draw.line(self.scene, (201, 143, 116), root, knee, 2)
        if (tip := tendril_tip(mimic, maze)) is not None:
            span = tip - mimic.position
            curve = Vec2d(-span.y, span.x).normalized() * (10 if mimic.attack.kind == "swing" else 4)
            bend = mimic.position + span * 0.60 + curve
            points = [self.at(position) for position in (mimic.position, bend, tip)]
            pygame.draw.lines(self.scene, (101, 49, 47), False, points, 8)
            pygame.draw.lines(self.scene, (218, 146, 112), False, points, 4)
            pygame.draw.circle(self.scene, (226, 167, 133), points[-1], 4 if mimic.attack.kind == "swing" else 2)
        elif mimic.attack is not None:
            radius = round(mimic.radius + 4)
            pygame.draw.circle(self.scene, (179, 73, 75), self.at(mimic.position), radius, 2)
            start = self.at(mimic.position)
            forward = self.at(mimic.position + mimic.attack.direction * (radius + 12))
            pygame.draw.line(self.scene, (199, 105, 97), start, forward, 3)
        move = "action" if mimic.attack else "walk" if mimic.desired_velocity.length > 1 else "idle"
        state, age, loop = self.animation(mimic.shape, move)
        scale = mimic.radius / MIMIC_BASE_RADIUS
        size = (round(53 * scale), round(47 * scale))
        body = self.sprites.sprite("mimic_body", size, state, age, loop=loop)
        sprite = self.mimic_sprite(body, mimic.mimic_memory.appearance)
        if abs(mimic.tumble_angle) >= 0.04:
            sprite = pygame.transform.rotate(sprite, -degrees(mimic.tumble_angle))
        self.place(sprite, mimic.position - Vec2d(0, 3 * scale))

    def draw_mimic_corpse(self, corpse: MimicCorpse) -> None:
        if not self.in_view(corpse.position, 130):
            return
        swelling = max(0, 1 - (corpse.bursts_at - self.now) / MIMIC_BURST_DELAY)
        pressure = 1 + 0.76 * swelling + 0.12 * swelling * sin(self.now * 19)
        scale = corpse.radius / MIMIC_BASE_RADIUS * (0.54 if corpse.ruptured else pressure)
        size = (round(55 * scale), max(1, round((23 if corpse.ruptured else 44) * scale)))
        body = self.sprites.sprite("mimic_body", size)
        sprite = body.copy() if corpse.ruptured else self.mimic_sprite(body, corpse.appearance).copy()
        sprite.fill((150, 81, 96, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.place(sprite, corpse.position)

    def mimic_sprite(self, body: pygame.Surface, appearance: MimicAppearance) -> pygame.Surface:
        if not appearance.extra_eyes and not appearance.fragments:
            return body
        sprite = body.copy()
        bounds = body.get_bounding_rect(min_alpha=140)
        middle = Vec2d(*bounds.center)

        def skin(index: int, offset: float, inset: float) -> Vec2d:
            angle = offset + index * pi * (3 - sqrt(5))
            radius = inset - min(0.45, 0.14 * (index // 12))
            axis = Vec2d(cos(angle) * bounds.width / 2, sin(angle) * bounds.height / 2)
            for fraction in (1.0, 0.85, 0.7, 0.55):
                spot = middle + axis * (radius * fraction)
                if body.get_at(point(spot)).a >= 140:
                    return spot
            return middle

        patch_size = max(3, round(bounds.width * min(0.2, 0.55 / sqrt(1 + len(appearance.fragments)))))
        for index, spec in enumerate(appearance.fragments):
            texture = self.sprites.fragment(gear_sprite_name(spec))
            dimensions = texture.get_size()
            size = tuple(max(2, round(patch_size * side / max(dimensions))) for side in dimensions)
            patch = pygame.transform.scale(texture, size)
            sprite.blit(patch, patch.get_rect(center=point(skin(index, 0.5, 0.78))))
        eye_width = max(3, round(bounds.width * min(0.14, 0.48 / sqrt(1 + appearance.extra_eyes))))
        for index in range(appearance.extra_eyes):
            spot = skin(index, -0.85, 0.9)
            draw_eye(sprite, spot, eye_width, spot, bloodshot=True)
        return sprite

    def disguise_sprite(self, mimic: Enemy):
        key = (mimic.form, mimic.wall_cells, mimic.disguise_item, mimic.epic_chest, mimic.tint,
               mimic.eye_offset, disguise_eye_width(mimic), mimic.tilt, mimic.imperfection)
        if (cached := self.disguise_cache.get(mimic)) is not None and cached[0] == key:
            return cached[1], disguise_position(mimic)
        if mimic.wall_cells:
            first, last = mimic.wall_cells[0], mimic.wall_cells[-1]
            name = mimic.form
            width, height = ((1 + abs(first[axis] - last[axis])) * TILE for axis in range(2))
        elif mimic.form in ("item", "key"):
            spec = mimic.disguise_item
            name, width, height = gear_sprite_name(spec), round(spec.length), round(spec.width)
        else:
            name, (width, height) = MIMIC_DISGUISES[mimic.form]
            if mimic.epic_chest:
                name = "epic_chest"
        sprite = self.sprites.sprite(name, (width + mimic.imperfection // 2, height)).copy()
        sprite.fill((*mimic.tint, 255), special_flags=pygame.BLEND_RGBA_MULT)
        bounds = sprite.get_bounding_rect(min_alpha=140)
        spot = Vec2d(bounds.centerx + mimic.eye_offset.x * bounds.width * 0.28,
                     bounds.centery + mimic.eye_offset.y * bounds.height * 0.32)
        eye = min((Vec2d(x, y) for y in range(bounds.top, bounds.bottom) for x in range(bounds.left, bounds.right)
                   if sprite.get_at((x, y)).a >= 140), key=spot.get_dist_sqrd, default=spot)
        draw_eye(sprite, eye, disguise_eye_width(mimic), eye, bloodshot=True)
        if not mimic.wall_cells:
            sprite = pygame.transform.rotate(sprite, mimic.tilt * mimic.imperfection)
        self.disguise_cache[mimic] = key, sprite
        return sprite, disguise_position(mimic)

    def gear_sprite(self, item: Gear) -> pygame.Surface:
        name = gear_sprite_name(item.spec)
        damaged = item.durability is not None and item.durability <= item.spec.durability * 0.55
        activation = self.now - item.activated_at
        state, age, loop = self.animation(item.shape, "action" if activation < 0.27
                                          else "damaged" if damaged else "idle",
                                          activation if activation < 0.27 else None)
        if item.jaw_closure > 0:
            state, age, loop = "clamp", item.jaw_closure, False
        sprite = self.sprites.sprite(name, (round(item.spec.length), round(item.spec.width)), state, age,
                                     loop=loop)
        return pygame.transform.rotate(sprite, -degrees(item.body.angle))

    def draw_fire_and_shots(self, maze: Maze) -> None:
        for fire in maze.fires.values():
            if self.in_view(fire.position, 20):
                self.place(self.sprites.sprite("fire", (15, 19), "idle", self.now), fire.position)
        for well in maze.gravity_wells:
            if self.in_view(well.position, 36):
                size = round(8 + well.radius * 0.078 + 4 * sin(self.now * 19))
                pos = self.at(well.position)
                pygame.draw.circle(self.scene, (135, 94, 223), pos, size, 1)
                pygame.draw.circle(self.scene, (208, 173, 252), pos, 5)
                pygame.draw.circle(self.scene, (39, 22, 64), pos, 3)
        for shot in maze.projectiles:
            spike = shot.spike
            if ((spike is not None and spike.hp <= 0)
                    or not self.in_view(shot.position, round(spike.length) if spike is not None else 14)):
                continue
            if spike is not None:
                tip = self.at(shot.position)
                tail = self.at(shot.position - shot.direction * spike.length)
                tint = (245, 189, 137) if spike.piercing else (232, 225, 207)
                alpha = min(1.0, max(0.0, (spike.expires - self.now) / 0.32))
                color = tuple(round(channel * alpha) for channel in tint)
                pygame.draw.line(self.scene, color, tail, tip, width=3 if spike.piercing else 1)
                if spike.piercing:
                    point_color = tuple(round(channel * alpha) for channel in (254, 236, 206))
                    pygame.draw.line(self.scene, point_color, self.at(shot.position - shot.direction * 8), tip, 1)
            elif shot.effect == "pulse":
                ring = [
                    self.at(shot.position + Vec2d(3 * cos(i * pi / 8), 13 * sin(i * pi / 8))
                            .rotated(shot.direction.angle))
                    for i in range(16)
                ]
                pygame.draw.lines(self.scene, (141, 236, 248), True, ring, 2)
            elif shot.effect == "molotov":
                position = self.at(shot.position)
                trail = self.at(shot.position - shot.direction * 19)
                pygame.draw.line(self.scene, (220, 82, 34), trail, position, 3)
                pygame.draw.circle(self.scene, (114, 157, 105), position, 4)
                pygame.draw.circle(self.scene, (255, 194, 94), position, 2)
            elif shot.target is not None:
                pos = self.at(shot.position)
                pygame.draw.line(self.scene, (246, 176, 97),
                                 self.at(shot.position - shot.direction * 23), pos, 3)
                pygame.draw.circle(self.scene, (255, 230, 159), pos, 3)
            else:
                color = (204, 161, 246) if shot.effect == "gravity" else GEAR_COLORS[shot.mode]
                pygame.draw.circle(self.scene, color, self.at(shot.position), round(max(3, shot.radius)))
        for beam in maze.beams:
            start, end = self.at(beam.start), self.at(beam.end)
            glow = (67, 141, 227) if beam.slashing else (175, 33, 68)
            core = (163, 218, 255) if beam.slashing else (255, 183, 166)
            pygame.draw.line(self.scene, glow, start, end, beam.width)
            pygame.draw.line(self.scene, core, start, end, max(2, beam.width // 2))
        for start, end, expires in maze.flame_trails:
            age = max(0.0, (expires - self.now) / 0.14)
            color = (252, round(80 + age * 91), 41)
            for index in range(1, 8):
                portion = (index - 0.50 * age) / 8
                position = start.interpolate_to(end, portion)
                radius = max(1, round((2 + 5 * portion) * (0.50 + 0.50 * age)))
                pygame.draw.circle(self.scene, color, self.at(position), radius)
        for actor, burn in maze.burns.items():
            if not actor.active or self.now >= burn.expires or not self.in_view(actor.position, 24):
                continue
            for index in range(5):
                angle = 2 * pi * index / 5 + self.now * 2.7
                spark = actor.position + Vec2d(actor.radius * 0.76, 0).rotated(angle)
                spark -= Vec2d(0, (self.now * 22 + index * 3) % 15)
                pygame.draw.circle(self.scene, (244, 128 + index * 14, 59), self.at(spark), 2)
        for flash in maze.weapon_flashes:
            age = min(1.0, (self.now - flash.started) / 0.34)
            radius = flash.radius * (0.35 + 0.65 * min(1, 2.8 * age))
            color = tuple(round(channel * (1 - 0.63 * age)) for channel in flash.color)
            if flash.direction is None:
                pygame.draw.circle(self.scene, color, self.at(flash.position), round(radius), 2)
            else:
                points = [
                    self.at(flash.position + flash.direction.rotated(flash.half_angle * (i / 12 - 1)) * radius)
                    for i in range(25)
                ]
                pygame.draw.lines(self.scene, color, False, points, max(1, round(5 * (1 - age))))

    def draw_drains(self, maze: Maze) -> None:
        for drain in maze.item_actions.drains:
            if not self.in_view(drain.position, DRAIN_RADIUS):
                continue
            pos = self.at(drain.position)
            radius = round(DRAIN_RADIUS - 2 + 2 * sin(self.now * 9))
            pygame.draw.circle(self.scene, (19, 8, 27), pos, radius)
            pygame.draw.circle(self.scene, (75, 31, 100), pos, radius, 2)
            pygame.draw.circle(self.scene, (47, 18, 66), pos, radius - 6, 2)
            for index in range(7):
                angle = index * 2 * pi / 7 - self.now * 1.7
                vertex = drain.position + Vec2d(23 + 7 * sin(index + self.now * 7), 0).rotated(angle)
                pygame.draw.circle(self.scene, EPIC_PURPLE, self.at(vertex), 1)

    def draw_epic_gear(self, maze: Maze) -> None:
        actions = maze.item_actions
        for item, whip in actions.whips.items():
            if item in actions.stowed or not self.in_view(item.position, 132):
                continue
            for tail in whip.tails:
                points = [self.at(point) for point in tail.points]
                pygame.draw.lines(self.scene, (50, 30, 67), False, points, 3)
                pygame.draw.lines(self.scene, (152, 97, 154), False, points)
                pygame.draw.circle(self.scene, EPIC_PURPLE, points[-1], 3)
                pygame.draw.circle(self.scene, (237, 201, 255), points[-1], 1)
        for item in actions.loads:
            if item in actions.stowed:
                continue
            muzzle = self.at(item.point_on_axis(1))
            pygame.draw.circle(self.scene, EPIC_PURPLE, muzzle, round(6 + 2 * sin(self.now * 10)), 2)
        for marks in actions.marks.values():
            for mark in marks:
                position = self.at(mark)
                pygame.draw.circle(self.scene, EPIC_PURPLE, position, 6, 1)
                pygame.draw.circle(self.scene, (235, 196, 255), position, 2)

    def draw_seals(self, maze: Maze) -> None:
        for seal in maze.item_actions.seals:
            points = [self.at(vertex) for vertex in seal.outline]
            if len(points) >= 2:
                pygame.draw.lines(self.scene, (81, 30, 106), len(points) > 2, points, 5)
                pygame.draw.lines(self.scene, (225, 171, 255), len(points) > 2, points, 2)
            age = self.now - seal.started
            for vertex in points:
                pygame.draw.circle(self.scene, EPIC_PURPLE, vertex, 4)
                pygame.draw.circle(self.scene, (249, 209, 255), vertex, 2)
                if age < 0.48:
                    pygame.draw.circle(self.scene, EPIC_PURPLE, vertex, 4 + round(19 * age), 1)

    def draw_whip_tail(self, root: Vec2d, tip: Vec2d, velocity: Vec2d, tint: tuple[int, int, int]) -> None:
        offset = tip - root
        axis = offset.normalized()
        bend = velocity - axis * velocity.dot(axis)
        if bend.length > 70:
            bend = bend.normalized() * 70
        elbow = root + offset * 0.45 - bend * 0.14
        points = [self.at(root * (1 - t) ** 2 + elbow * (2 * t * (1 - t)) + tip * t * t)
                  for t in (0, 0.25, 0.5, 0.75, 1)]
        pygame.draw.lines(self.scene, (49, 26, 41), False, points, 3)
        pygame.draw.lines(self.scene, (173, 110, 116), False, points)
        pygame.draw.circle(self.scene, tint, points[-1], 2)
        pygame.draw.circle(self.scene, (234, 189, 224), (points[-1][0], points[-1][1] - 1), 1)

    def draw_aim(self, maze: Maze) -> None:
        actor = maze.player
        if (held := maze.wielding.held.get(actor)) is None:
            return
        item = held.item
        reach = HAND_REACH + actor.radius - 10
        direction = held.rush.direction if held.rush is not None else actor.aim
        points = [self.at(actor.position)]
        points.extend(self.at(actor.position + direction.rotated(CONE_SPAN * (i / 10 - 0.5)) * reach)
                      for i in range(11))
        pygame.draw.lines(self.scene, (103, 133, 136), True, points, 1)
        if item.spec.mode in ("ranged", "magic") and held.lock_target is None:
            muzzle = item.point_on_axis(1)
            solid = maze.hit_solid(muzzle, muzzle + item.axis * 92)
            distance = 92 if solid is None else muzzle.get_distance(solid.point)
            for offset in range(11, int(distance), 12):
                pos = self.at(muzzle + item.axis * offset)
                pygame.draw.circle(self.scene, (210, 206, 174) if offset < 45 else (130, 119, 112), pos, 1)

    def draw_target_marks(self, maze: Maze) -> None:
        held = maze.item_actions.held_by(maze.player)
        if held is None:
            return
        if (destination := held.sword_destination) is not None:
            color = (116, 186, 244)
            x, y = self.at(destination)
            draw_dashes(self.scene, color, self.at(maze.player.position), (x, y), 6, 5)
            pygame.draw.circle(self.scene, color, (x, y), 8, 1)
            pygame.draw.line(self.scene, color, (x - 11, y), (x + 11, y))
            pygame.draw.line(self.scene, color, (x, y - 11), (x, y + 11))
        target = held.lock_target
        if target is not None and target.alive and self.in_view(target.position, 12) and maze.player_sees(target.position):
            x, y = self.at(target.position)
            radius = round(target.radius + 6)
            pygame.draw.circle(self.scene, (242, 183, 103), (x, y), radius, 1)
            pygame.draw.circle(self.scene, (255, 221, 134), (x, y - radius), 3)

    def draw_grip_tension(self, maze: Maze) -> None:
        held = maze.wielding.held.get(maze.player)
        if held is None or held.item.contender is None:
            return
        for actor, color in ((held.item.holder, (232, 233, 193)), (held.item.contender, (214, 138, 92))):
            grip = maze.wielding.held[actor].grip
            pygame.draw.line(self.scene, color, self.at(actor.position), self.at(grip), 1)

    def show_glow(self, sprite: pygame.Surface, position: Vec2d) -> None:
        location = sprite.get_rect(center=self.at(position))
        mask = pygame.mask.from_surface(sprite)
        glow = mask.to_surface(setcolor=(240, 240, 230, 255), unsetcolor=(0, 0, 0, 0))
        glow.set_alpha(round(51 + sin(self.now * 4) * 10))
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            self.scene.blit(glow, location.move(dx, dy))
        self.scene.blit(sprite, location)

    def draw_interaction(self, maze: Maze, cursor: Vec2d) -> None:
        if not maze.player.active or maze.shop is not None:
            return
        target = maze.nearby_target(maze.player) if not maze.near_mirror else None
        if maze.near_mirror:
            self.show_glow(self.gear_sprite(maze.mirror), maze.mirror.position)
        elif isinstance(target, Door):
            self.show_glow(self.door_sprite(maze, target), target.position)
        elif isinstance(target, Chest):
            self.show_glow(self.chest_sprite(target), target.position)
        elif isinstance(target, Merchant):
            self.show_glow(self.actor_sprite(target.actor), target.position - Vec2d(0, target.actor.radius * 0.16))
        elif isinstance(target, Enemy) and target.kind == "mimic":
            self.show_glow(*self.disguise_sprite(target))
        if (held := maze.wielding.held.get(maze.player)) is not None:
            grip = self.at(held.grip)
            pygame.draw.circle(self.scene, (40, 51, 55), grip, 3)
            pygame.draw.circle(self.scene, (229, 249, 239), grip, 2, 1)
        elif (mimic := maze.disguised_gear_at(cursor)) is not None:
            self.show_glow(*self.disguise_sprite(mimic))
        elif (candidate := maze.grab_candidate(maze.player, cursor)) is not None:
            item, grip = candidate
            self.show_glow(self.gear_sprite(item), item.position)
            color = (244, 102, 106) if item.sharp_at(grip) else (255, 253, 238)
            pygame.draw.circle(self.scene, color, self.at(grip), 2)

    def draw_feedback(self, maze: Maze) -> None:
        colors = {"flesh": (236, 112, 105), "stone": (238, 211, 168), "metal": (248, 231, 197)}
        for hit in maze.hits:
            position = self.at(hit.position)
            if not self.visible.get_rect().collidepoint(position):
                continue
            if max(self.visible.get_at(position)[:3]) < 22:
                continue
            age = self.now - hit.started
            if not 0 <= age < 0.36:
                continue
            intensity = 1 - age / 0.36
            color = tuple(round(value * intensity) for value in colors[hit.kind])
            for i in range(4 + min(hit.damage, 4)):
                angle = i * 2.399 + hit.position.x * 0.021
                delta = Vec2d(cos(angle), sin(angle)) * (4 + age * (26 + 3 * i))
                pygame.draw.circle(self.scene, color, self.at(hit.position + delta), 1 + (age < 0.12))
        for guard in maze.guards:
            actor = guard
            if not actor.active or not self.in_view(actor.position, 28) or not maze.player_sees(actor.position):
                continue
            x, y = self.at(actor.position)
            if guard.boss:
                rect = pygame.Rect(x - 13, y + round(actor.radius * 1.55), 26, 3)
                pygame.draw.rect(self.scene, (51, 30, 41), rect)
                pygame.draw.rect(self.scene, (213, 93, 104), (rect.x, rect.y, round(26 * actor.hp / actor.max_hp), 3))
            alert = guard.alert
            if not guard.following_eye_tip and alert.role is None and alert.yellow > 0:
                rect = pygame.Rect(x - 14, y - round(actor.radius * 1.40) - 8, 28, 15)
                pygame.draw.arc(self.scene, (250, 204, 93), rect, pi * (1 - alert.yellow), pi, 2)
                if alert.red > 0:
                    pygame.draw.arc(self.scene, (235, 70, 74), rect, pi * (1 - alert.red), pi, 2)

    def draw_mirror_eye(self, maze: Maze) -> None:
        elapsed = maze.victory.elapsed
        mirror = maze.mirror
        openness = min(1.0, elapsed / 0.22, (MIRROR_BLINK_AT - elapsed) / 0.16)
        reflection = pygame.Surface((38, 20), pygame.SRCALPHA)
        middle = Vec2d(19, 10)
        gaze = (maze.player.position - mirror.position).rotated(-mirror.body.angle)
        draw_eye(reflection, middle, 29, middle + gaze, bloodshot=True)
        reflection.set_alpha(min(255, round(90 + elapsed * 650)))
        shape = pygame.transform.scale(reflection, (38, max(2, round(20 * openness))))
        self.place(pygame.transform.rotate(shape, -degrees(mirror.body.angle)), mirror.position)

    def draw_death_body(self, maze: Maze, death: Death) -> None:
        elapsed = death.elapsed
        sprite = self.actor_sprite(maze.player)
        if death.delivery == "magic":
            ash = min(1, elapsed / 0.9)
            if ash < 1:
                faded = sprite.copy()
                faded.set_alpha(round(255 * (1 - ash)))
                self.place(faded, death.position)
            for index in range(29):
                spread = Vec2d(1, 0).rotated(index * 2.399963)
                speed = 11 + (index * 37) % 78
                where = death.position + spread * (2 + (index % 6) + elapsed * speed)
                where += death.direction * (elapsed * 24)
                opacity = max(0, 1 - elapsed / (1.14 + (index % 6) * 0.085))
                if opacity > 0:
                    rgb = (round(190 * opacity), round(158 * opacity), round(242 * opacity))
                    pygame.draw.rect(self.scene, rgb, (*self.at(where), 2 + index % 3, 2))
            return
        fall = min(1.0, elapsed / (0.28 if death.delivery == "projectile" else 0.49))
        ease = 1 - (1 - fall) ** 3
        spin = 90 if death.delivery == "projectile" else 450
        sign = 1 if death.direction.x >= 0 else -1
        self.place(pygame.transform.rotate(sprite, -sign * spin * ease), death.position)
        if death.delivery == "melee" and elapsed <= 0.85:
            for index in range(4):
                offset = -death.direction * (4 + index * 5) + Vec2d(0, index % 3 - 1)
                pygame.draw.circle(self.scene, (105, 93, 89), self.at(death.position + offset), 1 + index % 2)

    def draw_fatal_hit(self, screen: pygame.Surface, death: Death) -> None:
        center_x = self.screen_size[0] // 2
        box = pygame.Rect(0, 0, 390, 67)
        box.midbottom = center_x, self.screen_size[1] - 24
        pygame.draw.rect(screen, (29, 18, 26), box, border_radius=6)
        pygame.draw.rect(screen, (112, 65, 72), box, width=1, border_radius=6)
        weapon = self.font.render(death.weapon, True, (251, 199, 182))
        screen.blit(weapon, weapon.get_rect(center=(center_x, box.y + 24)))
        description = {"magic": "Magic", "projectile": "Projectile", "melee": "Impact"}[death.delivery]
        if death.attacker is not None and death.attacker != death.weapon:
            description += f"  ·  {death.attacker}"
        caption = self.small.render(description, True, (209, 179, 175))
        screen.blit(caption, caption.get_rect(center=(center_x, box.y + 46)))
        if death.attacker_position is not None and self.in_view(death.attacker_position, 0):
            vector = death.attacker_position - self.origin
            x = round(vector.x * self.screen_size[0] / self.field_size[0])
            y = round(vector.y * self.screen_size[1] / self.field_size[1])
            pygame.draw.circle(screen, (221, 122, 128), (x, y), 24, 1)

    def frame_death(self, player_position: Vec2d, death: Death) -> None:
        points = [player_position, death.position]
        if death.attacker_position is not None:
            points.append(death.attacker_position)
        left, right = min(p.x for p in points), max(p.x for p in points)
        top, bottom = min(p.y for p in points), max(p.y for p in points)
        width, height = self.screen_size
        zoom = min(ZOOM_LEVELS[self.zoom_index], width / (right - left + 4 * TILE),
                   max(1, height - 100) / (bottom - top + 4 * TILE))
        self.resize_view(zoom)
        focus = Vec2d((left + right) / 2, (top + bottom) / 2 + 45 / zoom)
        self.origin = focus - Vec2d(*self.field_size) / 2

    def draw(self, screen: pygame.Surface, maze: Maze, cursor: Vec2d | None = None,
             *, screen_pointer: tuple[float, float] | None = None) -> None:
        phase = maze.victory.phase if maze.victory is not None else None
        gaze = Vec2d(*(screen_pointer if screen_pointer is not None else pygame.mouse.get_pos()))
        death = maze.death
        if phase == "blackout":
            screen.fill(0)
            return
        if phase == "eye" or death is not None and death.elapsed >= DEATH_ANIMATION_SECONDS:
            draw_eye_report(self, screen, maze, gaze, defeated=death is not None)
            return
        if phase == "results":
            screen.fill((13, 11, 18))
            draw_end(self, screen, maze, gaze)
            return
        if death is None:
            self.resize_view()
            self.origin = maze.player.position - Vec2d(*self.field_size) / 2
        else:
            self.frame_death(maze.player.position, death)
        self.now = maze.wielding.time
        self.hit_times = {hit.shape: hit.started for hit in maze.hits}
        lit_tiles = self.draw_terrain(maze)
        for x, y in maze.corpses:
            position = center((x, y))
            if self.in_view(position, 20):
                self.place(self.corpse_sprites[(x + y) % 2], position + Vec2d(0, 8))
        for chest in maze.chests:
            if self.in_view(chest.position, 18):
                self.place(self.chest_sprite(chest), chest.position)
        for orb in maze.xp_drops:
            if self.in_view(orb.position, 6):
                at = self.at(orb.position)
                pygame.draw.circle(self.scene, (74, 39, 113), at, 4 + orb.amount)
                pygame.draw.circle(self.scene, EPIC_PURPLE, at, 2 + orb.amount)
        self.draw_drains(maze)
        for eye in maze.eyes:
            if eye.alive and self.in_view(eye.position, 20):
                draw_eye(self.scene, eye.position - self.origin, 26, maze.eye_gaze(eye.position) - self.origin,
                         bloodshot=maze.eye_angry)
                if eye.hp < EYE_HP:
                    x, y = self.at(eye.position)
                    pygame.draw.rect(self.scene, (57, 23, 35), (x - 11, y + 11, 22, 2))
                    pygame.draw.rect(self.scene, (216, 92, 124), (x - 11, y + 11, 22 * eye.hp / EYE_HP, 2))
        for guard in maze.guards:
            self.draw_actor(guard, guard.windup > 0)
            if guard.active and guard.windup > 0 and self.in_view(guard.position, 24):
                mark = self.at(guard.position + guard.aim * (guard.radius + 7))
                color = (255, 169, 105) if guard.attack_special else (204, 156, 117)
                pygame.draw.circle(self.scene, color, mark, 3 if guard.attack_special else 2)
        for scribe in maze.scribes:
            self.draw_actor(scribe, scribe.writing > 0 and not scribe.uploaded)
        for merchant in maze.merchants:
            if merchant.actor not in maze.guards:
                self.draw_actor(merchant.actor)
        for target in maze.targets:
            self.draw_actor(target)
        for corpse in maze.mimic_corpses:
            self.draw_mimic_corpse(corpse)
        for mimic in maze.mimics:
            self.draw_mimic(maze, mimic)
        self.draw_actor(maze.player)
        self.draw_aim(maze)
        self.draw_grip_tension(maze)
        self.draw_epic_gear(maze)
        for item in maze.wielding.items:
            if self.in_view(item.position, 40):
                sprite = self.gear_sprite(item)
                if maze.wielding.active_swing(item):
                    self.show_glow(sprite, item.position)
                else:
                    self.place(sprite, item.position)
                if item.holder is not None:
                    held = maze.wielding.held[item.holder]
                    if held.charge > 0:
                        color = (192, 135, 239) if held.gravity_charging else (230, 209, 150)
                        pygame.draw.circle(self.scene, color, self.at(item.point_on_axis(1)),
                                           round(3 + held.charge * 7), 1)
        self.draw_seals(maze)
        self.draw_fire_and_shots(maze)
        if death is None:
            self.apply_visibility(maze, lit_tiles)
            self.record_visible_terrain(maze)
        else:
            self.visible.fill((255, 255, 255))
        if death is None and maze.radar_visible and maze.radar_target is not None:
            draw_dashes(self.scene, RADAR_BEARING, self.at(maze.player.position),
                        self.at(maze.radar_target), 9, 6)
            if len(maze.radar_path) > 1:
                path = [self.at(center(cell)) for cell in maze.radar_path]
                pygame.draw.lines(self.scene, RADAR_BEARING, False, path, 4)
                pygame.draw.lines(self.scene, RADAR_ROUTE, False, path, 2)
        self.draw_feedback(maze)
        if maze.player.alive:
            self.draw_target_marks(maze)
        if not maze.finished and cursor is not None:
            self.draw_interaction(maze, cursor)
        if phase == "mirror":
            self.draw_mirror_eye(maze)
        if death is not None:
            self.draw_death_body(maze, death)
        pygame.transform.scale(self.scene, self.screen_size, screen)
        if death is not None:
            self.draw_fatal_hit(screen, death)
            return
        if maze.eye_angry:
            screen.blit(self.vignette, (0, 0))
        self.draw_hud(screen, maze)
        self.draw_minimap(screen, maze)
        if maze.finished and phase is None:
            draw_end(self, screen, maze)
        elif maze.shop is not None:
            draw_merchant_shop(self, screen, maze)
        if maze.debug is not None and maze.debug.is_open:
            draw_debug_menu(self, screen, maze, gaze)

    def label(self, screen: pygame.Surface, text: str, position: tuple[int, int],
              *, right: bool = False, color: tuple[int, int, int] = (245, 238, 226)) -> None:
        label = self.small.render(text, True, color)
        shadow = self.small.render(text, True, (17, 15, 23))
        rect = label.get_rect(topright=position) if right else label.get_rect(topleft=position)
        screen.blit(shadow, rect.move(1, 1))
        screen.blit(label, rect)

    def capture_game(self, screen: pygame.Surface) -> None:
        self.pause_backdrop = screen.copy()

    def draw_hud(self, screen: pygame.Surface, maze: Maze) -> None:
        player = maze.player
        for index, (name, value, maximum, color) in enumerate((
            ("Health", player.hp, player.max_hp, (216, 99, 110)),
            ("Stamina", player.stamina, player.max_stamina, (109, 208, 174)),
        )):
            y = 12 + index * 27
            self.label(screen, f"{name} {round(value)}", (13, y))
            rect = pygame.Rect(13, y + 15, 101, 5)
            pygame.draw.rect(screen, (45, 39, 52), rect, border_radius=2)
            pygame.draw.rect(screen, color, (rect.x, rect.y, round(rect.width * value / maximum), 5), border_radius=2)
        eye = Vec2d(27, 78)
        if maze.watched:
            draw_eye(screen, eye, 24, eye + Vec2d(4, -2), bloodshot=maze.eye_angry)
        else:
            closed_eye(screen, eye, 24, (219, 146, 151))
        elapsed = int(maze.wielding.time)
        status = f"Sandbox  ·  XP {maze.progress.xp}" if maze.sandbox else (
            f"Level {maze.progress.level}   {elapsed // 60}:{elapsed % 60:02d}   XP {maze.progress.xp}"
        )
        if maze.progress.hard_mode:
            status = f"Hard Mode   ·   {status}"
        self.label(screen, status, (self.screen_size[0] - 14, 13), right=True)
        if player.captured:
            self.label(screen, "Captured by All-launcher", (14, self.screen_size[1] - 26), color=EPIC_PURPLE)
        if (held := maze.wielding.held.get(player)) is not None:
            item = held.item
            y = self.screen_size[1] - 26
            labels = [item.spec.name]
            if item.durability is not None:
                labels.append(f"{item.durability}/{item.spec.durability}")
            if item.contender is not None:
                labels.append("Contested")
            self.label(screen, "  ·  ".join(labels), (14, y))
            self.draw_item_status(screen, maze, held, y)
        if maze.escape_time > 0 and not maze.escaped:
            progress = min(1, maze.escape_time / MIRROR_ESCAPE_SECONDS)
            x, y = self.screen_size[0] // 2 - 45, self.screen_size[1] // 2 + 32
            self.label(screen, f"Mirror {ceil(MIRROR_ESCAPE_SECONDS - maze.escape_time)}s", (x, y - 17))
            pygame.draw.rect(screen, (44, 34, 46), (x, y, 90, 5))
            pygame.draw.rect(screen, (224, 209, 161), (x, y, round(90 * progress), 5))

    def draw_item_status(self, screen: pygame.Surface, maze: Maze, held: HeldGear, y: int) -> None:
        item, actions = held.item, maze.item_actions
        if item.holder is not maze.player:
            return
        effect = item.spec.effect
        remaining = max(0, item.special_ready - self.now)
        label = f"{remaining:.1f}s" if remaining else ""
        charge = (held.primary_down and item.spec.draw_time > 0) or held.gravity_charging or held.bow_lance_charging
        meter = held.charge if charge else None
        if effect == "grimoire":
            label = f"{remaining:.1f}s" if remaining else f"{actions.book_streaks.get(item, 0)}/{GRIMOIRE_PAGES}"
            if remaining:
                meter = max(0, 1 - remaining / GRIMOIRE_COOLDOWN)
        elif effect == "seal":
            if actions.seals:
                label = f"{max(0, actions.seals[0].expires - self.now):.1f}s"
            else:
                label = f"{len(actions.marks.get(item, ()))}/{SEAL_LIMIT}"
        elif effect == "whip":
            whip = actions.whips.get(item)
            label = f"{whip.count if whip else 1}/9 tails"
        elif effect == "launcher" and (load := actions.loads.get(item)) is not None:
            target = load.captive
            name = target.spec.name if isinstance(target, Gear) else ATTACKER_NAMES.get(target.kind, target.kind.title())
            label = f"{name} · {max(0, ceil(load.expires - self.now))}s"
        elif held.lock_target is not None:
            label = "Locked"
        elif held.sword_destination is not None:
            label = "Marked"
        if label:
            self.label(screen, label, (14, y - 19))
        if meter is not None:
            pygame.draw.rect(screen, (54, 43, 64), (14, y - 27, 103, 4))
            pygame.draw.rect(screen, (211, 181, 224), (14, y - 27, round(103 * meter), 4))

