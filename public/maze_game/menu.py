"""Pause, upgrades, De Bug, merchant, and Eye screens."""

from typing import TYPE_CHECKING

from math import ceil
import pygame
from .physics import Vec2d
from .view_primitives import closed_eye, draw_eye, gear_sprite_name, point, scribe_eye_position
from .item import Equipment, HEAL_PACK, healing_pack_hp
from .progression import FLOOR_HEAL_FRACTION, MERCHANT_PRICES, SHOP_ITEMS, STAT_GAIN, UPGRADES
from .item_debug import EFFECT_PAGE_SIZE, GOLD, SUMMON_ITEMS, SUMMON_MOBS, SandboxTools
from .item_debug_effects import EFFECT_OPTIONS
from .settings import TITLE

if TYPE_CHECKING:
    from .view import MazeView
    from .world import Maze

PAUSE_CONTROLS = (
    ("WASD · Shift · Space", "Move · Sprint · Dodge"),
    ("E · Hold E", "Grab / Interact · Mirror"),
    ("Left · Right click", "Primary · Special"),
    ("Q · Hold F + drag", "Drop · Rotate"),
    ("M · Wheel / + −", "Map size · Zoom"),
    ("Merchant · 1–5 / E", "Buy · Close"),
    ("De Bug · Left / Right / Both", "Items · Mobs · Effects"),
    ("Enter · Esc", "Continue · Pause"),
)

PANEL = (30, 24, 36)
EDGE = (89, 61, 81)
TITLE_COLOR = (248, 221, 200)
MUTED = (184, 166, 181)
ACTION = (171, 219, 201)


def draw_panel(screen: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(screen, PANEL, rect, border_radius=12)
    pygame.draw.rect(screen, EDGE, rect, width=1, border_radius=12)


def menu_buttons(view: "MazeView", screen: pygame.Surface, first: str, second: str,
                 pointer: tuple[float, float]) -> None:
    for rect, text in zip(view.menu_rects, (first, second)):
        hovered = rect.collidepoint(pointer)
        pygame.draw.rect(screen, (64, 44, 53) if hovered else (36, 29, 40), rect, border_radius=5)
        pygame.draw.rect(screen, (173, 119, 106) if hovered else (112, 82, 86),
                         rect, width=1, border_radius=5)
        label = view.font.render(text, True, (241, 223, 210))
        screen.blit(label, label.get_rect(center=rect.center))


def draw_pause(view: "MazeView", screen: pygame.Surface, pointer: tuple[float, float]) -> None:
    if view.pause_backdrop is not None:
        screen.blit(view.pause_backdrop, (0, 0))
    shade = pygame.Surface(view.screen_size, pygame.SRCALPHA)
    shade.fill((11, 8, 14, 178))
    screen.blit(shade, (0, 0))
    panel = pygame.Rect(0, 0, 612, 584)
    panel.center = (view.screen_size[0] // 2, view.screen_size[1] // 2)
    draw_panel(screen, panel)
    title = view.large.render("Paused", True, TITLE_COLOR)
    screen.blit(title, title.get_rect(center=(panel.centerx, panel.y + 44)))
    for index, (keys, action) in enumerate(PAUSE_CONTROLS):
        y = panel.y + 94 + index * 37
        row = pygame.Rect(panel.x + 34, y, panel.width - 68, 31)
        pygame.draw.rect(screen, (40, 31, 47), row, border_radius=4)
        screen.blit(view.small.render(keys, True, MUTED), (row.x + 10, row.y + 8))
        screen.blit(view.font.render(action, True, TITLE_COLOR), (row.x + 258, row.y + 5))
    menu_buttons(view, screen, "Resume", "Main menu", pointer)
    restart = view.pause_restart_rect
    hovered = restart.collidepoint(pointer)
    pygame.draw.rect(screen, (77, 43, 48) if hovered else (45, 30, 36), restart, border_radius=5)
    label = view.font.render("Restart run", True, (239, 190, 175))
    screen.blit(label, label.get_rect(center=restart.center))


def draw_main_menu(view: "MazeView", screen: pygame.Surface, pointer: tuple[float, float]) -> None:
    screen.fill((16, 12, 20))
    midpoint = Vec2d(view.screen_size[0] / 2, view.screen_size[1] / 2 - 100)
    frame = pygame.Rect(0, 0, 412, 210)
    frame.center = point(midpoint)
    pygame.draw.ellipse(screen, EDGE, frame, 1)
    pygame.draw.ellipse(screen, (45, 30, 44), frame.inflate(-44, -26), 1)
    draw_eye(screen, midpoint, 180, Vec2d(*pointer))
    title = view.large.render(TITLE, True, (238, 193, 176))
    screen.blit(title, title.get_rect(center=(view.screen_size[0] // 2, view.screen_size[1] // 2 + 20)))
    menu_buttons(view, screen, "Start Run", "Sandbox", pointer)
    hard = view.hard_mode_rect
    pygame.draw.rect(screen, (179, 60, 81) if hard.collidepoint(pointer) else (148, 44, 63),
                     hard, border_radius=5)
    label = view.font.render("Hard Mode", True, (255, 241, 237))
    screen.blit(label, label.get_rect(center=hard.center))


def station_card(view: "MazeView") -> pygame.Rect:
    card = pygame.Rect(0, 0, 648, 572)
    card.center = (view.screen_size[0] // 2, view.screen_size[1] // 2)
    return card


def station_rows(view: "MazeView"):
    card = station_card(view)
    for index, upgrade in enumerate(UPGRADES):
        yield pygame.Rect(card.x + 24, card.y + 185 + index * 42, 190, 38), upgrade
    column_size = (len(SHOP_ITEMS) + 1) // 2
    for index, (item, _) in enumerate(SHOP_ITEMS):
        column, row = divmod(index, column_size)
        yield pygame.Rect(card.x + 224 + column * 200, card.y + 185 + row * 29, 190, 25), item


def next_floor_button(view: "MazeView") -> pygame.Rect:
    card = station_card(view)
    return pygame.Rect(card.right - 158, card.bottom - 54, 134, 34)


def buy_at_station(view: "MazeView", maze: "Maze", pointer: tuple[float, float]) -> None:
    for rect, option in station_rows(view):
        if rect.collidepoint(*pointer):
            if isinstance(option, str):
                maze.progress.buy(option)
            else:
                maze.buy_loadout(option)
            return


def draw_end(view: "MazeView", screen: pygame.Surface, maze: "Maze", gaze: Vec2d | None = None) -> None:
    center_x, center_y = view.screen_size[0] // 2, view.screen_size[1] // 2
    if maze.won:
        card = station_card(view)
        draw_panel(screen, card)
        title = view.large.render("Stat Upgrades", True, TITLE_COLOR)
        screen.blit(title, (card.x + 24, card.y + 14))
        floor = view.font.render(f"Floor {maze.progress.level} cleared", True, ACTION)
        screen.blit(floor, floor.get_rect(topright=(card.right - 24, card.y + 25)))
        for index, (earned, criterion) in enumerate(zip(maze.level_eyes, ("Elite", "Scribe", "< 5 min"))):
            eye = Vec2d(card.x + 68 + index * 92, card.y + 81)
            if earned:
                draw_eye(screen, eye, 34, eye)
            else:
                closed_eye(screen, eye, 34, (117, 109, 123))
            label = view.small.render(criterion, True, (211, 192, 177))
            screen.blit(label, label.get_rect(center=(eye.x, eye.y + 20)))
        for index, (earned, criterion) in enumerate(zip(maze.bonus_eyes, ("Mimic", "< 3 min"))):
            eye = Vec2d(card.x + 367 + index * 92, card.y + 81)
            if earned:
                draw_eye(screen, eye, 24, eye)
            else:
                closed_eye(screen, eye, 24, (117, 109, 123))
            label = view.small.render(criterion, True, (211, 192, 177))
            screen.blit(label, label.get_rect(center=(eye.x, eye.y + 20)))
        score = view.font.render(f"Score  {maze.progress.score}", True, (242, 219, 163))
        screen.blit(score, (card.x + 24, card.y + 120))
        detail = view.font.render(f"{maze.progress.xp} XP", True, (241, 203, 131))
        screen.blit(detail, detail.get_rect(topright=(card.right - 24, card.y + 120)))
        pygame.draw.line(screen, EDGE, (card.x + 24, card.y + 148), (card.right - 24, card.y + 148))
        for x, title_text in ((24, f"{STAT_GAIN:+.0%} / Rank"), (224, "Starting Gear")):
            label = view.font.render(title_text, True, ACTION)
            screen.blit(label, (card.x + x, card.y + 157))
        prices = dict(SHOP_ITEMS)
        slots = len(maze.progress.loadout)
        for rect, option in station_rows(view):
            if isinstance(option, str):
                cost = maze.progress.cost(option)
                label = option
                available = cost <= maze.progress.xp
            else:
                cost = prices[option]
                label = option.name
                available = slots < 3 and cost <= maze.progress.xp
            hovered = available and gaze is not None and rect.collidepoint(*gaze)
            fill = (63, 66, 75) if hovered else (47, 40, 55) if available else (37, 31, 44)
            text_color = (233, 234, 223) if available else (150, 158, 172)
            pygame.draw.rect(screen, fill, rect, border_radius=4)
            screen.blit(view.small.render(label, True, text_color), (rect.x + 7, rect.y + 6))
            price = view.small.render(f"{cost} XP", True, (241, 203, 131) if available else text_color)
            screen.blit(price, price.get_rect(topright=(rect.right - 7, rect.y + 6)))
            if isinstance(option, str):
                gain = f"{maze.progress.bonus(option):+.0%}"
                screen.blit(view.small.render(gain, True, ACTION), (rect.x + 7, rect.y + 21))
        recovery = view.small.render(f"Recovery  +{FLOOR_HEAL_FRACTION:.0%} HP", True, ACTION)
        screen.blit(recovery, (card.x + 24, card.y + 492))
        items = ", ".join(item.name for item in maze.progress.loadout)
        packed = f"Packed {slots}/3" + (f"  ·  {items}" if items else "")
        screen.blit(view.small.render(packed, True, (226, 192, 143)), (card.x + 24, card.y + 521))
        button = next_floor_button(view)
        pygame.draw.rect(screen, (74, 104, 98) if gaze is not None and button.collidepoint(*gaze) else (49, 76, 74),
                         button, border_radius=5)
        label = view.font.render("Next Floor", True, ACTION)
        screen.blit(label, label.get_rect(center=button.center))
        return
    card = pygame.Rect(center_x - 191, center_y - 52, 382, 104)
    draw_panel(screen, card)
    title = view.large.render("Defeat", True, TITLE_COLOR)
    help_text = view.font.render("Esc  Menu", True, ACTION)
    screen.blit(title, title.get_rect(center=(center_x, center_y - 13)))
    screen.blit(help_text, help_text.get_rect(center=(center_x, center_y + 24)))


def debug_card(view: "MazeView") -> pygame.Rect:
    card = pygame.Rect(0, 0, min(660, view.screen_size[0] - 32), 602)
    card.center = (view.screen_size[0] // 2, view.screen_size[1] // 2)
    return card


def debug_tabs(view: "MazeView"):
    card = debug_card(view)
    return tuple((pygame.Rect(card.x + 24 + index * 150, card.y + 77, 140, 33), page)
                 for index, page in enumerate(("items", "mobs", "effects")))


def debug_page_buttons(view: "MazeView"):
    card = debug_card(view)
    return (pygame.Rect(card.x + 24, card.bottom - 79, 91, 26),
            pygame.Rect(card.x + 124, card.bottom - 79, 91, 26))


def debug_close_rect(view: "MazeView") -> pygame.Rect:
    card = debug_card(view)
    return pygame.Rect(card.right - 112, card.bottom - 42, 88, 28)


def debug_slider_rect(view: "MazeView") -> pygame.Rect:
    card = debug_card(view)
    return pygame.Rect(card.x + 28, card.y + 161, card.width - 56, 20)


def debug_rows(view: "MazeView", tools: SandboxTools):
    card = debug_card(view)
    if tools.page == "effects":
        start = tools.effect_page * EFFECT_PAGE_SIZE
        options = EFFECT_OPTIONS[start:start + EFFECT_PAGE_SIZE]
    else:
        options = SUMMON_ITEMS if tools.page == "items" else SUMMON_MOBS
    column_size = ceil(len(options) / 2)
    height = 46 if tools.page == "mobs" else 34
    top = card.y + (220 if tools.page == "mobs" else 151)
    width = (card.width - 60) // 2
    for index, option in enumerate(options):
        column, row = divmod(index, column_size)
        rect = pygame.Rect(card.x + 24 + column * (width + 12), top + row * height, width, height - 4)
        yield rect, option


def debug_menu_input(view: "MazeView", maze: "Maze", event: pygame.event.Event, pointer: tuple[float, float]) -> None:
    tools = maze.debug
    if tools is None or not tools.is_open:
        return
    if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
        tools.close()
        return
    if event.type == pygame.KEYDOWN and tools.page == "mobs" and event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        tools.set_hostility_chance(tools.hostility_chance + (0.05 if event.key == pygame.K_RIGHT else -0.05))
    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        tools.dragging_slider = False
        return
    clicked = event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
    slider = debug_slider_rect(view)
    if tools.page == "mobs" and clicked and slider.inflate(20, 14).collidepoint(pointer):
        tools.dragging_slider = True
    if tools.dragging_slider:
        tools.set_hostility_chance((pointer[0] - slider.left) / slider.width)
        return
    if not clicked:
        return
    if debug_close_rect(view).collidepoint(pointer):
        tools.close()
        return
    for rect, page in debug_tabs(view):
        if rect.collidepoint(pointer):
            tools.page = page
            tools.message = ""
            return
    if tools.page == "effects":
        previous, following = debug_page_buttons(view)
        if previous.collidepoint(pointer) or following.collidepoint(pointer):
            change = -1 if previous.collidepoint(pointer) else 1
            tools.effect_page = (tools.effect_page + change) % ceil(len(EFFECT_OPTIONS) / EFFECT_PAGE_SIZE)
            tools.message = ""
            return
    for rect, option in debug_rows(view, tools):
        if rect.collidepoint(pointer):
            if isinstance(option, Equipment):
                tools.spawn_item(option)
            elif tools.page == "effects":
                tools.spawn_effect(option[1])
            else:
                tools.spawn_mob(option[1])
            return


def draw_debug_menu(view: "MazeView", screen: pygame.Surface, maze: "Maze", gaze: Vec2d) -> None:
    tools = maze.debug
    shade = pygame.Surface(view.screen_size, pygame.SRCALPHA)
    shade.fill((8, 9, 14, 190))
    screen.blit(shade, (0, 0))
    card = debug_card(view)
    draw_panel(screen, card)
    icon = view.sprites.sprite("de_bug", (39, 20))
    screen.blit(icon, icon.get_rect(center=(card.x + 46, card.y + 38)))
    screen.blit(view.large.render("De Bug", True, GOLD), (card.x + 78, card.y + 16))
    for rect, page in debug_tabs(view):
        fill = (98, 79, 37) if tools.page == page else (43, 42, 49)
        pygame.draw.rect(screen, fill, rect, border_radius=4)
        label = view.font.render(page.title(), True, (249, 235, 194))
        screen.blit(label, label.get_rect(center=rect.center))
    if tools.page == "mobs":
        view.label(screen, f"New Mob Hostility  ·  {tools.hostility_chance:.0%}",
                   (card.x + 24, card.y + 128), color=GOLD)
        slider = debug_slider_rect(view)
        track = pygame.Rect(slider.x, slider.centery - 3, slider.width, 6)
        pygame.draw.rect(screen, (67, 58, 49), track, border_radius=3)
        filled = round(track.width * tools.hostility_chance)
        pygame.draw.rect(screen, GOLD, (track.x, track.y, filled, track.height), border_radius=3)
        pygame.draw.circle(screen, (253, 233, 170), (track.x + filled, track.centery), 9)
    for rect, option in debug_rows(view, tools):
        hovered = rect.collidepoint(*gaze)
        pygame.draw.rect(screen, (75, 65, 43) if hovered else (43, 42, 49), rect, border_radius=4)
        if isinstance(option, Equipment):
            name = option.name
            longest = max(option.length, option.width)
            dimensions = (max(7, round(26 * option.length / longest)),
                          max(7, round(26 * option.width / longest)))
            sprite = view.sprites.sprite(gear_sprite_name(option), dimensions)
            screen.blit(sprite, sprite.get_rect(center=(rect.x + 23, rect.centery)))
        elif tools.page == "mobs":
            name, kind = option
            sprite_name = {"dummy": "guard", "elite_boss": "elite", "scribe_boss": "scribe"}.get(kind, kind)
            sprite = view.sprites.sprite(sprite_name, (24, 30), "idle", view.now)
            screen.blit(sprite, sprite.get_rect(center=(rect.x + 23, rect.centery)))
        else:
            name, _ = option
            pygame.draw.circle(screen, GOLD, (rect.x + 23, rect.centery), 6, 1)
            pygame.draw.circle(screen, (252, 231, 153), (rect.x + 23, rect.centery), 2)
        label = view.small.render(name, True, (241, 230, 206))
        screen.blit(label, label.get_rect(midleft=(rect.x + 46, rect.centery)))
    if tools.page == "effects":
        for rect, text in zip(debug_page_buttons(view), ("Previous", "Next")):
            pygame.draw.rect(screen, (58, 53, 43), rect, border_radius=4)
            label = view.small.render(text, True, GOLD)
            screen.blit(label, label.get_rect(center=rect.center))
        count = ceil(len(EFFECT_OPTIONS) / EFFECT_PAGE_SIZE)
        page = f"{tools.effect_page + 1} / {count}"
        view.label(screen, page, (card.x + 231, card.bottom - 72), color=GOLD)
    if tools.message:
        y = card.bottom - (110 if tools.page == "effects" else 76)
        view.label(screen, tools.message, (card.x + 24, y), color=GOLD)
    close = debug_close_rect(view)
    pygame.draw.rect(screen, (77, 65, 39), close, border_radius=4)
    label = view.font.render("E  Close", True, (249, 235, 194))
    screen.blit(label, label.get_rect(center=close.center))


def draw_merchant_shop(view: "MazeView", screen: pygame.Surface, maze: "Maze") -> None:
    if maze.shop is None:
        return
    card = pygame.Rect(0, 0, 412, 274)
    card.center = (view.screen_size[0] // 2, view.screen_size[1] // 2)
    draw_panel(screen, card)
    title = view.large.render("Merchant", True, TITLE_COLOR)
    screen.blit(title, (card.x + 23, card.y + 14))
    balance = view.font.render(f"{maze.progress.xp} XP", True, ACTION)
    screen.blit(balance, balance.get_rect(topright=(card.right - 24, card.y + 27)))
    for index, item in enumerate(maze.shop.stock):
        cost = MERCHANT_PRICES[item]
        color = TITLE_COLOR if cost <= maze.progress.xp else MUTED
        power = f" (+{healing_pack_hp(maze.player.max_hp)} HP)" if item is HEAL_PACK else ""
        row = pygame.Rect(card.x + 20, card.y + 72 + index * 30, card.width - 40, 27)
        pygame.draw.rect(screen, (43, 35, 50), row, border_radius=4)
        label = view.font.render(f"{index + 1}    {item.name}{power}", True, color)
        screen.blit(label, (row.x + 8, row.y + 5))
        price = view.small.render(f"{cost} XP", True, color)
        screen.blit(price, price.get_rect(midright=(row.right - 10, row.centery)))
    prompt = view.font.render("E  Close", True, ACTION)
    screen.blit(prompt, (card.x + 23, card.bottom - 35))


def draw_eye_report(view: "MazeView", screen: pygame.Surface, maze: "Maze", gaze: Vec2d,
                    *, defeated: bool = False) -> None:
    screen.fill((13, 8, 15))
    width, height = view.screen_size
    midpoint = Vec2d(width / 2, height / 2 - 82)
    for diameter in (375, 340, 299):
        glow = pygame.Rect(0, 0, diameter, diameter // 2)
        glow.center = point(midpoint)
        pygame.draw.ellipse(screen, (37, 17, 29), glow, 1)
    draw_eye(screen, midpoint, 220, gaze, bloodshot=defeated)
    for index in range(maze.progress.scribe_reports):
        draw_eye(screen, scribe_eye_position(index, midpoint), 32, gaze)

    def centered(text: str, font: pygame.font.Font, y: int, color: tuple[int, int, int]) -> None:
        surface = font.render(text, True, color)
        screen.blit(surface, surface.get_rect(center=(width // 2, y)))

    panel = pygame.Rect(0, 0, min(654, width - 30), 160)
    panel.midbottom = (width // 2, height - 57)
    draw_panel(screen, panel)
    title = "Defeat" if defeated else "The Eye remembers"
    centered(title, view.large, 67, TITLE_COLOR)
    if defeated:
        death = maze.death
        progress = maze.progress
        centered(death.quote, view.font, panel.top - 24, TITLE_COLOR)
        cause = death.weapon
        if death.attacker and death.attacker != cause:
            cause += f"  ·  {death.attacker}"
        centered(cause, view.font, panel.top + 28, TITLE_COLOR)
        seconds = int(progress.elapsed_floors + maze.wielding.time)
        metrics = (("Floor", progress.level), ("Time", f"{seconds // 60}:{seconds % 60:02d}"),
                   ("Score", progress.score), ("XP", progress.xp), ("Defeats", progress.enemies_defeated),
                   ("Dealt", progress.damage_dealt), ("Taken", progress.damage_taken))
        draw_metrics(view, screen, panel, metrics)
        centered("Esc  Menu", view.font, height - 29, ACTION)
    else:
        victory = maze.victory
        centered(victory.quote, view.font, panel.top - 24, TITLE_COLOR)
        centered(f"Enemy Upgrades  ·  {victory.enemy_upgrade_xp} XP", view.font, panel.top + 23, TITLE_COLOR)
        gains = [f"{stat} {ranks * STAT_GAIN:+.0%}" for stat, ranks in victory.enemy_upgrades.items()]
        centered("  ·  ".join(gains) or "XP Banked", view.small, panel.top + 49, ACTION)
        metrics = (("Attack", f"{maze.habits.combat_read():.0%}"), ("Evasion", f"{maze.habits.evasion_read():.0%}"),
                   ("Reports", maze.progress.scribe_reports), ("Trials", maze.progress.policy.observations))
        draw_metrics(view, screen, panel, metrics)
        centered("Enter  Stat Upgrades", view.font, height - 29, ACTION)


def draw_metrics(view: "MazeView", screen: pygame.Surface, panel: pygame.Rect, metrics) -> None:
    width = (panel.width - 32) // len(metrics)
    for index, (name, value) in enumerate(metrics):
        rect = pygame.Rect(panel.x + 16 + index * width, panel.bottom - 79, width - 4, 61)
        pygame.draw.rect(screen, (41, 32, 48), rect, border_radius=5)
        reading = view.font.render(str(value), True, TITLE_COLOR)
        label = view.small.render(name, True, MUTED)
        screen.blit(reading, reading.get_rect(center=(rect.centerx, rect.y + 20)))
        screen.blit(label, label.get_rect(center=(rect.centerx, rect.y + 43)))
