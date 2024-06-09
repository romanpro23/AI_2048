import pygame

WINDOW_SIZE = 400
MONITORING_PANEL_SIZE = 400
GRID_SIZE = 4
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
CELL_PADDING = 3
TILE_SIZE = CELL_SIZE - CELL_PADDING * 2
FONT_SIZE = TILE_SIZE // 2
HEADER_HEIGHT = 80

pygame.font.init()
font = pygame.font.Font(None, FONT_SIZE)
score_font = pygame.font.Font(None, 30)
header_font = pygame.font.Font(None, 48)

BACKGROUND_COLOR = (187, 173, 160)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (237, 191, 38),
    8192: (246, 76, 82),
}