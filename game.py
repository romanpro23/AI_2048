import random
import sys

import numpy as np
import pygame

# Константи
WINDOW_SIZE = 400
GRID_SIZE = 4
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
CELL_PADDING = 3
TILE_SIZE = CELL_SIZE - CELL_PADDING * 2
FONT_SIZE = TILE_SIZE // 2
HEADER_HEIGHT = 80

# Ініціалізація Pygame
pygame.init()

# Налаштування екрану
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + HEADER_HEIGHT))
pygame.display.set_caption('2048')

# Шрифти
font = pygame.font.Font(None, FONT_SIZE)
score_font = pygame.font.Font(None, 30)
header_font = pygame.font.Font(None, 48)

# Кольори
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


class Game:
    def __init__(self):
        self.board = Board()
        self.board.add_new_tile()
        self.board.add_new_tile()
        self.score = 0
        self.max_score = 0

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    moved = False
                    if event.key == pygame.K_LEFT:
                        moved = self.board.move_left()
                    elif event.key == pygame.K_RIGHT:
                        moved = self.board.move_right()
                    elif event.key == pygame.K_UP:
                        moved = self.board.move_up()
                    elif event.key == pygame.K_DOWN:
                        moved = self.board.move_down()

                    if moved:
                        self.score += self.board.merged_score
                        self.max_score = max(self.max_score, self.score)
                        self.board.add_new_tile()

            screen.fill(BACKGROUND_COLOR)
            self.board.draw(screen)
            self.draw_header(screen)
            pygame.display.flip()

            if not self.board.can_move():
                print("Game over!")
                running = False

        pygame.quit()
        sys.exit()

    def draw_header(self, surface):
        pygame.draw.rect(surface, BACKGROUND_COLOR, (0, 0, WINDOW_SIZE, HEADER_HEIGHT))

        score_text_1 = score_font.render("Score:", True, (255, 255, 255))
        score_text_2 = score_font.render(str(self.score), True, (255, 255, 255))
        max_score_text_1 = score_font.render("Max score:", True, (255, 255, 255))
        max_score_text_2 = score_font.render(str(self.max_score), True, (255, 255, 255))

        box_width = 180
        score_rect_bg = pygame.Rect(10, 10, box_width, 60)
        max_score_rect_bg = pygame.Rect(WINDOW_SIZE - box_width - 10, 10, box_width, 60)

        pygame.draw.rect(surface, TILE_COLORS[0], score_rect_bg, border_radius=5)
        pygame.draw.rect(surface, TILE_COLORS[0], max_score_rect_bg, border_radius=5)

        score_text_rect_1 = score_text_1.get_rect(center=(score_rect_bg.centerx, score_rect_bg.centery - 10))
        score_text_rect_2 = score_text_2.get_rect(center=(score_rect_bg.centerx, score_rect_bg.centery + 10))
        max_score_text_rect_1 = max_score_text_1.get_rect(
            center=(max_score_rect_bg.centerx, max_score_rect_bg.centery - 10))
        max_score_text_rect_2 = max_score_text_2.get_rect(
            center=(max_score_rect_bg.centerx, max_score_rect_bg.centery + 10))

        surface.blit(score_text_1, score_text_rect_1)
        surface.blit(score_text_2, score_text_rect_2)
        surface.blit(max_score_text_1, max_score_text_rect_1)
        surface.blit(max_score_text_2, max_score_text_rect_2)


class Board:
    def __init__(self):
        self.tiles = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.merged_score = 0

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.tiles == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.tiles[i][j] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        moved = False
        self.merged_score = 0
        for row in range(GRID_SIZE):
            new_row, change, score = self._merge(self.tiles[row])
            self.tiles[row] = new_row
            if change:
                moved = True
            self.merged_score += score
        return moved

    def move_right(self):
        self.tiles = np.fliplr(self.tiles)
        moved = self.move_left()
        self.tiles = np.fliplr(self.tiles)
        return moved

    def move_up(self):
        self.tiles = np.transpose(self.tiles)
        moved = self.move_left()
        self.tiles = np.transpose(self.tiles)
        return moved

    def move_down(self):
        self.tiles = np.transpose(self.tiles)
        moved = self.move_right()
        self.tiles = np.transpose(self.tiles)
        return moved

    def _merge(self, row):
        tight = row[row != 0]
        merged = []
        skip = False
        score = 0
        for j in range(len(tight)):
            if skip:
                skip = False
                continue
            if j + 1 < len(tight) and tight[j] == tight[j + 1]:
                merged.append(tight[j] * 2)
                score += tight[j] * 2
                skip = True
            else:
                merged.append(tight[j])
        merged.extend([0] * (GRID_SIZE - len(merged)))
        change = not np.array_equal(row, merged)
        return np.array(merged), change, score

    def can_move(self):
        if np.any(self.tiles == 0):
            return True
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE - 1):
                if self.tiles[row][col] == self.tiles[row][col + 1]:
                    return True
                if self.tiles[col][row] == self.tiles[col + 1][row]:
                    return True
        return False

    def draw(self, surface):
        surface.fill(BACKGROUND_COLOR)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                Tile(int(self.tiles[row][col])).draw(surface, row, col)


class Tile:
    def __init__(self, value=0):
        self.value = value

    def draw(self, surface, row, col):
        color = TILE_COLORS.get(self.value, (60, 58, 50))
        pygame.draw.rect(surface, color, (
            col * CELL_SIZE + CELL_PADDING, row * CELL_SIZE + HEADER_HEIGHT + CELL_PADDING, TILE_SIZE, TILE_SIZE))
        if self.value != 0:
            text = font.render(str(self.value), True, (0, 0, 0) if self.value <= 4 else (255, 255, 255))
            text_rect = text.get_rect(
                center=(col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + HEADER_HEIGHT + CELL_SIZE // 2))
            surface.blit(text, text_rect)


if __name__ == "__main__":
    game = Game()
    game.run()
