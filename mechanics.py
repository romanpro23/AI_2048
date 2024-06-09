import random

import numpy as np
from options import *

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
            new_row, change, score = self.__merge(self.tiles[row])
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

    def move(self, action):
        if action == 0:
            return self.move_left()
        elif action == 1:
            return self.move_right()
        elif action == 2:
            return self.move_up()
        elif action == 3:
            return self.move_down()
        return False

    def __merge(self, row):
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
