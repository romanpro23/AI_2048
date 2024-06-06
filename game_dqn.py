import random
import sys

import numpy as np
import pygame
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from dqn import DQNAgent

# Константи
WINDOW_SIZE = 400
MONITORING_PANEL_SIZE = 400
GRID_SIZE = 4
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
CELL_PADDING = 3
TILE_SIZE = CELL_SIZE - CELL_PADDING * 2
FONT_SIZE = TILE_SIZE // 2
HEADER_HEIGHT = 80

# Ініціалізація Pygame
pygame.init()

# Налаштування екрану
screen = pygame.display.set_mode((WINDOW_SIZE + MONITORING_PANEL_SIZE, WINDOW_SIZE + HEADER_HEIGHT))
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
        self.agent = DQNAgent(GRID_SIZE * GRID_SIZE, 4)
        self.batch_size = 32
        self.episode = 0
        self.scores_history = []
        self.moves_history = []

        self.last_scores_len = 0
        self.last_moves_len = 0

    def run(self):
        running = True
        while running:
            self.episode += 1
            state = self.board.tiles.flatten()
            state[state == 0] = 1
            state = np.log2(state)

            done = False
            reward = 0
            moves = 0
            while not done:
                action = self.agent.act(state)
                moved = self.board.move(action)

                next_state = self.board.tiles.flatten()
                next_state[next_state == 0] = 1
                next_state = np.log2(next_state)

                done = not self.board.can_move()

                reward = -10 if done else np.log2(
                    self.board.merged_score + 1) if moved else -1 if reward > 0 else reward - 0.1

                if moved:
                    self.board.add_new_tile()

                    self.score += self.board.merged_score
                    self.max_score = max(self.max_score, self.score)

                self.agent.remember(state, action, reward, next_state, done)
                state = next_state

                screen.fill(BACKGROUND_COLOR)
                self.board.draw(screen)
                self.draw_header(screen)
                self.draw_monitoring(screen)
                pygame.display.flip()

                moves += 1

                if len(self.agent.memory) > self.batch_size:
                    self.agent.replay(self.batch_size)

                if done:
                    print(f"Episode: {self.episode}, Score: {self.score}, Moves: {moves}, Epsilon: {self.agent.epsilon:.2}")
                    self.scores_history.append(self.score)
                    self.moves_history.append(moves)
                    self.board = Board()
                    self.board.add_new_tile()
                    self.board.add_new_tile()
                    self.score = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()
        sys.exit()

    def draw_monitoring(self, surface):
        # Drawing text on the monitoring panel
        monitoring_font = pygame.font.Font(None, 32)
        label_font = pygame.font.Font(None, 28)  # Smaller font for labels
        header_text = monitoring_font.render("Monitoring Panel", True, (255, 255, 255))

        header_rect_bg = pygame.Rect(10 + WINDOW_SIZE, 10, MONITORING_PANEL_SIZE - 20, 60)
        pygame.draw.rect(surface, TILE_COLORS[0], header_rect_bg, border_radius=5)

        header_text_rect = header_text.get_rect(center=(header_rect_bg.centerx, header_rect_bg.centery))
        surface.blit(header_text, header_text_rect)

        # Empty rectangle for hyperparameters
        hyperparams_rect_bg = pygame.Rect(10 + WINDOW_SIZE, 80, MONITORING_PANEL_SIZE - 20, 60)
        pygame.draw.rect(surface, (238, 228, 218), hyperparams_rect_bg, border_radius=5)

        # Labels for scores and moves
        score_label_bg = pygame.Rect(10 + WINDOW_SIZE, 150, (MONITORING_PANEL_SIZE - 40) // 2, 30)
        moves_label_bg = pygame.Rect(20 + WINDOW_SIZE + (MONITORING_PANEL_SIZE - 40) // 2, 150,
                                     (MONITORING_PANEL_SIZE - 40) // 2, 30)
        pygame.draw.rect(surface, TILE_COLORS[0], score_label_bg, border_radius=5)
        pygame.draw.rect(surface, TILE_COLORS[0], moves_label_bg, border_radius=5)

        score_rect_bg = pygame.Rect(10 + WINDOW_SIZE, 190, (MONITORING_PANEL_SIZE - 40) // 2, 100)
        moves_rect_bg = pygame.Rect(20 + WINDOW_SIZE + (MONITORING_PANEL_SIZE - 40) // 2, 190,
                                    (MONITORING_PANEL_SIZE - 40) // 2, 100)
        pygame.draw.rect(surface, (238, 228, 218), score_rect_bg, border_radius=5)
        pygame.draw.rect(surface, (238, 228, 218), moves_rect_bg, border_radius=5)

        # Plotting score schedule
        if len(self.scores_history) > 1:
            if len(self.scores_history) != self.last_scores_len:
                plt.figure(figsize=(2, 1))

                x = np.arange(1, len(self.scores_history) + 1)
                y = np.array(self.scores_history)
                cs = CubicSpline(x, y, bc_type='natural')
                x_new = np.linspace(1, len(self.scores_history), num=1000)
                plt.plot(x_new, cs(x_new), color='black', linewidth=2)

                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)

                plt.savefig('score_schedule.png', transparent=True)
                plt.close()

            plot_image = pygame.image.load('score_schedule.png')
            plot_rect = plot_image.get_rect(topleft=(WINDOW_SIZE-3, 190))
            surface.blit(plot_image, plot_rect)

            self.last_scores_len = len(self.scores_history)

        # Plotting moves schedule
        if len(self.moves_history) > 1:
            if len(self.moves_history) != self.last_moves_len:
                plt.figure(figsize=(2, 1))

                x = np.arange(1, len(self.moves_history) + 1)
                y = np.array(self.moves_history)
                cs = CubicSpline(x, y, bc_type='natural')
                x_new = np.linspace(1, len(self.moves_history), num=1000)
                plt.plot(x_new, cs(x_new), color='black', linewidth=2)

                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)

                plt.savefig('moves_schedule.png', transparent=True)
                plt.close()

            plot_image = pygame.image.load('moves_schedule.png')
            plot_rect = plot_image.get_rect(topleft=(7 + WINDOW_SIZE + (MONITORING_PANEL_SIZE - 40) // 2, 190))
            surface.blit(plot_image, plot_rect)

            self.last_moves_len = len(self.moves_history)

        # Adding labels for the plots
        score_label = label_font.render("Scores History", True, (255, 255, 255))
        moves_label = label_font.render("Moves History", True, (255, 255, 255))

        score_label_rect = score_label.get_rect(center=(score_label_bg.centerx, score_label_bg.centery))
        moves_label_rect = moves_label.get_rect(center=(moves_label_bg.centerx, moves_label_bg.centery))

        surface.blit(score_label, score_label_rect)
        surface.blit(moves_label, moves_label_rect)

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
