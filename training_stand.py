import sys

from ac import ActorCriticAgent, A2CAgent
from agent import Agent
from dqn import DQNAgent, DoubleDQNAgent
from environments import Environment, EnvironmentHotEncoding
from help_window import HelpWindow
from mechanics import Board
from monitoring_panel import MonitoringPanel
from options import *
from ppo import PPOAgent
from vpg import VPGAgent

pygame.init()

screen = pygame.display.set_mode((WINDOW_SIZE + MONITORING_PANEL_SIZE, WINDOW_SIZE + HEADER_HEIGHT))
pygame.display.set_caption('2048')


class Game:
    agent: Agent
    environment: Environment

    def __init__(self):
        self.board = Board()
        self.board.add_new_tile()
        self.board.add_new_tile()
        self.score = 0
        self.max_score = 0

        self.agent = DQNAgent(12 * GRID_SIZE * GRID_SIZE, 4)

        self.episode = 0
        self.scores_history = []
        self.moves_history = []

        self.monitoring_panel = MonitoringPanel(self)
        self.environment = EnvironmentHotEncoding()

        self.help_window = HelpWindow()

        self.actions = []
        self.manual_control = False

        self.agent_name = "dqn"

    def run(self):
        running = True
        while running:
            self.episode += 1
            state = self.environment.get_state(self.board.tiles.flatten())

            done = False
            moves = 0

            while not done and not self.manual_control:
                action = self.agent.act(state)
                moved = self.board.move(action)

                next_state = self.environment.get_state(self.board.tiles.flatten())
                done = not self.board.can_move() or not moved

                reward = self.environment.get_reward(done, moved, self.board.merged_score)
                self.actions.append(action)

                if moved:
                    self.board.add_new_tile()

                    self.score += self.board.merged_score

                if self.score > self.max_score:
                    self.max_score = self.score

                self.agent.remember(state, action, reward, next_state, done)
                state = next_state

                self.draw()

                moves += 1

                if isinstance(self.agent, DQNAgent):
                    self.agent.replay()

                if done:
                    self.restart(moves)
                    self.agent.replay()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_r:
                        self.manual_control = not self.manual_control
                        self.restart()
                        self.draw()
                    elif event.key == pygame.K_F1:
                        self.help_window.toggle_visibility()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1:
                        self.change_agent("dqn")
                    elif event.key == pygame.K_2:
                        self.change_agent("vpg")
                    elif event.key == pygame.K_3:
                        self.change_agent("ddqn")
                    elif event.key == pygame.K_4:
                        self.change_agent("ac")
                    elif event.key == pygame.K_5:
                        self.change_agent("a2c")
                    elif event.key == pygame.K_6:
                        self.change_agent("ppo")
                    elif event.key == pygame.K_l:
                        self.load_weight()

                    if self.manual_control:
                        if event.key == pygame.K_UP or event.key == pygame.K_w:
                            self.manual_controller(2)
                        elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                            self.manual_controller(3)
                        elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                            self.manual_controller(0)
                        elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                            self.manual_controller(1)

        pygame.quit()
        sys.exit()

    def change_agent(self, agent_name):
        self.agent_name = agent_name

        self.moves_history = []
        self.scores_history = []

        if agent_name == "dqn":
            self.agent = DQNAgent(12 * 16, 4)
        elif agent_name == "ddqn":
            self.agent = DoubleDQNAgent(12 * 16, 4)
        elif agent_name == "vpg":
            self.agent = VPGAgent(12 * 16, 4)
        elif agent_name == "ac":
            self.agent = ActorCriticAgent(12 * 16, 4)
        elif agent_name == "a2c":
            self.agent = A2CAgent(12 * 16, 4)
        elif agent_name == "ppo":
            self.agent = PPOAgent(12 * 16, 4)

        self.restart()
        self.draw()

    def load_weight(self):
        print("load")
        self.agent.load(f"models/{self.agent_name}")

    def manual_controller(self, action):
        moved = self.board.move(action)
        done = not self.board.can_move()

        if moved:
            self.board.add_new_tile()
            self.score += self.board.merged_score

        if self.score > self.max_score:
            self.max_score = self.score
        if done:
            self.restart()

        self.draw()

    def draw(self):
        screen.fill(BACKGROUND_COLOR)
        self.board.draw(screen)
        self.draw_header(screen)
        self.draw_monitoring(screen)

        if self.help_window.visible:
            self.help_window.draw(screen)

        pygame.display.flip()

    def restart(self, moves=None):
        if moves is not None:
            self.moves_history.append(moves)
            self.scores_history.append(self.score)

        self.board = Board()
        self.board.add_new_tile()
        self.board.add_new_tile()
        self.score = 0

    def draw_monitoring(self, surface):
        self.monitoring_panel.draw(surface)

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


if __name__ == "__main__":
    game = Game()
    game.run()
