import numpy as np
from matplotlib import pyplot as plt
from options import *

class MonitoringPanel:
    def __init__(self, game):
        self.game = game
        self.prev_episode = 1

    def draw(self, surface):
        # Drawing text on the monitoring panel
        monitoring_font = pygame.font.Font(None, 32)
        label_font = pygame.font.Font(None, 28)  # Smaller font for labels
        header_text = monitoring_font.render(f"Monitoring Panel for {self.game.agent.get_name()}", True,
                                             (255, 255, 255))

        header_rect_bg = pygame.Rect(10 + WINDOW_SIZE, 10, MONITORING_PANEL_SIZE - 30, 60)
        pygame.draw.rect(surface, TILE_COLORS[0], header_rect_bg, border_radius=5)

        header_text_rect = header_text.get_rect(center=(header_rect_bg.centerx, header_rect_bg.centery))
        surface.blit(header_text, header_text_rect)

        # Hyperparameters info
        hyperparams = self.game.agent.get_hyperparams()

        small_font = pygame.font.Font(None, 24)  # Smaller font for hyperparameters

        # Display hyperparameters in two columns
        y_offset = 90  # Starting y position for hyperparameters
        column_1_x = 10 + WINDOW_SIZE  # Moved 10 pixels left
        column_2_x = column_1_x + 190  # Adjust the horizontal spacing as needed
        index = 0

        for param, value in hyperparams.items():
            # Draw the background rectangle for each hyperparameter
            param_bg_rect = pygame.Rect(column_1_x if index % 2 == 0 else column_2_x,
                                        y_offset + (index // 2) * 35, 180, 30)
            pygame.draw.rect(surface, TILE_COLORS[0], param_bg_rect, border_radius=5)

            param_text = small_font.render(f"{param}: {value}", True, (255, 255, 255))
            param_text_rect = param_text.get_rect(center=param_bg_rect.center)
            surface.blit(param_text, param_text_rect)
            index += 1

        # Labels for scores and moves
        score_label_bg = pygame.Rect(10 + WINDOW_SIZE, 170, (MONITORING_PANEL_SIZE - 40) // 2, 30)
        moves_label_bg = pygame.Rect(20 + WINDOW_SIZE + (MONITORING_PANEL_SIZE - 40) // 2, 170,
                                     (MONITORING_PANEL_SIZE - 40) // 2, 30)
        pygame.draw.rect(surface, TILE_COLORS[0], score_label_bg, border_radius=5)
        pygame.draw.rect(surface, TILE_COLORS[0], moves_label_bg, border_radius=5)

        score_rect_bg = pygame.Rect(10 + WINDOW_SIZE, 210, (MONITORING_PANEL_SIZE - 40) // 2, 100)
        moves_rect_bg = pygame.Rect(20 + WINDOW_SIZE + (MONITORING_PANEL_SIZE - 40) // 2, 210,
                                    (MONITORING_PANEL_SIZE - 40) // 2, 100)
        pygame.draw.rect(surface, (238, 228, 218), score_rect_bg, border_radius=5)
        pygame.draw.rect(surface, (238, 228, 218), moves_rect_bg, border_radius=5)

        # Plotting score schedule
        self.plot_score_schedule(surface, (score_rect_bg.left - 13, score_rect_bg.top))

        # Plotting moves schedule
        self.plot_moves_schedule(surface, (moves_rect_bg.left - 13, moves_rect_bg.top))

        # Adding labels for the plots
        score_label = label_font.render("Scores History", True, (255, 255, 255))
        moves_label = label_font.render("Moves History", True, (255, 255, 255))

        score_label_rect = score_label.get_rect(center=(score_label_bg.centerx, score_label_bg.centery))
        moves_label_rect = moves_label.get_rect(center=(moves_label_bg.centerx, moves_label_bg.centery))

        surface.blit(score_label, score_label_rect)
        surface.blit(moves_label, moves_label_rect)

        # Adding histogram of weights distribution
        weights_rect_bg = pygame.Rect(10 + WINDOW_SIZE, 360, (MONITORING_PANEL_SIZE - 40) // 2, 100)
        pygame.draw.rect(surface, (238, 228, 218), weights_rect_bg, border_radius=5)

        weights_label_bg = pygame.Rect(10 + WINDOW_SIZE, 320, (MONITORING_PANEL_SIZE - 40) // 2, 30)
        pygame.draw.rect(surface, TILE_COLORS[0], weights_label_bg, border_radius=5)

        self.plot_weights_distribution(surface, (weights_rect_bg.left - 13, weights_rect_bg.top))

        # Adding histogram of actions distribution
        actions_rect_bg = pygame.Rect(20 + WINDOW_SIZE + (MONITORING_PANEL_SIZE - 40) // 2, 360,
                                      (MONITORING_PANEL_SIZE - 40) // 2, 100)
        pygame.draw.rect(surface, (238, 228, 218), actions_rect_bg, border_radius=5)

        actions_label_bg = pygame.Rect(20 + WINDOW_SIZE + (MONITORING_PANEL_SIZE - 40) // 2, 320,
                                       (MONITORING_PANEL_SIZE - 40) // 2, 30)
        pygame.draw.rect(surface, TILE_COLORS[0], actions_label_bg, border_radius=5)

        self.plot_actions_distribution(surface, (actions_rect_bg.left - 13, actions_rect_bg.top))

        # Adding labels for the histograms
        weights_label = label_font.render("Weights", True, (255, 255, 255))
        actions_label = label_font.render("Actions", True, (255, 255, 255))

        weights_label_rect = weights_label.get_rect(center=(weights_label_bg.centerx, weights_label_bg.centery))
        actions_label_rect = actions_label.get_rect(center=(actions_label_bg.centerx, actions_label_bg.centery))

        surface.blit(weights_label, weights_label_rect)
        surface.blit(actions_label, actions_label_rect)

        self.prev_episode = self.game.episode

    def plot_score_schedule(self, surface, topleft):
        if self.game.episode != self.prev_episode and len(self.game.scores_history) >= 2:
            plt.figure(figsize=(2, 1))

            x = np.arange(1, len(self.game.scores_history) + 1)
            y = np.array(self.game.scores_history)
            plt.plot(x, y, color='black', linewidth=2)

            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            plt.savefig('images/score_schedule.png', transparent=True)
            plt.close()

        if len(self.game.scores_history) >= 2:
            plot_image = pygame.image.load('./images/score_schedule.png')
            plot_rect = plot_image.get_rect(topleft=topleft)
            surface.blit(plot_image, plot_rect)

    def plot_moves_schedule(self, surface, topleft):
        if len(self.game.moves_history) > 2 and self.game.episode != self.prev_episode:
            plt.figure(figsize=(2, 1))

            x = np.arange(1, len(self.game.moves_history) + 1)
            y = np.array(self.game.moves_history)
            plt.plot(x, y, color='black', linewidth=2)

            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

            plt.savefig('images/moves_schedule.png', transparent=True)
            plt.close()

        if len(self.game.moves_history) > 2:
            plot_image = pygame.image.load('images/moves_schedule.png')
            plot_rect = plot_image.get_rect(topleft=topleft)
            surface.blit(plot_image, plot_rect)

    def plot_weights_distribution(self, surface, topleft):
        if self.game.episode != self.prev_episode:
            weights = np.concatenate(
                [param.detach().cpu().numpy().flatten() for param in self.game.agent.get_weights()])
            plt.figure(figsize=(2, 1))
            plt.hist(weights, bins=30, color='black')

            xmin, xmax = plt.xlim()
            limit = max(abs(xmin), abs(xmax))
            plt.xlim(-limit, limit)

            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.savefig('images/weights_distribution.png', transparent=True)
            plt.close()

        plot_image = pygame.image.load('images/weights_distribution.png')
        plot_rect = plot_image.get_rect(topleft=topleft)
        surface.blit(plot_image, plot_rect)

    def plot_actions_distribution(self, surface, topleft):
        if len(self.game.moves_history) > 0 and self.game.episode != self.prev_episode and not self.game.manual_control:
            actions = self.game.actions
            plt.figure(figsize=(2, 1))
            plt.hist(actions, bins=np.arange(self.game.agent.action_size + 1) - 0.5, rwidth=0.8, color='black')
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.savefig('images/actions_distribution.png', transparent=True)
            plt.close()

            self.game.actions = []

        plot_image = pygame.image.load('images/actions_distribution.png')
        plot_rect = plot_image.get_rect(topleft=topleft)
        surface.blit(plot_image, plot_rect)
