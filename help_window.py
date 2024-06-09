from options import *


class HelpWindow:
    def __init__(self):
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.window_surface = pygame.Surface((400, 500))
        self.window_rect = self.window_surface.get_rect(
            center=(WINDOW_SIZE // 2 + MONITORING_PANEL_SIZE // 2, WINDOW_SIZE // 2 + 50))
        self.text_offset = 20
        self.visible = False

    def draw(self, surface):
        # Fill the help window background
        self.window_surface.fill(BACKGROUND_COLOR)

        # Help text lines
        help_text = [
            "Help Window",
            "",
            "Arrow keys or WASD: Move tiles",
            "F1: Open this window",
            "R: Toggle to manual/ai control",
            "L: Load pretrained model",
            "1: Change agent to Deep Q-Network",
            "2: Change agent to Policy Gradient",
            "3: Change agent to Double Deep Q-Network",
            "4: Change agent to Actor-Critic",
            "5: Change agent to Advantage Actor-Critic",
            "6: Change agent to Proximal Policy Optimization",
            "ESC: Quit the game"
        ]

        # Render and draw each line of help text
        small_font = pygame.font.Font(None, 24)
        title_font = pygame.font.Font(None, 32)
        for i, text in enumerate(help_text):

            if text == "":
                continue

            param_bg_rect = pygame.Rect(
                self.window_rect.x - 190,
                self.text_offset + i * 35,
                self.window_rect.width - 20,
                30
            )
            pygame.draw.rect(self.window_surface, TILE_COLORS[0], param_bg_rect, border_radius=5)

            text_render = small_font.render(text, True, (255, 255, 255)) if i != 0 else title_font.render(text, True, (
            255, 255, 255))
            text_rect = text_render.get_rect(center=param_bg_rect.center)
            self.window_surface.blit(text_render, text_rect)

        pygame.draw.rect(self.window_surface, (0, 0, 0), (0, 0, 400, 480), 2)
        surface.blit(self.window_surface, self.window_rect.topleft)

    def toggle_visibility(self):
        self.visible = not self.visible
