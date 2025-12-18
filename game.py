import pygame
import random
import numpy as np

# Initialisation de Pygame
pygame.init()

# Constantes du jeu
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 90
BALL_SIZE = 15
PADDLE_SPEED = 5
BALL_SPEED_X = 4
BALL_SPEED_Y = 4

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Paddle:
    """Classe représentant une raquette"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = PADDLE_WIDTH
        self.height = PADDLE_HEIGHT
        self.speed = PADDLE_SPEED
        self.rect = pygame.Rect(x, y, self.width, self.height)

    def move_up(self):
        if self.y > 0:
            self.y -= self.speed
            self.rect.y = self.y

    def move_down(self):
        if self.y < WINDOW_HEIGHT - self.height:
            self.y += self.speed
            self.rect.y = self.y

    def move_to(self, y):
        self.y = max(0, min(y, WINDOW_HEIGHT - self.height))
        self.rect.y = self.y

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

class Ball:
    """Classe représentant la balle"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = WINDOW_WIDTH // 2
        self.y = WINDOW_HEIGHT // 2
        self.speed_x = BALL_SPEED_X * random.choice([-1, 1])
        self.speed_y = BALL_SPEED_Y * random.choice([-1, 1])
        self.rect = pygame.Rect(self.x, self.y, BALL_SIZE, BALL_SIZE)

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
        self.rect.x = self.x
        self.rect.y = self.y

        # Rebond sur les murs haut et bas
        if self.y <= 0 or self.y >= WINDOW_HEIGHT - BALL_SIZE:
            self.speed_y *= -1

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect)

class PongGame:
    """Classe principale du jeu PONG"""
    def __init__(self, mode='RL_vs_AI'):
        self.mode = mode
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f'PONG Q-Learning - {mode}')
        self.clock = pygame.time.Clock()

        # Création des raquettes
        self.player_paddle = Paddle(30, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.opponent_paddle = Paddle(WINDOW_WIDTH - 30 - PADDLE_WIDTH,
                                      WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)

        # Création de la balle
        self.ball = Ball()

        # Scores
        self.player_score = 0
        self.opponent_score = 0

        # État du jeu
        self.running = True
        self.game_over = False

        # Font pour l'affichage
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 36)

    def get_state(self):
        """Retourne l'état actuel du jeu pour l'agent RL"""
        ball_x_norm = self.ball.x / WINDOW_WIDTH
        ball_y_norm = self.ball.y / WINDOW_HEIGHT
        ball_speed_x_norm = self.ball.speed_x / BALL_SPEED_X
        ball_speed_y_norm = self.ball.speed_y / BALL_SPEED_Y
        paddle_y_norm = self.player_paddle.y / WINDOW_HEIGHT
        opponent_y_norm = self.opponent_paddle.y / WINDOW_HEIGHT
        distance = abs(self.ball.x - self.player_paddle.x) / WINDOW_WIDTH

        return np.array([
            ball_x_norm, ball_y_norm,
            ball_speed_x_norm, ball_speed_y_norm,
            paddle_y_norm, opponent_y_norm, distance
        ])

    def discretize_state(self):
        ball_y_rel = int((self.ball.y - self.player_paddle.y) / (WINDOW_HEIGHT / 10))
        ball_y_rel = max(-5, min(5, ball_y_rel))
        ball_dir_x = 0 if self.ball.speed_x < 0 else 1
        ball_dir_y = 0 if self.ball.speed_y < 0 else 1
        ball_distance = int(abs(self.ball.x - self.player_paddle.x) / (WINDOW_WIDTH / 5))
        ball_distance = min(4, ball_distance)
        return (ball_y_rel + 5, ball_dir_x, ball_dir_y, ball_distance)

    def step(self, action):
        reward = 0
        done = False

        # Action du joueur RL
        if action == 1:
            self.player_paddle.move_up()
        elif action == 2:
            self.player_paddle.move_down()

        # Mouvement adversaire simple
        if self.mode == 'RL_vs_AI':
            self.ai_opponent_move()
        elif self.mode == 'RL_vs_RL':
            pass

        # Déplacement balle
        self.ball.move()

        # Collision joueur
        if self.ball.rect.colliderect(self.player_paddle.rect):
            self.ball.speed_x *= -1
            self.ball.x = self.player_paddle.x + self.player_paddle.width
            reward = 0.1

        # Collision adversaire
        if self.ball.rect.colliderect(self.opponent_paddle.rect):
            self.ball.speed_x *= -1
            self.ball.x = self.opponent_paddle.x - BALL_SIZE

        # Vérification des points
        if self.ball.x <= 0:
            self.opponent_score += 1
            reward = -1
            self.ball.reset()
            if self.opponent_score >= 5:
                done = True

        if self.ball.x >= WINDOW_WIDTH - BALL_SIZE:
            self.player_score += 1
            reward = 1
            self.ball.reset()
            if self.player_score >= 5:
                done = True

        next_state = self.get_state()
        return next_state, reward, done

    def ai_opponent_move(self):
        if self.opponent_paddle.y + self.opponent_paddle.height // 2 < self.ball.y:
            self.opponent_paddle.move_down()
        elif self.opponent_paddle.y + self.opponent_paddle.height // 2 > self.ball.y:
            self.opponent_paddle.move_up()

    def handle_human_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.opponent_paddle.move_up()
        if keys[pygame.K_DOWN]:
            self.opponent_paddle.move_down()

    def render(self):
        self.screen.fill(BLACK)
        pygame.draw.line(self.screen, GRAY, (WINDOW_WIDTH // 2, 0),
                         (WINDOW_WIDTH // 2, WINDOW_HEIGHT), 2)
        self.player_paddle.draw(self.screen)
        self.opponent_paddle.draw(self.screen)
        self.ball.draw(self.screen)

        player_text = self.font.render(str(self.player_score), True, WHITE)
        opponent_text = self.font.render(str(self.opponent_score), True, WHITE)
        self.screen.blit(player_text, (WINDOW_WIDTH // 4, 20))
        self.screen.blit(opponent_text, (3 * WINDOW_WIDTH // 4, 20))

        mode_text = self.small_font.render(self.mode, True, WHITE)
        self.screen.blit(mode_text, (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT - 40))

        pygame.display.flip()

    def reset(self):
        self.player_paddle = Paddle(30, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.opponent_paddle = Paddle(WINDOW_WIDTH - 30 - PADDLE_WIDTH,
                                      WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2)
        self.ball.reset()
        self.player_score = 0
        self.opponent_score = 0
        self.game_over = False
        return self.get_state()

    def close(self):
        pygame.quit()
