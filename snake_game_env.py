import pygame
import random
import numpy as np

from enum import Enum
from collections import namedtuple

pygame.init()
# Using SysFont to avoid file path errors
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 200, 0)

BLOCK_SIZE = 20
SPEED = 80 # Adjust speed here (20 is slow, 40 is standard, 100 is fast)

# --- CLASS 1: STANDARD AI ENVIRONMENT ---
class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI Training')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)

# --- CLASS 2: VERSUS MODE (Human vs AI) ---
class SnakeGameVersus:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Screen width is doubled + 20px border
        self.display = pygame.display.set_mode((self.w * 2 + 20, self.h))
        pygame.display.set_caption('Human (Left) vs AI (Right)')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.reset()

    def reset(self):
        # --- PLAYER 1 (HUMAN) ---
        self.direction1 = Direction.RIGHT
        self.head1 = Point(self.w / 2, self.h / 2)
        self.snake1 = [self.head1, Point(self.head1.x - BLOCK_SIZE, self.head1.y), Point(self.head1.x - (2 * BLOCK_SIZE), self.head1.y)]
        self.score1 = 0
        self.food1 = None
        self._place_food(1)
        
        # --- PLAYER 2 (AI) ---
        self.direction2 = Direction.RIGHT
        self.head2 = Point(self.w / 2, self.h / 2)
        self.snake2 = [self.head2, Point(self.head2.x - BLOCK_SIZE, self.head2.y), Point(self.head2.x - (2 * BLOCK_SIZE), self.head2.y)]
        self.score2 = 0
        self.food2 = None
        self._place_food(2)
        
    def _place_food(self, player_id):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food = Point(x, y)
        if player_id == 1:
            if food in self.snake1: self._place_food(1)
            else: self.food1 = food
        else:
            if food in self.snake2: self._place_food(2)
            else: self.food2 = food

    def is_collision_ai(self, pt=None):
        if pt is None: pt = self.head2
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake2[1:]:
            return True
        return False

    def play_step(self, action_ai):
        # 1. User Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction1 != Direction.RIGHT:
                    self.direction1 = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction1 != Direction.LEFT:
                    self.direction1 = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction1 != Direction.DOWN:
                    self.direction1 = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction1 != Direction.UP:
                    self.direction1 = Direction.DOWN

        # 2. Move
        self._move_human()
        self._move_ai(action_ai)
        
        self.snake1.insert(0, self.head1)
        self.snake2.insert(0, self.head2)
        
        # 3. Game Over & Food Logic
        # Human
        game_over_1 = False
        if (self.head1.x > self.w - BLOCK_SIZE or self.head1.x < 0 or 
            self.head1.y > self.h - BLOCK_SIZE or self.head1.y < 0 or 
            self.head1 in self.snake1[1:]):
            game_over_1 = True
            
        if self.head1 == self.food1:
            self.score1 += 1
            self._place_food(1)
        else:
            self.snake1.pop()

        # AI
        game_over_2 = False
        if self.is_collision_ai():
            game_over_2 = True
            
        if self.head2 == self.food2:
            self.score2 += 1
            self._place_food(2)
        else:
            self.snake2.pop()

        if game_over_1:
            self.score1 = 0
            self._respawn(1)
        if game_over_2:
            self.score2 = 0
            self._respawn(2)
            
        # 4. UI
        self._update_ui()
        self.clock.tick(15) # Slower speed for human playability
        return self.score1, self.score2

    def _respawn(self, player_id):
        if player_id == 1:
            self.head1 = Point(self.w / 2, self.h / 2)
            self.snake1 = [self.head1, Point(self.head1.x-20, self.head1.y), Point(self.head1.x-40, self.head1.y)]
            self._place_food(1)
        else:
            self.head2 = Point(self.w / 2, self.h / 2)
            self.snake2 = [self.head2, Point(self.head2.x-20, self.head2.y), Point(self.head2.x-40, self.head2.y)]
            self._place_food(2)

    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw Human (Left)
        for pt in self.snake1:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food1.x, self.food1.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw Separator
        pygame.draw.rect(self.display, WHITE, pygame.Rect(self.w, 0, 20, self.h))

        # Draw AI (Right)
        OFFSET = self.w + 20
        for pt in self.snake2:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x + OFFSET, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + OFFSET+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food2.x + OFFSET, self.food2.y, BLOCK_SIZE, BLOCK_SIZE))

        text1 = self.font.render("Human: " + str(self.score1), True, WHITE)
        text2 = self.font.render("AI: " + str(self.score2), True, WHITE)
        self.display.blit(text1, [0, 0])
        self.display.blit(text2, [OFFSET, 0])
        pygame.display.flip()

    def _move_human(self):
        x = self.head1.x
        y = self.head1.y
        if self.direction1 == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction1 == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction1 == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction1 == Direction.UP: y -= BLOCK_SIZE
        self.head1 = Point(x, y)

    def _move_ai(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction2)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction2 = new_dir
        x = self.head2.x
        y = self.head2.y
        if self.direction2 == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction2 == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction2 == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction2 == Direction.UP: y -= BLOCK_SIZE
        self.head2 = Point(x, y)