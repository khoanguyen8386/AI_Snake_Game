import torch
import random
import numpy as np
from menu import MainMenu
from collections import deque
from snake_game_env import SnakeGameAI, SnakeGameVersus, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, training_mode=True):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.training_mode = training_mode
        self.loaded_record = 0 

        loaded, old_games, old_record = self.model.load() 
        
        if loaded:
            if training_mode:
                self.n_games = old_games 
                self.loaded_record = old_record 
                print(f">>> Continue training from the game: {self.n_games} <<<")
            else:
                self.n_games = 0 
        else:
            print(">>> New Training... <<<")

    def get_state(self, game):
        return self._calculate_state(game.head, game.snake, game.food, game.direction, game)

    def get_versus_state(self, game):
        return self._calculate_state(game.head2, game.snake2, game.food2, game.direction2, game, is_versus=True)

    def _calculate_state(self, head, snake, food, direction, game, is_versus=False):
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        if is_versus:
            check_col = game.is_collision_ai
        else:
            check_col = game.is_collision

        state = [
            (dir_r and check_col(point_r)) or 
            (dir_l and check_col(point_l)) or 
            (dir_u and check_col(point_u)) or 
            (dir_d and check_col(point_d)),

            (dir_u and check_col(point_r)) or 
            (dir_d and check_col(point_l)) or 
            (dir_l and check_col(point_u)) or 
            (dir_r and check_col(point_d)),

            (dir_d and check_col(point_r)) or 
            (dir_u and check_col(point_l)) or 
            (dir_r and check_col(point_u)) or 
            (dir_l and check_col(point_d)),
            
            dir_l, dir_r, dir_u, dir_d,
            food.x < head.x, 
            food.x > head.x, 
            food.y < head.y, 
            food.y > head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        if self.training_mode:
            self.epsilon = 80 - self.n_games
        else:
            self.epsilon = 0
            
        final_move = [0,0,0]
        if self.training_mode and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train_ai(mode_choice):
    agent = Agent(training_mode=True if mode_choice == '2' else False)
    game = SnakeGameAI()
    
    record = agent.loaded_record 
    
   
    plot_scores = [] 
    plot_mean_scores = []
    total_score = 0
    current_session_games = 0 
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if agent.training_mode:
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            current_session_games += 1 
            
            if agent.training_mode:
                agent.train_long_memory()
                
                if score > record:
                    record = score
                    agent.model.save(n_games=agent.n_games, record=record) 
                    print(f">>> New record: {record} (Saved) <<<")
                
                elif agent.n_games % 10 == 0:
                    agent.model.save(n_games=agent.n_games, record=record)
                    print(">>> Auto Save <<<")

            print(f'Game {agent.n_games} Score {score} Record {record}')
            
            if agent.training_mode:
                plot_scores.append(score)
                total_score += score
                
                mean_score = total_score / current_session_games
                
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

def play_versus():
    agent = Agent(training_mode=False) 
    game = SnakeGameVersus() 
    
    print("--- STARTING VERSUS MODE ---")
    print("You: Arrow Keys | AI: Auto")
    
    while True:
        state_ai = agent.get_versus_state(game)
        action_ai = agent.get_action(state_ai)
        score_human, score_ai = game.play_step(action_ai)

def reset_data():
    path = './model/model.pth'
    if os.path.exists(path):
        try:
            os.remove(path)
            print(">>> Data cleared! The AI is starting over from zero. <<<")
        except Exception as e:
            print(f"Error can not delete the file: {e}")
    else:
        print(">>> There is no data to delete <<<")

if __name__ == '__main__':
    while True:
        menu = MainMenu()
        choice = menu.run() 
        
      
        if choice == '1':
            train_ai('1')
        elif choice == '2':
            train_ai('2')
        elif choice == '3':
            play_versus()
        elif choice == '4':
           
            print("\n" + "="*30)
            confirm = input(">>> WARNING: Do you want to completely erase the AI's data? (y/n): ")
            if confirm.lower() == 'y':
                reset_data()
            print("="*30 + "\n")