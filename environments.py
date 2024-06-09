import numpy as np
import torch


class Environment:
    def __init__(self):
        self.reward = 0

    def get_state(self, state):
        return state

    def get_reward(self, done, moved, score):
        self.reward = -100 if done else score if moved else -1
        return self.reward


class EnvironmentLog2(Environment):
    def get_state(self, state):
        state[state == 0] = 1
        return np.log2(state)

    def get_reward(self, done, moved, score):
        self.reward = -10 if done else np.log2(score + 1) if moved else -1
        return self.reward


class EnvironmentHotEncoding(Environment):
    def get_state(self, state):
        state[state == 0] = 1
        state = np.log2(state)

        one_hot_matrix = torch.eye(12)[state]
        return one_hot_matrix.flatten()

    def get_reward(self, done, moved, score):
        self.reward = -10 if done else np.log2(score) if score > 0 else 0
        return self.reward
