import random
from collections import deque

import numpy as np
import torch
from torch import nn, optim

from agent import Agent


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent(Agent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory_len = 10000
        self.memory = deque(maxlen=self.memory_len)

        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = DQN(state_size, action_size)

        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.counter_target_update = 0
        self.update_frequency = 10

        self.batch_size = 32

        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_hyperparams(self):
        return {
            "Discount Factor": f"{self.gamma:.2f}",
            "Exploration Rate": f"{self.epsilon:.2f}",
            "Learning Rate": f"{self.learning_rate}",
            "Memory size": f"{self.memory_len}"
        }

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) > self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)

            state = torch.FloatTensor(np.stack([batch[0] for batch in minibatch]))
            action = np.stack([batch[1] for batch in minibatch])
            reward = torch.FloatTensor(np.stack([batch[2] for batch in minibatch]))
            next_state = np.stack([batch[3] for batch in minibatch])
            done = np.stack([batch[4] for batch in minibatch])

            done = torch.FloatTensor(done)
            target = (
                    reward + (1 - done) * self.gamma * torch.max(
                        self.target_model(torch.FloatTensor(next_state)).detach(), dim=1).values
            )
            target_f = self.model(torch.FloatTensor(state))

            for i in range(len(action)):
                target_f[i][action[i]] = target[i]

            self.optimizer.zero_grad()
            output = self.model(torch.FloatTensor(state))
            loss = nn.MSELoss()(output, target_f)
            loss.backward()

            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.counter_target_update += 1
            if self.counter_target_update % self.update_frequency == 0:
                self.update_target_model()
                self.counter_target_update = 0

    def get_weights(self):
        weights = [param.data for param in self.model.parameters()]
        return weights

    def load(self, name):
        self.model.load_state_dict(torch.load(f"{name}.pth"))

    def save(self, name):
        torch.save(self.model.state_dict(), f"{name}.pth")

    def get_name(self):
        return "DQN"


class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super(DoubleDQNAgent, self).__init__(state_size, action_size)

    def replay(self):
        if len(self.memory) > self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)

            state = torch.FloatTensor(np.stack([batch[0] for batch in minibatch]))
            action = np.stack([batch[1] for batch in minibatch])
            reward = torch.FloatTensor(np.stack([batch[2] for batch in minibatch]))
            next_state = np.stack([batch[3] for batch in minibatch])
            done = np.stack([batch[4] for batch in minibatch])

            done = torch.FloatTensor(done)
            next_q_values = self.model(torch.FloatTensor(next_state))
            next_q_state_values = self.target_model(torch.FloatTensor(next_state))

            best_actions = torch.argmax(next_q_values, dim=1)
            target = reward + (1 - done) * self.gamma * next_q_state_values[np.arange(self.batch_size), best_actions]
            target_f = self.model(torch.FloatTensor(state))

            for i in range(len(action)):
                target_f[i][action[i]] = target[i]

            self.optimizer.zero_grad()
            output = self.model(torch.FloatTensor(state))
            loss = nn.MSELoss()(output, target_f)
            loss.backward()

            self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.counter_target_update += 1
            if self.counter_target_update % self.update_frequency == 0:
                self.update_target_model()
                self.counter_target_update = 0

    def get_name(self):
        return "DoubleDQN"
