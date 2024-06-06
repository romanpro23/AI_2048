import random
from collections import deque

import numpy as np
import torch
from torch import nn, optim


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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        state = torch.FloatTensor(np.stack([batch[0] for batch in minibatch]))
        action = np.stack([batch[1] for batch in minibatch])
        reward = torch.FloatTensor(np.stack([batch[2] for batch in minibatch]))
        next_state = np.stack([batch[3] for batch in minibatch])
        done = np.stack([batch[4] for batch in minibatch])

        done = torch.FloatTensor(done)
        target = (
                reward + (1 - done) * self.gamma * torch.max(self.target_model(torch.FloatTensor(next_state)),
                                                             dim=1).values
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

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)