import numpy as np
import torch
from torch import nn, optim

from agent import Agent


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class VPGAgent(Agent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.9
        self.learning_rate = 0.001
        self.model = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.memory = []

    def get_hyperparams(self):
        return {
            "Discount Factor": f"{self.gamma:.2f}",
            "Learning Rate": f"{self.learning_rate}"
        }

    def get_name(self):
        return "VPG"

    def remember(self, state, action, reward, *args):
        self.memory.append((state, action, reward))

    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def replay(self):
        R = 0
        rewards = []
        for r in self.memory[::-1]:
            R = r[2] + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6) if len(rewards) > 1 else rewards

        self.optimizer.zero_grad()

        for (state, action, reward), r in zip(self.memory, rewards):
            state = torch.FloatTensor(state)
            action = torch.tensor(action)
            r = torch.tensor(np.array(r))

            action_probs = self.model(state)
            log_prob = torch.log(action_probs[action])
            loss = -log_prob * r
            loss.backward()

        self.optimizer.step()
        self.memory = []

    def get_weights(self):
        weights = [param.data.cpu()for param in self.model.parameters()]
        return weights

    def load(self, name):
        self.model.load_state_dict(torch.load(f"{name}.pth"))

    def save(self, name):
        torch.save(self.model.state_dict(), f"{name}.pth")
