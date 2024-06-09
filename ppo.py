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


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent(Agent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.9
        self.learning_rate = 0.001
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.policy_model = PolicyNetwork(state_size, action_size)
        self.value_model = ValueNetwork(state_size)
        self.optimizer_policy = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate)
        self.optimizer_value = optim.Adam(self.value_model.parameters(), lr=self.learning_rate)
        self.memory = []

    def get_hyperparams(self):
        return {
            "Discount Factor": f"{self.gamma:.2f}",
            "Learning Rate": f"{self.learning_rate}",
            "Clip Epsilon": f"{self.eps_clip}",
            "Epochs": f"{self.K_epochs}"
        }

    def get_name(self):
        return "PPO"

    def remember(self, state, action, reward, *args):
        action_probs = self.policy_model(torch.FloatTensor(state))
        log_prob = torch.log(action_probs[action])
        self.memory.append((state, action, reward, log_prob))

    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.policy_model(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def replay(self):
        # Prepare batches
        states = torch.FloatTensor(np.stack([item[0] for item in self.memory]))
        actions = torch.LongTensor(np.stack([item[1] for item in self.memory]))
        old_log_probs = torch.FloatTensor([item[3] for item in self.memory]).detach()

        # Compute rewards-to-go
        R = 0
        rewards = []
        for _, _, r, _ in self.memory[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6) if len(rewards) > 1 else rewards

        for _ in range(self.K_epochs):
            action_probs = self.policy_model(states)
            new_log_probs = torch.log(action_probs.gather(1, actions.view(-1, 1)).squeeze())

            values = self.value_model(states).squeeze()
            values = values.unsqueeze(0) if len(values.shape) == 0 else values
            advantages = rewards - values.detach()

            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            value_loss = nn.MSELoss()(values, rewards)

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        self.memory = []

    def get_weights(self):
        actor_weights = [param.data.cpu() for param in self.policy_model.parameters()]
        critic_weights = [param.data.cpu() for param in self.value_model.parameters()]

        all_weights = actor_weights + critic_weights
        return all_weights

    def load(self, name):
        self.policy_model.load_state_dict(torch.load(f"{name}_pol.pth"))
        self.value_model.load_state_dict(torch.load(f"{name}_val.pth"))

    def save(self, name):
        torch.save(self.policy_model.state_dict(), f"{name}_pol.pth")
        torch.save(self.value_model.state_dict(), f"{name}_val.pth")
