import numpy as np
import torch
from torch import nn, optim

from agent import Agent


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCriticAgent(Agent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.9
        self.learning_rate = 0.001

        self.actor = ActorNetwork(state_size, action_size)
        self.critic = CriticNetwork(state_size)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.memory = []

    def get_hyperparams(self):
        return {
            "Discount Factor": f"{self.gamma:.2f}",
            "Learning Rate": f"{self.learning_rate}"
        }

    def get_name(self):
        return "Actor-critic"

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def replay(self, *args):
        # Convert memory to batches
        states = torch.FloatTensor(np.stack([item[0] for item in self.memory]))
        actions = torch.LongTensor(np.stack([item[1] for item in self.memory]))
        rewards = torch.FloatTensor(np.stack([item[2] for item in self.memory]))
        next_states = torch.FloatTensor(np.stack([item[3] for item in self.memory]))
        dones = torch.FloatTensor(np.stack([item[4] for item in self.memory]))

        # Critic update
        values = self.critic(states).squeeze()
        values = values.unsqueeze(0) if len(values.shape) == 0 else values

        next_values = self.critic(next_states).squeeze()
        next_values = next_values.unsqueeze(0) if len(next_values.shape) == 0 else next_values

        targets = rewards + (1 - dones) * self.gamma * next_values
        critic_loss = nn.MSELoss()(values, targets.detach())

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor update
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs[range(len(actions)), actions])
        actor_loss = -(log_probs * values.detach()).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.memory = []

    def get_weights(self):
        actor_weights = [param.data.cpu() for param in self.actor.parameters()]
        critic_weights = [param.data.cpu() for param in self.critic.parameters()]

        all_weights = actor_weights + critic_weights
        return all_weights

    def load(self, name):
        self.actor.load_state_dict(torch.load(f"{name}_act.pth"))
        self.critic.load_state_dict(torch.load(f"{name}_crit.pth"))

    def save(self, name):
        torch.save(self.actor.state_dict(), f"{name}_act.pth")
        torch.save(self.critic.state_dict(), f"{name}_crit.pth")


class A2CAgent(ActorCriticAgent):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)

    def get_name(self):
        return "AdvantageAC"

    def replay(self, *args):
        # Convert memory to batches
        states = torch.FloatTensor(np.stack([item[0] for item in self.memory]))
        actions = torch.LongTensor(np.stack([item[1] for item in self.memory]))
        rewards = torch.FloatTensor(np.stack([item[2] for item in self.memory]))
        next_states = torch.FloatTensor(np.stack([item[3] for item in self.memory]))
        dones = torch.FloatTensor(np.stack([item[4] for item in self.memory]))

        # Critic update
        values = self.critic(states).squeeze()
        values = values.unsqueeze(0) if len(values.shape) == 0 else values

        next_values = self.critic(next_states).squeeze()
        next_values = next_values.unsqueeze(0) if len(next_values.shape) == 0 else next_values

        targets = rewards + (1 - dones) * self.gamma * next_values
        critic_loss = nn.MSELoss()(values, targets.detach())

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor update
        advantages = (targets - values).detach()
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs[range(len(actions)), actions])
        actor_loss = -(log_probs * advantages).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.memory = []
