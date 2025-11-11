import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import deque
import random

class PolicyNet(nn.Module):
    def __init__(self, n_states, hidden_dim, actions_dim):
        super(PolicyNet, self).__init__()
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.action_dim = actions_dim
        self.fc1 = nn.Linear(self.n_states, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)
        self.fc3 = nn.Linear(self.action_dim, self.action_dim)
    def forward(self, x):
        """27
        x = self.fc1(x)
        x = F.softmax(x)
        x = self.fc2(x)
        x = F.softmax(x)
        """

        x = self.fc1(x)
        x = F.softmax(x)
        x = self.fc2(x)
        x = F.softmax(x)


        x = self.fc3(x)
        x = F.softmax(x)

        return x

class ValueNet(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(ValueNet, self).__init__()
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.n_states, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x)
        x = nn.Sigmoid()(self.fc(x))
        x = self.fc2(x)
        return x

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen = self.capacity)

    def add(self, state, act, reward, state_):
        one_try = (state, act, reward, state_)
        self.buffer.append(one_try)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, acts, rewards, state_s = zip(*batch)
        return states, acts, rewards, state_s

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class ActorCritic(nn.Module):
    def __init__(self, states_nums, hidden_dim, action_nums, lr, gamma, epsilon, batch_size):
        super(ActorCritic, self).__init__()
        self.states_nums = states_nums
        self.hidden_dim = hidden_dim
        self.action_nums = action_nums
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        # self.target_update_nums = target_update_nums
        self.actor = PolicyNet(self.states_nums, self.hidden_dim, self.action_nums)
        self.actor_optimizer = self.optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.lr)
        self.critic = ValueNet(self.states_nums, self.hidden_dim)
        self.critic_optimizer = self.optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.lr)
        self.loss_func = torch.nn.MSELoss()
        # self.memery = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.count = 0

    def choose_action(self, x):
        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                action_value = self.actor(x).tolist()[0]
                action = action_value.index(max(action_value))
        else:
            action = np.random.choice(2)
        return action

    def update(self, states, rewards, states_):
        td_value = self.critic(states)
        td_target = rewards + self.gamma * self.critic(states_)
        td_error = td_target - td_value
        action_prob = self.actor(states)
        actor_loss = torch.mean(-action_prob * td_error.detach())
        critic_loss = torch.mean(self.loss_func(self.critic(states), td_error.detach()))
        self.actor_optimizer.zero_grad()
        self.critic.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.critic_optimizer.step()