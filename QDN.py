
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nn_utils import NNStrategy, ReplayMemory, Experience,\
    generate_state_feature


class DqnNN(nn.Module):
    def __init__(self, input_size, output_size, lr):
        super(DqnNN, self).__init__()
        self.dqn = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, state):
        return self.dqn(state)

    def reinforce(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQN:
    def __init__(self, output_size, lr, buffer_size, batch_size,
                 xp, xp_deay, xp_decay_rounds):
        """ No need for target net as every state is terminal.
        """
        self.size = output_size
        self.lr = lr
        self.memory = ReplayMemory(buffer_size)
        self.batch_size = batch_size
        self.rounds_done = 0
        self.xp = xp
        self.xp_decay = xp_deay
        self.xp_decay_rounds = xp_decay_rounds
        self.last_state_action = None
        self.mse = nn.MSELoss()

    def initialize(self, num):
        input_size = num + num**2 + 8
        self.policy_net = DqnNN(input_size, self.size, self.lr).double()

    def action(self, state):
        self.rounds_done += 1
        if self.rounds_done == self.xp_decay_rounds:
            self.xp *= self.xp_decay
            self.rounds_done = 0
        if np.random.random() > self.xp:
            with torch.no_grad():
                decision = torch.argmax(self.policy_net(state)).item()
        else:
            low_index = int(state[0].item() * self.size)
            decision = np.random.randint(low_index, self.size)
        self.last_state_action = [state, decision]
        return decision

    def process(self, last_reward):
        if self.last_state_action:
            self.memory.push(Experience(*self.last_state_action, last_reward))
        if len(self.memory) < self.batch_size:
            return
        experiecne = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiecne))
        batch_state = torch.stack(batch.state).double()
        batch_action = torch.tensor(batch.action)
        batch_reward = torch.tensor(batch.reward).reshape(-1, 1).double()
        expected_value = self.policy_net(
            batch_state).gather(1, batch_action.view(-1, 1))
        loss = self.mse(batch_reward, expected_value)
        self.policy_net.reinforce(loss)


class DQNStrategy(NNStrategy):
    def __init__(self, output_size, lr, buffer_size=1000, batch_size=64,
                 xp=0.1, xp_deay=0.99, xp_decay_rounds=10000):
        super().__init__(DQN, generate_state_feature, output_size, lr,
                         buffer_size, batch_size, xp, xp_deay, xp_decay_rounds)
