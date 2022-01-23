
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from nn_utils import NNStrategy, ReplayMemory,\
    PastEps, generate_state_feature


class ActorNN(nn.Module):
    def __init__(self, input_size, output_size, lr) -> None:
        super(ActorNN, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=0)
        )
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, state):
        action_prob = self.actor(state)
        actions = Categorical(action_prob)
        action = actions.sample()
        return (action.item(), actions.log_prob(action).clamp(-1000, 0),
                actions.entropy())

    def reinforce(self, loss):
        self.optimizer.zero_grad()
        # perform backprop
        loss.backward()
        self.optimizer.step()


class CriticNN(nn.Module):
    def __init__(self, input_size, lr) -> None:
        super(CriticNN, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, state):
        return self.critic(state)

    def reinforce(self, loss):
        self.optimizer.zero_grad()
        # perform backprop
        loss.backward()
        self.optimizer.step()


class ActorCritic():
    """
    implements both actor and critic in one model
    """
    def __init__(self, output_size, alpha, beta,
                 batch_size, memory_size):
        self.output_size = output_size
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)
        self.mse = nn.MSELoss()
        self.last_record = None

    def initialize(self, num):
        input_size = 8 + num + num**2
        self.actor = ActorNN(input_size, self.output_size, self.alpha).double()
        self.critic = CriticNN(input_size, self.beta).double()

    def process(self, last_reward):
        if self.last_record:
            self.memory.push(PastEps(*self.last_record, last_reward))
        if len(self.memory) < self.batch_size:
            return
        batch_experience = self.memory.sample(self.batch_size)
        batch = PastEps(*zip(*batch_experience))
        entropy_batch = torch.tensor(batch.entropy, requires_grad=True)
        state_batch = torch.stack(batch.state)
        reward_batch = torch.tensor(batch.reward).double()
        log_prob_batch = torch.tensor(batch.log_prob, requires_grad=True)
        # State values from critic
        state_values = self.critic(state_batch)
        # value(critic) loss
        value_loss = self.mse(state_values, reward_batch.reshape(-1, 1))
        # the policy gradeint loss function of policy(actor) is
        #  -loglikihood * advantage,
        # (since we want to maximize), whose gradeint is exactly the
        #  policy gradient.
        advantage = reward_batch - state_values.detach()
        policy_gredient_loss = -(log_prob_batch * advantage).mean()
        # entropy loss
        entropy_loss = -entropy_batch.mean()
        # policy(actor) loss
        policy_loss = policy_gredient_loss + entropy_loss
        self.critic.reinforce(value_loss)
        self.actor.reinforce(policy_loss)

    def action(self, state):
        decision, log_prob, entropy = self.actor(state)
        self.last_record = [state, log_prob, entropy]
        return decision


class ACStrategy(NNStrategy):
    def __init__(self, output_size, alpha, beta,
                 batch_size=64, memory_size=1000):
        super().__init__(ActorCritic, generate_state_feature,
                         output_size, alpha, beta, batch_size, memory_size)
