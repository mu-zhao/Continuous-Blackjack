from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from continuous_blackjack.core.strategy import BaseStrategy
from continuous_blackjack.core.types import RoundRecord
from continuous_blackjack.rl.features import generate_state_feature
from continuous_blackjack.rl.memory import PastExperience, ReplayMemory


class ActorNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, lr: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, state: torch.Tensor) -> tuple[int, torch.Tensor, torch.Tensor]:
        action_prob = self.network(state)
        distribution = Categorical(action_prob)
        action = distribution.sample()
        return int(action.item()), distribution.log_prob(action).clamp(-1e6, 0), distribution.entropy()

    def reinforce(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CriticNetwork(nn.Module):
    def __init__(self, input_size: int, lr: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def reinforce(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ActorCriticAgent:
    def __init__(
        self,
        *,
        action_bins: int,
        actor_lr: float,
        critic_lr: float,
        batch_size: int = 64,
        memory_size: int = 1000,
        entropy_weight: float = 1.0,
    ) -> None:
        if action_bins <= 1:
            raise ValueError("action_bins must be greater than 1")
        if actor_lr <= 0 or critic_lr <= 0:
            raise ValueError("actor_lr and critic_lr must be positive")
        if batch_size <= 0 or memory_size <= 0:
            raise ValueError("batch_size and memory_size must be positive")

        self.action_bins = action_bins
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight

        self.memory = ReplayMemory[PastExperience](memory_size)
        self.mse = nn.MSELoss()

        self.actor: ActorNetwork | None = None
        self.critic: CriticNetwork | None = None
        self.last_record: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def checkpoint(self) -> dict[str, object]:
        if self.actor is None or self.critic is None:
            raise RuntimeError("ActorCriticAgent is not initialized")
        return {
            "agent_type": "actor_critic",
            "action_bins": self.action_bins,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "batch_size": self.batch_size,
            "entropy_weight": self.entropy_weight,
            "actor_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor.optimizer.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_optimizer_state_dict": self.critic.optimizer.state_dict(),
        }

    def restore(self, checkpoint: dict[str, object]) -> None:
        if self.actor is None or self.critic is None:
            raise RuntimeError("ActorCriticAgent is not initialized")
        checkpoint_action_bins = int(checkpoint.get("action_bins", -1))
        if checkpoint_action_bins != self.action_bins:
            raise ValueError(
                f"Checkpoint action_bins={checkpoint_action_bins} "
                f"does not match strategy action_bins={self.action_bins}"
            )

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic.optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    def initialize(self, num_players: int) -> None:
        input_size = 8 + num_players + num_players**2
        self.actor = ActorNetwork(input_size, self.action_bins, self.actor_lr).double()
        self.critic = CriticNetwork(input_size, self.critic_lr).double()
        self.last_record = None

    def action(self, state: torch.Tensor) -> int:
        if self.actor is None:
            raise RuntimeError("ActorCriticAgent is not initialized")
        decision, log_prob, entropy = self.actor(state)
        self.last_record = (state, log_prob, entropy)
        return decision

    def process(self, reward: float) -> None:
        if self.actor is None or self.critic is None:
            raise RuntimeError("ActorCriticAgent is not initialized")

        if self.last_record is not None:
            self.memory.push(PastExperience(self.last_record[0], reward))
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state_batch = torch.stack([e.state for e in batch]).double()
        reward_batch = torch.tensor([e.reward for e in batch], dtype=torch.double).view(-1, 1)

        state_values = self.critic(state_batch)
        value_loss = self.mse(state_values, reward_batch)
        self.critic.reinforce(value_loss)

        if self.last_record is None:
            return
        state, log_prob, entropy = self.last_record
        with torch.no_grad():
            advantage = reward - self.critic(state)
        policy_gradient_loss = -log_prob * advantage
        entropy_loss = -self.entropy_weight * entropy
        policy_loss = policy_gradient_loss + entropy_loss
        self.actor.reinforce(policy_loss)


class ActorCriticStrategy(BaseStrategy):
    def __init__(
        self,
        *,
        action_bins: int = 1024,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        batch_size: int = 64,
        memory_size: int = 1000,
        entropy_weight: float = 1.0,
    ) -> None:
        super().__init__(name=f"ActorCritic(bins={action_bins})")
        self.action_bins = action_bins
        self.agent = ActorCriticAgent(
            action_bins=action_bins,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            batch_size=batch_size,
            memory_size=memory_size,
            entropy_weight=entropy_weight,
        )
        self._processed_round: RoundRecord | None = None
        self._pending_checkpoint: dict[str, object] | None = None

    def set_parameters(self) -> None:
        self.agent.initialize(self.num_players)
        if self._pending_checkpoint is not None:
            self.agent.restore(self._pending_checkpoint)
            self._pending_checkpoint = None
        self._processed_round = None

    def calibrate(
        self,
        position: int,
        order,
        current_scores,
        current_round_hands,
        last_round: RoundRecord | None,
    ) -> None:
        table_max = max(current_scores) if current_scores else 0.0
        state = generate_state_feature(table_max, position, order)

        if last_round is not None and last_round is not self._processed_round:
            reward = 1.0 if last_round.winner_id == self.player_id else 0.0
            self.agent.process(reward)
            self._processed_round = last_round

        self._critical_value = self.agent.action(state) / self.action_bins

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.checkpoint(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str | Path, *, map_location: str = "cpu") -> None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if self.agent.actor is None or self.agent.critic is None:
            self._pending_checkpoint = checkpoint
        else:
            self.agent.restore(checkpoint)
