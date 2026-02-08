from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from continuous_blackjack.core.strategy import BaseStrategy
from continuous_blackjack.core.types import RoundRecord
from continuous_blackjack.rl.features import generate_state_feature
from continuous_blackjack.rl.memory import Experience, ReplayMemory


class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, lr: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid(),
        )
        self.optimizer = optim.Adam(self.parameters(), lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def reinforce(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQNAgent:
    """One-step DQN policy for terminal-state reward."""

    def __init__(
        self,
        *,
        action_bins: int,
        lr: float,
        buffer_size: int = 1000,
        batch_size: int = 64,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.99,
        exploration_decay_rounds: int = 10_000,
    ) -> None:
        if action_bins <= 1:
            raise ValueError("action_bins must be greater than 1")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= exploration_rate <= 1:
            raise ValueError("exploration_rate must be in [0, 1]")
        if not 0 < exploration_decay <= 1:
            raise ValueError("exploration_decay must be in (0, 1]")
        if exploration_decay_rounds <= 0:
            raise ValueError("exploration_decay_rounds must be positive")

        self.action_bins = action_bins
        self.lr = lr
        self.memory = ReplayMemory[Experience](buffer_size)
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_decay_rounds = exploration_decay_rounds
        self._rounds_done = 0
        self.last_state_action: tuple[torch.Tensor, int] | None = None
        self.mse = nn.MSELoss()
        self.policy_net: DQNNetwork | None = None

    def checkpoint(self) -> dict[str, object]:
        if self.policy_net is None:
            raise RuntimeError("DQNAgent is not initialized")
        return {
            "agent_type": "dqn",
            "action_bins": self.action_bins,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "exploration_rate": self.exploration_rate,
            "exploration_decay": self.exploration_decay,
            "exploration_decay_rounds": self.exploration_decay_rounds,
            "rounds_done": self._rounds_done,
            "policy_net_state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.policy_net.optimizer.state_dict(),
        }

    def restore(self, checkpoint: dict[str, object]) -> None:
        if self.policy_net is None:
            raise RuntimeError("DQNAgent is not initialized")
        checkpoint_action_bins = int(checkpoint.get("action_bins", -1))
        if checkpoint_action_bins != self.action_bins:
            raise ValueError(
                f"Checkpoint action_bins={checkpoint_action_bins} "
                f"does not match strategy action_bins={self.action_bins}"
            )

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.policy_net.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.exploration_rate = float(checkpoint.get("exploration_rate", self.exploration_rate))
        self._rounds_done = int(checkpoint.get("rounds_done", 0))

    def initialize(self, num_players: int) -> None:
        input_size = 8 + num_players + num_players**2
        self.policy_net = DQNNetwork(input_size, self.action_bins, self.lr).double()
        self._rounds_done = 0
        self.last_state_action = None

    def action(self, state: torch.Tensor) -> int:
        if self.policy_net is None:
            raise RuntimeError("DQNAgent is not initialized")

        self._rounds_done += 1
        if self._rounds_done >= self.exploration_decay_rounds:
            self.exploration_rate *= self.exploration_decay
            self._rounds_done = 0

        if np.random.random() > self.exploration_rate:
            with torch.no_grad():
                action = int(torch.argmax(self.policy_net(state)).item())
        else:
            low_idx = min(self.action_bins - 1, int(max(0.0, state[0].item()) * self.action_bins))
            action = int(np.random.randint(low_idx, self.action_bins))

        self.last_state_action = (state, action)
        return action

    def process(self, reward: float) -> None:
        if self.policy_net is None:
            raise RuntimeError("DQNAgent is not initialized")

        if self.last_state_action is not None:
            self.memory.push(Experience(self.last_state_action[0], self.last_state_action[1], reward))
        if len(self.memory) < self.batch_size:
            return

        experience_batch = self.memory.sample(self.batch_size)
        batch_state = torch.stack([e.state for e in experience_batch]).double()
        batch_action = torch.tensor([e.action for e in experience_batch], dtype=torch.long)
        batch_reward = torch.tensor([e.reward for e in experience_batch], dtype=torch.double).view(-1, 1)

        expected_value = self.policy_net(batch_state).gather(1, batch_action.view(-1, 1))
        loss = self.mse(batch_reward, expected_value)
        self.policy_net.reinforce(loss)


class DQNStrategy(BaseStrategy):
    def __init__(
        self,
        *,
        action_bins: int = 1024,
        lr: float = 1e-3,
        buffer_size: int = 1000,
        batch_size: int = 64,
        exploration_rate: float = 0.1,
        exploration_decay: float = 0.99,
        exploration_decay_rounds: int = 10_000,
    ) -> None:
        super().__init__(name=f"DQN(bins={action_bins})")
        self.action_bins = action_bins
        self.agent = DQNAgent(
            action_bins=action_bins,
            lr=lr,
            buffer_size=buffer_size,
            batch_size=batch_size,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            exploration_decay_rounds=exploration_decay_rounds,
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
        if self.agent.policy_net is None:
            self._pending_checkpoint = checkpoint
        else:
            self.agent.restore(checkpoint)
