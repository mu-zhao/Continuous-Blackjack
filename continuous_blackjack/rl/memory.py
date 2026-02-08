from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class Experience:
    state: object
    action: int
    reward: float


@dataclass(frozen=True)
class PastExperience:
    state: object
    reward: float


class ReplayMemory(Generic[T]):
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._memory: deque[T] = deque(maxlen=capacity)

    def push(self, item: T) -> None:
        self._memory.append(item)

    def sample(self, batch_size: int) -> list[T]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return random.sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)
