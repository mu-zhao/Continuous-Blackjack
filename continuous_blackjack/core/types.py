from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoundRecord:
    """Immutable snapshot of a completed round."""

    winner_position: int
    order: tuple[int, ...]
    scores: tuple[float, ...]
    post_hands: tuple[tuple[float, ...], ...]

    @property
    def winner_id(self) -> int:
        return self.order[self.winner_position]
