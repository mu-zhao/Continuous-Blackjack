from __future__ import annotations

import argparse
from pathlib import Path
import re

from continuous_blackjack.core import ContinuousBlackjackGame
from continuous_blackjack.strategies import (
    AdaptiveNashEquilibriumStrategy,
    AdaptiveStrategy,
    EpsilonGreedyBanditStrategy,
    NaiveStrategy,
    NashEquilibriumStrategy,
    PolicyGradientBanditStrategy,
    StatisticalStrategy,
    UCBBanditStrategy,
    UniformStrategy,
    ZeroIntelligenceStrategy,
)


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def _checkpoint_path(model_dir: Path, player_id: int, strategy_name: str) -> Path:
    return model_dir / f"player_{player_id:02d}_{_safe_name(strategy_name)}.pt"


def maybe_load_models(game: ContinuousBlackjackGame, model_dir: str | None) -> None:
    if not model_dir:
        return
    model_path = Path(model_dir)
    loaded = 0
    for player_id, strategy in enumerate(game.players):
        if not hasattr(strategy, "load_checkpoint"):
            continue
        checkpoint_path = _checkpoint_path(model_path, player_id, strategy.name)
        if checkpoint_path.exists():
            strategy.load_checkpoint(checkpoint_path)
            loaded += 1
    print(f"loaded checkpoints: {loaded} from {model_path}")


def maybe_save_models(game: ContinuousBlackjackGame, model_dir: str | None) -> None:
    if not model_dir:
        return
    model_path = Path(model_dir)
    saved = 0
    for player_id, strategy in enumerate(game.players):
        if not hasattr(strategy, "save_checkpoint"):
            continue
        checkpoint_path = _checkpoint_path(model_path, player_id, strategy.name)
        strategy.save_checkpoint(checkpoint_path)
        saved += 1
    print(f"saved checkpoints: {saved} to {model_path}")


def build_strategy_sets():
    ne = NashEquilibriumStrategy()
    zi = ZeroIntelligenceStrategy()
    naive = NaiveStrategy()
    adaptive_ne = AdaptiveNashEquilibriumStrategy()
    uninformed = UniformStrategy("uninformed")
    informed = UniformStrategy("informed")
    adaptive = AdaptiveStrategy()
    statistical = StatisticalStrategy()
    ucb = UCBBanditStrategy(confidence_level=3)
    greedy = EpsilonGreedyBanditStrategy(
        exploration_rate=0.1,
        exploration_decay=0.99,
        exploration_decay_rounds=10_000,
    )
    policy_gradient = PolicyGradientBanditStrategy(baseline=1.0, learning_rate=0.01)

    strategy_sets = [
        [ne, naive, uninformed, informed, statistical],
        [ne, uninformed, uninformed, ne, adaptive_ne],
        [ne, uninformed, informed, ne, greedy],
        [ne, zi, zi, ne, adaptive_ne],
        [ne, zi, naive, uninformed, ucb],
        [ne, uninformed, informed, naive, policy_gradient],
        [ne, uninformed, zi, greedy, adaptive],
    ]

    try:
        from continuous_blackjack.rl import ActorCriticStrategy, DQNStrategy

        dqn = DQNStrategy(action_bins=1024, lr=0.001)
        actor_critic = ActorCriticStrategy(
            action_bins=1024,
            actor_lr=0.001,
            critic_lr=0.001,
        )
        strategy_sets.extend(
            [
                [ne, uninformed, zi, ucb, dqn],
                [ne, uninformed, zi, greedy, actor_critic],
                [ne, zi, zi, zi, actor_critic],
                [ne, informed, ucb, statistical, dqn],
            ]
        )
    except Exception:
        # RL dependencies are optional for non-RL experiments.
        pass

    return strategy_sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Continuous Blackjack simulation")
    parser.add_argument("--set-index", type=int, default=1, help="strategy set index")
    parser.add_argument(
        "--actor-critic",
        action="store_true",
        help="run the first strategy set that includes ActorCriticStrategy",
    )
    parser.add_argument("--blocks", type=int, default=100, help="number of blocks")
    parser.add_argument("--rounds-per-block", type=int, default=10_000, help="rounds per block")
    parser.add_argument("--log", action="store_true", help="enable block logging")
    parser.add_argument(
        "--load-model-dir",
        type=str,
        default=None,
        help="directory containing model checkpoints to load for RL strategies",
    )
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default=None,
        help="directory where model checkpoints will be saved for RL strategies",
    )
    args = parser.parse_args()

    strategy_sets = build_strategy_sets()
    if args.actor_critic:
        actor_critic_idx = next(
            (
                i
                for i, strategies in enumerate(strategy_sets)
                if any(strategy.__class__.__name__ == "ActorCriticStrategy" for strategy in strategies)
            ),
            None,
        )
        if actor_critic_idx is None:
            raise RuntimeError(
                "No ActorCritic strategy set is available. Install RL dependencies first."
            )
        args.set_index = actor_critic_idx

    if args.set_index < 0 or args.set_index >= len(strategy_sets):
        raise ValueError(f"--set-index must be in [0, {len(strategy_sets)-1}]")

    game = ContinuousBlackjackGame(strategy_sets[args.set_index])
    maybe_load_models(game, args.load_model_dir)
    game.run(num_blocks=args.blocks, rounds_per_block=args.rounds_per_block, log=args.log)
    maybe_save_models(game, args.save_model_dir)
    summary = game.summary()
    print(f"total rounds: {int(summary['total_rounds'])}")
    print("cumulative reward by player:")
    for row in summary["player_summary"]:
        print(f"  {row['label']}: wins={row['wins']:.0f}, win_rate={row['win_rate']:.3%}")
    print("reward by strategy:")
    for row in summary["strategy_summary"]:
        print(
            f"  {row['strategy']}: wins={row['wins']:.0f}, "
            f"win_rate={row['win_rate']:.3%}, seats={row['seats']}"
        )
    game.post_game_analysis()


if __name__ == "__main__":
    main()
