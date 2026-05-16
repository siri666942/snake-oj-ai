from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import NS, SnakeEnv
from .features import mlp_features
from .teacher import choose_fast_teacher_action, choose_teacher_action


def episode_value_samples(env: SnakeEnv, policy) -> tuple[list[np.ndarray], list[int], int]:
    features: list[np.ndarray] = []
    scores: list[int] = []
    while not env.done:
        action = policy(env)
        features.append(mlp_features(env))
        scores.append(env.score)
        env.step(action)
    return features, scores, env.score


def generate_value_data(samples: int, output: Path, seed: int, max_steps: int, teacher: str = "strong") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    xs_mlp: list[np.ndarray] = []
    ys_value: list[float] = []
    env = SnakeEnv(max_steps=max_steps)
    policy = choose_teacher_action if teacher == "strong" else choose_fast_teacher_action
    episode = 0
    while len(ys_value) < samples:
        n = NS[episode % len(NS)]
        env.reset(seed=seed + episode * 97, n=n)
        ep_features, ep_scores, final_score = episode_value_samples(env, policy)
        for x, score in zip(ep_features, ep_scores):
            if len(ys_value) >= samples:
                break
            xs_mlp.append(x)
            ys_value.append(float(final_score - score))
        episode += 1
    np.savez_compressed(
        output,
        x_mlp=np.stack(xs_mlp),
        y_value=np.asarray(ys_value, dtype=np.float32),
    )
    print(f"saved {len(ys_value)} value samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate state value-regression data.")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("strategies/dl_rl/data/value_50k.npz"))
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--teacher", choices=["strong", "fast"], default="strong")
    args = parser.parse_args()
    generate_value_data(args.samples, args.output, args.seed, args.max_steps, args.teacher)


if __name__ == "__main__":
    main()
