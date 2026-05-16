from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import NS, SnakeEnv
from .features import cnn_tensor, mlp_features
from .teacher import choose_fast_teacher_action, choose_teacher_action


def generate_teacher_data(samples: int, output: Path, seed: int, max_steps: int, teacher: str = "strong") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    xs_mlp: list[np.ndarray] = []
    xs_cnn: list[np.ndarray] = []
    ys: list[int] = []
    env = SnakeEnv(max_steps=max_steps)
    policy = choose_teacher_action if teacher == "strong" else choose_fast_teacher_action
    episode = 0
    while len(ys) < samples:
        n = NS[episode % len(NS)]
        env.reset(seed=seed + episode * 97, n=n)
        while not env.done and len(ys) < samples:
            action = policy(env)
            xs_mlp.append(mlp_features(env))
            xs_cnn.append(cnn_tensor(env))
            ys.append(action)
            env.step(action)
        episode += 1
    np.savez_compressed(
        output,
        x_mlp=np.stack(xs_mlp),
        x_cnn=np.stack(xs_cnn),
        y=np.asarray(ys, dtype=np.int64),
    )
    print(f"saved {len(ys)} samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher imitation-learning data.")
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--output", type=Path, default=Path("strategies/dl_rl/data/teacher_300k.npz"))
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--teacher", choices=["strong", "fast"], default="strong")
    args = parser.parse_args()
    generate_teacher_data(args.samples, args.output, args.seed, args.max_steps, args.teacher)


if __name__ == "__main__":
    main()
