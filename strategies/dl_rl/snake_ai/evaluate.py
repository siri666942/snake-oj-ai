from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .env import ACTION_CHARS, NS, SnakeEnv
from .features import cnn_tensor, mlp_features
from .models import CNNPolicy, MLPPolicy
from .teacher import choose_fast_teacher_action, choose_teacher_action


def safe_policy_action(env: SnakeEnv, scores: np.ndarray) -> int:
    for action in np.argsort(scores)[::-1].tolist():
        if env.clone()._move_no_food_spawn(int(action))[0]:
            return int(action)
    legal = env.legal_actions()
    return legal[0] if legal else 0


def make_model_policy(kind: str, checkpoint: Path) -> Callable[[SnakeEnv], int]:
    data = torch.load(checkpoint, map_location="cpu")
    if kind == "mlp":
        model = MLPPolicy(int(data["input_dim"]))
        encoder = mlp_features
    elif kind == "cnn":
        model = CNNPolicy()
        encoder = cnn_tensor
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model.load_state_dict(data["model"])
    model.eval()

    def policy(env: SnakeEnv) -> int:
        with torch.no_grad():
            x = torch.from_numpy(encoder(env)).float().unsqueeze(0)
            scores = model(x).squeeze(0).numpy()
        return safe_policy_action(env, scores)

    return policy


def evaluate_policy(policy: Callable[[SnakeEnv], int], cases: int, seed: int, max_steps: int) -> dict:
    scores = []
    steps = []
    foods = []
    deaths = 0
    env = SnakeEnv(max_steps=max_steps)
    for i in range(cases):
        env.reset(seed=seed + i * 97, n=NS[i % len(NS)])
        while not env.done:
            env.step(policy(env))
        scores.append(env.score)
        steps.append(env.steps)
        foods.append(env.food_count)
        if env.death_reason != "max_steps":
            deaths += 1
    arr = np.asarray(scores, dtype=np.float32)
    return {
        "cases": cases,
        "avg_score": float(arr.mean()),
        "min_score": int(arr.min()),
        "score_std": float(arr.std()),
        "avg_steps": float(np.mean(steps)),
        "death_rate": deaths / cases,
        "food_count": float(np.mean(foods)),
        "final_metric": float(arr.mean() - 0.3 * arr.std() + 0.2 * arr.min()),
    }


def weighted_total(scores: list[int]) -> float:
    return sum(score * (1.0 / (math.log2(n) + 1.0)) for score, n in zip(scores, NS))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher or trained model policies.")
    parser.add_argument("--policy", choices=["teacher", "fast_teacher", "mlp", "cnn"], default="teacher")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--csv", type=Path, default=Path("strategies/dl_rl/results/eval.csv"))
    args = parser.parse_args()

    if args.policy == "teacher":
        policy = choose_teacher_action
    elif args.policy == "fast_teacher":
        policy = choose_fast_teacher_action
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for model policies")
        policy = make_model_policy(args.policy, args.checkpoint)

    metrics = evaluate_policy(policy, args.cases, args.seed, args.max_steps)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.csv.exists()
    with args.csv.open("a", newline="", encoding="utf-8") as f:
        row = {"policy": args.policy, "checkpoint": str(args.checkpoint or ""), **metrics}
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
