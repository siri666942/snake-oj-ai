from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from .env import NS, SnakeEnv
from .features import cnn_tensor, mlp_features
from .teacher import choose_fast_teacher_action, choose_teacher_action


def _generate_arrays(samples: int, seed: int, max_steps: int, teacher: str, episode_offset: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_mlp: list[np.ndarray] = []
    xs_cnn: list[np.ndarray] = []
    ys: list[int] = []
    env = SnakeEnv(max_steps=max_steps)
    policy = choose_teacher_action if teacher == "strong" else choose_fast_teacher_action
    episode = episode_offset
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
    return np.stack(xs_mlp), np.stack(xs_cnn), np.asarray(ys, dtype=np.int64)


def _generate_shard(args: tuple[int, int, int, int, str, str]) -> str:
    worker_idx, samples, seed, max_steps, teacher, shard_path = args
    start = time.time()
    x_mlp, x_cnn, y = _generate_arrays(
        samples=samples,
        seed=seed + worker_idx * 1_000_003,
        max_steps=max_steps,
        teacher=teacher,
        episode_offset=worker_idx * 100_000,
    )
    np.savez(shard_path, x_mlp=x_mlp, x_cnn=x_cnn, y=y)
    print(f"worker={worker_idx} saved {len(y)} samples to {shard_path} in {time.time() - start:.1f}s", flush=True)
    return shard_path


def _sample_counts(total: int, workers: int) -> list[int]:
    base = total // workers
    rest = total % workers
    return [base + (1 if i < rest else 0) for i in range(workers)]


def generate_teacher_data(samples: int, output: Path, seed: int, max_steps: int, teacher: str = "strong", workers: int = 1) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers, samples))

    if workers == 1:
        x_mlp, x_cnn, y = _generate_arrays(samples, seed, max_steps, teacher)
        np.savez_compressed(output, x_mlp=x_mlp, x_cnn=x_cnn, y=y)
        print(f"saved {len(y)} samples to {output}")
        return

    shard_dir = output.parent / f"{output.stem}_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    counts = _sample_counts(samples, workers)
    jobs = [
        (i, count, seed, max_steps, teacher, str(shard_dir / f"shard_{i:03d}.npz"))
        for i, count in enumerate(counts)
        if count > 0
    ]

    print(f"generating {samples} samples with {len(jobs)} worker processes", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        shard_paths = list(pool.imap_unordered(_generate_shard, jobs))

    shard_paths = sorted(shard_paths)
    xs_mlp = []
    xs_cnn = []
    ys = []
    for shard_path in shard_paths:
        data = np.load(shard_path)
        xs_mlp.append(data["x_mlp"])
        xs_cnn.append(data["x_cnn"])
        ys.append(data["y"])

    y = np.concatenate(ys, axis=0)[:samples]
    x_mlp = np.concatenate(xs_mlp, axis=0)[:samples]
    x_cnn = np.concatenate(xs_cnn, axis=0)[:samples]
    np.savez_compressed(
        output,
        x_mlp=x_mlp,
        x_cnn=x_cnn,
        y=y,
    )
    print(f"saved {len(y)} samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher imitation-learning data.")
    parser.add_argument("--samples", type=int, default=300_000)
    parser.add_argument("--output", type=Path, default=Path("strategies/dl_rl/data/teacher_300k.npz"))
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--teacher", choices=["strong", "fast"], default="strong")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    generate_teacher_data(args.samples, args.output, args.seed, args.max_steps, args.teacher, args.workers)


if __name__ == "__main__":
    main()
