from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .device import cpu_state_dict, resolve_device
from .env import NS, SnakeEnv
from .features import cnn_tensor, mlp_features
from .models import CNNPolicy, MLPPolicy


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.items = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.items.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.items, batch_size)
        return zip(*batch)

    def __len__(self) -> int:
        return len(self.items)


def train_dqn(
    kind: str,
    output: Path,
    episodes: int,
    batch_size: int,
    lr: float,
    gamma: float,
    buffer_size: int,
    target_update: int,
    seed: int,
    init_checkpoint: Path | None,
    device_name: str = "auto",
) -> None:
    device = resolve_device(device_name)
    print(f"device={device}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SnakeEnv()
    encoder = mlp_features if kind == "mlp" else cnn_tensor
    sample_dim = mlp_features(env.reset(seed=seed, n=32)).shape[0]
    q = MLPPolicy(sample_dim) if kind == "mlp" else CNNPolicy()
    target = MLPPolicy(sample_dim) if kind == "mlp" else CNNPolicy()
    q.to(device)
    target.to(device)
    if init_checkpoint is not None and init_checkpoint.exists():
        ckpt = torch.load(init_checkpoint, map_location=device)
        q.load_state_dict(ckpt["model"], strict=False)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(buffer_size)
    global_step = 0
    best_score = -1
    output.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, episodes + 1):
        env.reset(seed=seed + ep * 97, n=NS[ep % len(NS)])
        state = encoder(env).copy()
        epsilon = max(0.05, 1.0 - ep / max(episodes * 0.7, 1))
        while not env.done:
            if random.random() < epsilon:
                action = random.choice(env.legal_actions() or [0])
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    scores = q(state_t).squeeze(0).detach().cpu().numpy()
                legal = env.legal_actions()
                action = max(legal or [0], key=lambda a: scores[a])
            _, reward, done, _ = env.step(action)
            next_state = encoder(env).copy()
            replay.push(state, action, reward, next_state, done)
            state = next_state
            global_step += 1

            if len(replay) >= batch_size:
                states, actions, rewards, next_states, dones = replay.sample(batch_size)
                states_t = torch.from_numpy(np.stack(list(states))).float().to(device)
                actions_t = torch.tensor(list(actions), dtype=torch.long, device=device).unsqueeze(1)
                rewards_t = torch.tensor(list(rewards), dtype=torch.float32, device=device)
                next_t = torch.from_numpy(np.stack(list(next_states))).float().to(device)
                dones_t = torch.tensor(list(dones), dtype=torch.float32, device=device)
                q_values = q(states_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    target_values = rewards_t + gamma * target(next_t).max(dim=1).values * (1.0 - dones_t)
                loss = loss_fn(q_values, target_values)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if global_step % target_update == 0:
                target.load_state_dict(q.state_dict())

        if env.score > best_score:
            best_score = env.score
            torch.save({"model": cpu_state_dict(q), "kind": kind, "input_dim": sample_dim, "score": env.score}, output)
        if ep == 1 or ep % 100 == 0:
            print(f"episode={ep} score={env.score} epsilon={epsilon:.3f} best={best_score}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN/MLP DQN models.")
    parser.add_argument("--kind", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    train_dqn(args.kind, args.output, args.episodes, args.batch_size, args.lr, args.gamma, args.buffer_size, args.target_update, args.seed, args.init_checkpoint, args.device)


if __name__ == "__main__":
    main()
