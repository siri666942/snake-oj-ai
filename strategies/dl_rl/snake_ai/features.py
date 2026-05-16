from __future__ import annotations

import numpy as np

from .env import ACTION_TO_DELTA, OPPOSITE, SIZE, SnakeEnv
from .teacher import bfs_distance, reachable_space


def cnn_tensor(env: SnakeEnv) -> np.ndarray:
    x = np.zeros((8, SIZE, SIZE), dtype=np.float32)
    tail = env.snake[-1]
    body = set(env.snake[1:])
    for r in range(SIZE):
        for c in range(SIZE):
            ch = env.base[r][c]
            if ch == "#":
                x[0, r, c] = 1.0
            elif ch == "O":
                x[1, r, c] = 1.0
    hr, hc = env.snake[0]
    x[2, hr, hc] = 1.0
    for r, c in body:
        x[3, r, c] = 1.0
    if env.food is not None:
        x[4, env.food[0], env.food[1]] = 1.0
    x[5, tail[0], tail[1]] = 1.0
    x[6, :, :] = env.cur_dir / 3.0
    x[7, :, :] = (env.steps % env.n) / max(env.n, 1)
    return x


def mlp_features(env: SnakeEnv) -> np.ndarray:
    values: list[float] = []
    for action in range(4):
        nxt = env.clone()
        ok, ate, _ = nxt._move_no_food_spawn(action)
        values.append(1.0 if ok else 0.0)
        if ok and nxt.food is not None:
            dist_food, _ = bfs_distance(nxt, nxt.food, True)
            values.append((dist_food if dist_food >= 0 else 80) / 80.0)
            space = reachable_space(nxt, True)
            values.append(space / 400.0)
            dist_tail, _ = bfs_distance(nxt, nxt.snake[-1], True)
            values.append(1.0 if dist_tail >= 0 else 0.0)
            values.append(1.0 if space < len(nxt.snake) + 5 else 0.0)
        elif ok:
            values.extend([0.0, reachable_space(nxt, True) / 400.0, 1.0, 0.0])
        else:
            values.extend([1.0, 0.0, 0.0, 1.0])
        values.append(1.0 if OPPOSITE[env.cur_dir] == action else 0.0)
        values.append(1.0 if ate else 0.0)

    head = env.snake[0]
    food = env.food or head
    values.extend(
        [
            len(env.snake) / 400.0,
            env.score / 1000.0,
            env.steps / 1000.0,
            (env.steps % env.n) / max(env.n, 1),
            env.n / 512.0,
        ]
    )
    values.extend([1.0 if env.cur_dir == i else 0.0 for i in range(4)])
    values.extend([(food[0] - head[0]) / 20.0, (food[1] - head[1]) / 20.0])
    values.append((abs(food[0] - head[0]) + abs(food[1] - head[1])) / 40.0)
    return np.asarray(values, dtype=np.float32)

