from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from .env import SnakeEnv
from .features import mlp_features
from .models import MLPValue

IMMEDIATE_DEATH_VALUE = -10000.0


def load_mlp_value(checkpoint: Path) -> Callable[[SnakeEnv], float]:
    data = torch.load(checkpoint, map_location="cpu")
    model = MLPValue(int(data["input_dim"]))
    model.load_state_dict(data["model"])
    model.eval()

    def value(env: SnakeEnv) -> float:
        with torch.no_grad():
            x = torch.from_numpy(mlp_features(env)).float().unsqueeze(0)
            return float(model(x).item())

    return value


def search_value(env: SnakeEnv, depth: int, value_fn: Callable[[SnakeEnv], float], death_penalty: float, gained_score: float = 0.0) -> float:
    if env.done:
        return gained_score - death_penalty
    if depth <= 0:
        return gained_score + value_fn(env)

    best = float("-inf")
    for action in range(4):
        nxt = env.clone()
        score_before = nxt.score
        _state, _reward, done, _info = nxt.step(action)
        immediate_gain = float(nxt.score - score_before)
        if done:
            value = gained_score + immediate_gain - death_penalty
        else:
            value = search_value(nxt, depth - 1, value_fn, death_penalty, gained_score + immediate_gain)
        best = max(best, value)
    return best


def make_value_search_policy(checkpoint: Path, depth: int, death_penalty: float = 50.0) -> Callable[[SnakeEnv], int]:
    value_fn = load_mlp_value(checkpoint)

    def policy(env: SnakeEnv) -> int:
        best_action = 0
        best_value = float("-inf")
        legal_fallback: int | None = None
        for action in range(4):
            nxt = env.clone()
            score_before = nxt.score
            _state, _reward, done, _info = nxt.step(action)
            if done:
                value = IMMEDIATE_DEATH_VALUE
            else:
                if legal_fallback is None:
                    legal_fallback = action
                immediate_gain = float(nxt.score - score_before)
                value = search_value(nxt, depth - 1, value_fn, death_penalty, immediate_gain)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action if best_value > IMMEDIATE_DEATH_VALUE else legal_fallback or best_action

    return policy
