from __future__ import annotations

from pathlib import Path

from .data import generate_teacher_data
from .evaluate import evaluate_policy
from .teacher import choose_fast_teacher_action
from .train_dqn import train_dqn
from .train_supervised import train_supervised
from .train_value import train_value
from .value_data import generate_value_data
from .value_search import make_value_search_policy


def main() -> None:
    root = Path("strategies/dl_rl")
    data_path = root / "data" / "smoke_teacher.npz"
    generate_teacher_data(samples=256, output=data_path, seed=20260516, max_steps=300, teacher="fast")
    train_supervised(data_path, "mlp", root / "checkpoints" / "smoke_mlp_supervised.pt", epochs=1, batch_size=32, lr=1e-3)
    train_supervised(data_path, "cnn", root / "checkpoints" / "smoke_cnn_supervised.pt", epochs=1, batch_size=32, lr=1e-3)
    value_data_path = root / "data" / "smoke_value.npz"
    value_ckpt = root / "checkpoints" / "smoke_mlp_value.pt"
    generate_value_data(samples=256, output=value_data_path, seed=20260516, max_steps=300, teacher="fast")
    train_value(value_data_path, "mlp", value_ckpt, epochs=1, batch_size=32, lr=1e-3)
    train_dqn("mlp", root / "checkpoints" / "smoke_mlp_dqn.pt", episodes=2, batch_size=8, lr=1e-3, gamma=0.99, buffer_size=200, target_update=20, seed=20260516, init_checkpoint=None)
    train_dqn("cnn", root / "checkpoints" / "smoke_cnn_dqn.pt", episodes=1, batch_size=4, lr=1e-3, gamma=0.99, buffer_size=100, target_update=20, seed=20260516, init_checkpoint=None)
    print(evaluate_policy(choose_fast_teacher_action, cases=10, seed=20260516, max_steps=500))
    print(evaluate_policy(make_value_search_policy(value_ckpt, depth=1), cases=3, seed=20260516, max_steps=200))
    print(evaluate_policy(make_value_search_policy(value_ckpt, depth=3), cases=3, seed=20260516, max_steps=200))


if __name__ == "__main__":
    main()
