from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import generate_teacher_data
from .evaluate import evaluate_policy, make_model_policy
from .export_mlp_c import export as export_mlp_c
from .teacher import choose_fast_teacher_action, choose_teacher_action
from .train_dqn import train_dqn
from .train_supervised import train_supervised


@dataclass
class PipelineConfig:
    preset: str
    seed: int
    teacher: str
    samples: int
    max_steps: int
    supervised_epochs: int
    supervised_batch_size: int
    supervised_lr: float
    dqn_episodes: int
    dqn_batch_size: int
    dqn_lr: float
    dqn_gamma: float
    dqn_buffer_size: int
    dqn_target_update: int
    eval_cases: int
    run_cnn: bool
    run_dqn: bool
    resume: bool


def preset_defaults(name: str) -> dict:
    if name == "smoke":
        return {
            "teacher": "fast",
            "samples": 256,
            "max_steps": 300,
            "supervised_epochs": 1,
            "dqn_episodes": 2,
            "eval_cases": 10,
            "dqn_batch_size": 8,
            "dqn_buffer_size": 200,
            "dqn_target_update": 20,
        }
    if name == "fast":
        return {
            "teacher": "fast",
            "samples": 50_000,
            "max_steps": 1000,
            "supervised_epochs": 5,
            "dqn_episodes": 5000,
            "eval_cases": 100,
            "dqn_batch_size": 64,
            "dqn_buffer_size": 50_000,
            "dqn_target_update": 1000,
        }
    if name == "full":
        return {
            "teacher": "strong",
            "samples": 300_000,
            "max_steps": 5000,
            "supervised_epochs": 10,
            "dqn_episodes": 50_000,
            "eval_cases": 300,
            "dqn_batch_size": 64,
            "dqn_buffer_size": 100_000,
            "dqn_target_update": 1000,
        }
    raise ValueError(f"unknown preset: {name}")


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def should_skip(path: Path, resume: bool) -> bool:
    return resume and path.exists()


def run_step(name: str, fn, *args, **kwargs) -> None:
    start = time.time()
    log(f"开始：{name}")
    fn(*args, **kwargs)
    log(f"完成：{name}，耗时 {time.time() - start:.1f}s")


def run_pytest(tests_dir: Path) -> None:
    subprocess.run([sys.executable, "-m", "pytest", str(tests_dir), "-q"], check=True)


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_config(args: argparse.Namespace) -> PipelineConfig:
    defaults = preset_defaults(args.preset)
    get = lambda key: getattr(args, key) if getattr(args, key) is not None else defaults[key]
    return PipelineConfig(
        preset=args.preset,
        seed=args.seed,
        teacher=get("teacher"),
        samples=get("samples"),
        max_steps=get("max_steps"),
        supervised_epochs=get("supervised_epochs"),
        supervised_batch_size=args.supervised_batch_size,
        supervised_lr=args.supervised_lr,
        dqn_episodes=get("dqn_episodes"),
        dqn_batch_size=get("dqn_batch_size"),
        dqn_lr=args.dqn_lr,
        dqn_gamma=args.dqn_gamma,
        dqn_buffer_size=get("dqn_buffer_size"),
        dqn_target_update=get("dqn_target_update"),
        eval_cases=get("eval_cases"),
        run_cnn=not args.skip_cnn,
        run_dqn=not args.skip_dqn,
        resume=args.resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command DL/RL snake training pipeline.")
    parser.add_argument("--preset", choices=["smoke", "fast", "full"], default="full", help="训练规模预设；默认 full。")
    parser.add_argument("--root", type=Path, default=Path("strategies/dl_rl"), help="dl_rl 输出目录。")
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--teacher", choices=["strong", "fast"], help="覆盖预设 teacher。")
    parser.add_argument("--samples", type=int, help="覆盖预设样本数。")
    parser.add_argument("--max-steps", type=int, help="覆盖每局最大步数。")
    parser.add_argument("--supervised-epochs", type=int, help="覆盖监督训练 epoch。")
    parser.add_argument("--supervised-batch-size", type=int, default=64)
    parser.add_argument("--supervised-lr", type=float, default=1e-4)
    parser.add_argument("--dqn-episodes", type=int, help="覆盖 DQN episodes。")
    parser.add_argument("--dqn-batch-size", type=int, help="覆盖 DQN batch size。")
    parser.add_argument("--dqn-lr", type=float, default=1e-4)
    parser.add_argument("--dqn-gamma", type=float, default=0.99)
    parser.add_argument("--dqn-buffer-size", type=int, help="覆盖 replay buffer 大小。")
    parser.add_argument("--dqn-target-update", type=int, help="覆盖 target network 同步步数。")
    parser.add_argument("--eval-cases", type=int, help="覆盖评估局数。")
    parser.add_argument("--skip-cnn", action="store_true", help="只训练 MLP，跳过 CNN。")
    parser.add_argument("--skip-dqn", action="store_true", help="只做监督学习，跳过 DQN。")
    parser.add_argument("--resume", action="store_true", help="已有数据/模型则跳过对应阶段。")
    args = parser.parse_args()

    cfg = build_config(args)
    root = args.root
    code_root = Path(__file__).resolve().parents[1]
    tests_dir = code_root / "tests"
    data_path = root / "data" / f"teacher_{cfg.teacher}_{cfg.samples}.npz"
    ckpt_dir = root / "checkpoints"
    results_dir = root / "results"
    export_dir = root / "export"

    mlp_sup = ckpt_dir / "mlp_supervised.pt"
    cnn_sup = ckpt_dir / "cnn_supervised.pt"
    mlp_dqn = ckpt_dir / "mlp_dqn.pt"
    cnn_dqn = ckpt_dir / "cnn_dqn.pt"
    mlp_weights = export_dir / "mlp_weights.h"
    metrics_path = results_dir / f"pipeline_{cfg.preset}_metrics.json"

    log("训练流水线配置：")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    run_step("环境规则单测", run_pytest, tests_dir)

    if should_skip(data_path, cfg.resume):
        log(f"跳过：teacher 数据已存在 {data_path}")
    else:
        run_step("生成 teacher 数据", generate_teacher_data, cfg.samples, data_path, cfg.seed, cfg.max_steps, cfg.teacher)

    if should_skip(mlp_sup, cfg.resume):
        log(f"跳过：MLP supervised 已存在 {mlp_sup}")
    else:
        run_step("训练 MLP supervised", train_supervised, data_path, "mlp", mlp_sup, cfg.supervised_epochs, cfg.supervised_batch_size, cfg.supervised_lr)

    if cfg.run_cnn:
        if should_skip(cnn_sup, cfg.resume):
            log(f"跳过：CNN supervised 已存在 {cnn_sup}")
        else:
            run_step("训练 CNN supervised", train_supervised, data_path, "cnn", cnn_sup, cfg.supervised_epochs, cfg.supervised_batch_size, cfg.supervised_lr)

    if cfg.run_dqn:
        if should_skip(mlp_dqn, cfg.resume):
            log(f"跳过：MLP DQN 已存在 {mlp_dqn}")
        else:
            run_step(
                "训练 MLP DQN",
                train_dqn,
                "mlp",
                mlp_dqn,
                cfg.dqn_episodes,
                cfg.dqn_batch_size,
                cfg.dqn_lr,
                cfg.dqn_gamma,
                cfg.dqn_buffer_size,
                cfg.dqn_target_update,
                cfg.seed,
                mlp_sup,
            )
        if cfg.run_cnn:
            if should_skip(cnn_dqn, cfg.resume):
                log(f"跳过：CNN DQN 已存在 {cnn_dqn}")
            else:
                run_step(
                    "训练 CNN DQN",
                    train_dqn,
                    "cnn",
                    cnn_dqn,
                    cfg.dqn_episodes,
                    cfg.dqn_batch_size,
                    cfg.dqn_lr,
                    cfg.dqn_gamma,
                    cfg.dqn_buffer_size,
                    cfg.dqn_target_update,
                    cfg.seed,
                    None,
                )

    rows: list[dict] = []
    teacher_policy = choose_teacher_action if cfg.teacher == "strong" else choose_fast_teacher_action
    run_step("评估 teacher", lambda: rows.append({"policy": f"{cfg.teacher}_teacher", **evaluate_policy(teacher_policy, cfg.eval_cases, cfg.seed, cfg.max_steps)}))
    run_step("评估 MLP supervised", lambda: rows.append({"policy": "mlp_supervised", **evaluate_policy(make_model_policy("mlp", mlp_sup), cfg.eval_cases, cfg.seed, cfg.max_steps)}))
    if cfg.run_cnn and cnn_sup.exists():
        run_step("评估 CNN supervised", lambda: rows.append({"policy": "cnn_supervised", **evaluate_policy(make_model_policy("cnn", cnn_sup), cfg.eval_cases, cfg.seed, cfg.max_steps)}))
    if cfg.run_dqn and mlp_dqn.exists():
        run_step("评估 MLP DQN", lambda: rows.append({"policy": "mlp_dqn", **evaluate_policy(make_model_policy("mlp", mlp_dqn), cfg.eval_cases, cfg.seed, cfg.max_steps)}))
    if cfg.run_dqn and cfg.run_cnn and cnn_dqn.exists():
        run_step("评估 CNN DQN", lambda: rows.append({"policy": "cnn_dqn", **evaluate_policy(make_model_policy("cnn", cnn_dqn), cfg.eval_cases, cfg.seed, cfg.max_steps)}))
    write_metrics(metrics_path, rows)
    log(f"评估结果已写入 {metrics_path}")

    run_step("导出 MLP supervised C 权重", export_mlp_c, mlp_sup, mlp_weights)
    log(f"MLP 权重已导出 {mlp_weights}")
    log("流水线结束")


if __name__ == "__main__":
    main()
