import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "strategies" / "bfs_dfs_survival" / "snake_oj.c"
OUT_ROOT = ROOT / "strategies" / "generated_tuning"


PARAM_SETS = [
    {"name": "base"},
    {"name": "more_candidates", "MAX_CANDIDATES": 60, "MAX_DFS_NODES": 6000},
    {"name": "deeper_food", "DELTA": 8, "MAX_CANDIDATES": 80, "MAX_DFS_NODES": 9000, "MAX_PATH_LEN": 120},
    {"name": "space_safe", "SPACE_WEIGHT": 14, "TAIL_WEIGHT": 1500, "TRAP_PENALTY": 5000, "TRAP_MARGIN": 12},
    {"name": "space_aggressive", "PATH_PENALTY": 2, "SPACE_WEIGHT": 10, "TAIL_WEIGHT": 900, "TRAP_PENALTY": 2500, "TRAP_MARGIN": 6},
    {"name": "tail_first", "SPACE_WEIGHT": 9, "TAIL_WEIGHT": 3000, "TRAP_PENALTY": 4500, "TRAP_MARGIN": 10},
    {"name": "short_path", "DELTA": 2, "PATH_PENALTY": 1, "SPACE_WEIGHT": 8, "TAIL_WEIGHT": 1000},
    {"name": "long_safe", "DELTA": 10, "MAX_CANDIDATES": 100, "MAX_DFS_NODES": 12000, "MAX_PATH_LEN": 140, "PATH_PENALTY": 3, "SPACE_WEIGHT": 16, "TAIL_WEIGHT": 1800, "TRAP_PENALTY": 6000, "TRAP_MARGIN": 14},
    {"name": "anti_trap", "DELTA": 6, "MAX_CANDIDATES": 70, "MAX_DFS_NODES": 8000, "SPACE_WEIGHT": 18, "TAIL_WEIGHT": 2200, "TRAP_PENALTY": 8000, "TRAP_MARGIN": 18},
    {"name": "eat_bias", "DELTA": 5, "MAX_CANDIDATES": 50, "MAX_DFS_NODES": 5000, "PATH_PENALTY": 0, "SPACE_WEIGHT": 7, "TAIL_WEIGHT": 700, "TRAP_PENALTY": 2000, "TRAP_MARGIN": 5},
    {"name": "anti_trap_mild", "DELTA": 5, "MAX_CANDIDATES": 60, "MAX_DFS_NODES": 7000, "SPACE_WEIGHT": 15, "TAIL_WEIGHT": 1800, "TRAP_PENALTY": 6500, "TRAP_MARGIN": 14},
    {"name": "anti_trap_plus", "DELTA": 6, "MAX_CANDIDATES": 80, "MAX_DFS_NODES": 10000, "PATH_PENALTY": 3, "SPACE_WEIGHT": 20, "TAIL_WEIGHT": 2600, "TRAP_PENALTY": 9500, "TRAP_MARGIN": 20},
    {"name": "anti_trap_hard", "DELTA": 7, "MAX_CANDIDATES": 90, "MAX_DFS_NODES": 12000, "PATH_PENALTY": 3, "SPACE_WEIGHT": 24, "TAIL_WEIGHT": 3200, "TRAP_PENALTY": 13000, "TRAP_MARGIN": 24},
    {"name": "anti_trap_fast", "DELTA": 4, "MAX_CANDIDATES": 55, "MAX_DFS_NODES": 6500, "PATH_PENALTY": 1, "SPACE_WEIGHT": 18, "TAIL_WEIGHT": 2400, "TRAP_PENALTY": 8500, "TRAP_MARGIN": 18},
    {"name": "anti_trap_low_path", "DELTA": 6, "MAX_CANDIDATES": 70, "MAX_DFS_NODES": 8500, "PATH_PENALTY": 0, "SPACE_WEIGHT": 18, "TAIL_WEIGHT": 2200, "TRAP_PENALTY": 8000, "TRAP_MARGIN": 18},
    {"name": "anti_trap_tail", "DELTA": 6, "MAX_CANDIDATES": 70, "MAX_DFS_NODES": 9000, "PATH_PENALTY": 3, "SPACE_WEIGHT": 17, "TAIL_WEIGHT": 4200, "TRAP_PENALTY": 9000, "TRAP_MARGIN": 18},
    {"name": "anti_trap_space", "DELTA": 6, "MAX_CANDIDATES": 70, "MAX_DFS_NODES": 9000, "PATH_PENALTY": 3, "SPACE_WEIGHT": 30, "TAIL_WEIGHT": 1800, "TRAP_PENALTY": 9000, "TRAP_MARGIN": 20},
    {"name": "short_anti_mix", "DELTA": 3, "MAX_CANDIDATES": 50, "MAX_DFS_NODES": 6000, "PATH_PENALTY": 0, "SPACE_WEIGHT": 16, "TAIL_WEIGHT": 2000, "TRAP_PENALTY": 7500, "TRAP_MARGIN": 16},
]


def patch_source(text: str, params: dict) -> str:
    replacements = {
        "DELTA": params.get("DELTA", 4),
        "MAX_CANDIDATES": params.get("MAX_CANDIDATES", 30),
        "MAX_PATH_LEN": params.get("MAX_PATH_LEN", 80),
        "MAX_DFS_NODES": params.get("MAX_DFS_NODES", 2500),
    }
    for key, value in replacements.items():
        text = replace_define(text, key, value)

    path_penalty = params.get("PATH_PENALTY", 5)
    space_weight = params.get("SPACE_WEIGHT", 8)
    tail_weight = params.get("TAIL_WEIGHT", 1000)
    trap_penalty = params.get("TRAP_PENALTY", 3000)
    trap_margin = params.get("TRAP_MARGIN", 5)
    text = text.replace(
        "int trapped = (space < s->len + 5);\n    return 10000 - 5 * pathLen + 8 * space + 1000 * canReachTail - 3000 * trapped;",
        f"int trapped = (space < s->len + {trap_margin});\n"
        f"    return 10000 - {path_penalty} * pathLen + {space_weight} * space + {tail_weight} * canReachTail - {trap_penalty} * trapped;",
    )
    return text


def replace_define(text: str, key: str, value: int) -> str:
    lines = []
    prefix = f"#define {key} "
    for line in text.splitlines():
        if line.startswith(prefix):
            lines.append(f"#define {key} {value}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def write_variant(params: dict) -> Path:
    name = params["name"]
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    text = BASE.read_text(encoding="utf-8")
    patched = patch_source(text, params)
    (out_dir / "snake_oj.c").write_text(patched, encoding="utf-8")
    (out_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_dir


def run_batch(strategy_dir: Path, runs: int, seed_start: int) -> dict:
    tag = strategy_dir.name
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "batch_judge.py"),
        str(strategy_dir),
        "--runs",
        str(runs),
        "--seed-start",
        str(seed_start),
        "--out-dir",
        str(ROOT / "results" / "tuning"),
        "--tag",
        tag,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    summary_path = ROOT / "results" / "tuning" / f"{tag}_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Generate and score C snake strategy variants.")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=20260516)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--names", default="", help="comma-separated variant names to run")
    args = parser.parse_args()

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sets = PARAM_SETS[: args.limit] if args.limit else PARAM_SETS
    if args.names:
        wanted = {name.strip() for name in args.names.split(",") if name.strip()}
        sets = [params for params in sets if params["name"] in wanted]
    ranking = []
    for idx, params in enumerate(sets, 1):
        strategy_dir = write_variant(params)
        print(f"=== {idx}/{len(sets)} {params['name']} ===", flush=True)
        summary = run_batch(strategy_dir, args.runs, args.seed_start)
        ranking.append({"name": params["name"], "params": params, "summary": summary})

    ranking.sort(key=lambda x: (x["summary"]["top1"], x["summary"]["p90"], x["summary"]["avg"]), reverse=True)
    out = ROOT / "results" / "tuning" / "ranking.json"
    out.write_text(json.dumps(ranking, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("=== RANKING ===")
    for row in ranking:
        s = row["summary"]
        print(f"{row['name']}: top1={s['top1']:.2f} p90={s['p90']:.2f} avg={s['avg']:.2f} best_seed={s['best_seed']}")
    print(f"ranking={out}")


if __name__ == "__main__":
    main()
