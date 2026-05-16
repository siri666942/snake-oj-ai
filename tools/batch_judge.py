import argparse
import csv
import json
import os
import random
import statistics
import sys
from pathlib import Path

import judge


NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def percentile(values, q):
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def make_seeds(count, seed_start, random_mode):
    if random_mode:
        rng = random.SystemRandom()
        return [rng.randint(1, 2_147_483_647) for _ in range(count)]
    return [seed_start + i * 10007 for i in range(count)]


def run_one(exe, base_seed):
    cases = [judge.make_base(base_seed + i * 97, n) for i, n in enumerate(NS)]
    scores = []
    reasons = []
    for case in cases:
        score, reason, _final_output, err = judge.run_case(exe, case)
        if err.strip():
            reason = reason + "|stderr"
        scores.append(score)
        reasons.append(reason)
    total = judge.weighted_total(scores, NS)
    return {
        "base_seed": base_seed,
        "weighted_total": total,
        "scores": scores,
        "reasons": reasons,
    }


def summarize(rows):
    totals = [r["weighted_total"] for r in rows]
    best = max(rows, key=lambda r: r["weighted_total"]) if rows else None
    per_n_avg = []
    per_n_best = []
    per_n_deaths = []
    for idx, n in enumerate(NS):
        vals = [r["scores"][idx] for r in rows]
        per_n_avg.append(sum(vals) / max(len(vals), 1))
        per_n_best.append(max(vals) if vals else 0)
        per_n_deaths.append(sum(1 for r in rows if r["reasons"][idx] != "max_steps"))
    return {
        "runs": len(rows),
        "avg": statistics.mean(totals) if totals else 0.0,
        "std": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "min": min(totals) if totals else 0.0,
        "p50": percentile(totals, 0.50),
        "p90": percentile(totals, 0.90),
        "p95": percentile(totals, 0.95),
        "top1": max(totals) if totals else 0.0,
        "top3": sorted(totals, reverse=True)[:3],
        "best_seed": best["base_seed"] if best else None,
        "best_scores_by_n": dict(zip(NS, best["scores"])) if best else {},
        "best_reasons_by_n": dict(zip(NS, best["reasons"])) if best else {},
        "avg_score_by_n": dict(zip(NS, per_n_avg)),
        "best_score_by_n": dict(zip(NS, per_n_best)),
        "death_count_by_n": dict(zip(NS, per_n_deaths)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run many local judge seeds and summarize top-score distribution.")
    parser.add_argument("strategy", nargs="?", default=judge.DEFAULT_STRATEGY)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=judge.DEFAULT_SEED)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results/batch_judge"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    strategy_dir = judge.resolve_strategy_dir(args.strategy)
    if not os.path.exists(os.path.join(strategy_dir, "snake_oj.c")):
        raise SystemExit(f"strategy source not found: {os.path.join(strategy_dir, 'snake_oj.c')}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(strategy_dir).name
    csv_path = args.out_dir / f"{tag}_runs.csv"
    json_path = args.out_dir / f"{tag}_summary.json"

    exe = judge.compile_program(strategy_dir)
    rows = []
    try:
        seeds = make_seeds(args.runs, args.seed_start, args.random)
        for idx, base_seed in enumerate(seeds, 1):
            row = run_one(exe, base_seed)
            rows.append(row)
            print(f"{idx}/{len(seeds)} seed={base_seed} weighted_total={row['weighted_total']:.2f}", flush=True)
    finally:
        try:
            os.remove(exe)
        except OSError:
            pass

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["base_seed", "weighted_total"] + [f"N{n}_score" for n in NS] + [f"N{n}_reason" for n in NS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {"base_seed": row["base_seed"], "weighted_total": f"{row['weighted_total']:.6f}"}
            flat.update({f"N{n}_score": row["scores"][i] for i, n in enumerate(NS)})
            flat.update({f"N{n}_reason": row["reasons"][i] for i, n in enumerate(NS)})
            writer.writerow(flat)

    summary = summarize(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"summary={json_path}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
