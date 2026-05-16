import argparse
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass

SIZE = 20
DEFAULT_SEED = 20260516
DIRS = {
    "W": (-1, 0),
    "A": (0, -1),
    "S": (1, 0),
    "D": (0, 1),
}
OPPOSITE = {
    "W": "S",
    "S": "W",
    "A": "D",
    "D": "A",
}


@dataclass
class Case:
    grid: list[str]
    n: int
    seed: int


def make_base(seed: int, n: int) -> Case:
    rng = random.Random(seed)
    grid = [["." for _ in range(SIZE)] for _ in range(SIZE)]
    for i in range(SIZE):
        grid[0][i] = "#"
        grid[SIZE - 1][i] = "#"
        grid[i][0] = "#"
        grid[i][SIZE - 1] = "#"

    snake = [(10, 10), (10, 9), (10, 8)]
    for r, c in snake[1:]:
        grid[r][c] = "B"
    grid[snake[0][0]][snake[0][1]] = "H"

    blocked = set(snake)
    obstacles = set()
    while len(obstacles) < 10:
        r = rng.randint(2, 17)
        c = rng.randint(2, 17)
        if (r, c) in blocked:
            continue
        if abs(r - 10) + abs(c - 10) <= 3:
            continue
        obstacles.add((r, c))
    for r, c in obstacles:
        grid[r][c] = "O"

    food = random_empty_cell(rng, grid)
    grid[food[0]][food[1]] = "F"
    return Case(["".join(row) for row in grid], n, seed)


def random_empty_cell(rng: random.Random, grid: list[list[str]] | list[str]) -> tuple[int, int]:
    while True:
        r = rng.randint(1, 18)
        c = rng.randint(1, 18)
        ch = grid[r][c]
        if ch == ".":
            return r, c


def parse_state(grid: list[str]):
    snake_cells = []
    head = None
    food = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "H":
                head = (r, c)
            elif ch == "B":
                snake_cells.append((r, c))
            elif ch == "F":
                food = (r, c)
    if head is None:
        raise ValueError("map has no snake head")

    snake = [head]
    prev = None
    cur = head
    for _ in range(len(snake_cells)):
        found = None
        for dr, dc in DIRS.values():
            nr, nc = cur[0] + dr, cur[1] + dc
            if (nr, nc) in snake_cells and (nr, nc) != prev:
                found = (nr, nc)
                break
        if found is None:
            break
        snake.append(found)
        prev, cur = cur, found
    return snake, food


def infer_dir(snake):
    head, neck = snake[0], snake[1]
    if neck == (head[0] - 1, head[1]):
        return "S"
    if neck == (head[0] + 1, head[1]):
        return "W"
    if neck == (head[0], head[1] - 1):
        return "D"
    if neck == (head[0], head[1] + 1):
        return "A"
    return "D"


def rebuild(base, snake, food):
    grid = [row[:] for row in base]
    for r, c in snake[1:]:
        grid[r][c] = "B"
    hr, hc = snake[0]
    grid[hr][hc] = "H"
    if food is not None:
        grid[food[0]][food[1]] = "F"
    return grid


def simulate_move(base, snake, food, score, step, cur_dir, n, move):
    if move not in DIRS:
        return None
    if OPPOSITE[cur_dir] == move:
        return None

    dr, dc = DIRS[move]
    nr, nc = snake[0][0] + dr, snake[0][1] + dc
    if base[nr][nc] in "#O":
        return None

    next_step = step + 1
    eat = food == (nr, nc)
    grow = eat or (next_step % n == 0)
    tail = snake[-1]
    for body in snake[1:]:
        if body == (nr, nc):
            if not grow and body == tail:
                continue
            return None

    if grow:
        new_snake = [(nr, nc)] + snake
    else:
        new_snake = [(nr, nc)] + snake[:-1]
    new_score = score + (10 if eat else 0)
    return new_snake, (None if eat else food), new_score, next_step, move, eat


def run_case(exe: str, case: Case, max_steps: int = 5000):
    rng = random.Random(case.seed + 1000003)
    snake, food = parse_state(case.grid)
    base = []
    for row in case.grid:
        base.append(["#" if ch == "#" else "O" if ch == "O" else "." for ch in row])

    proc = subprocess.Popen(
        ["cmd", "/c", ".\\" + os.path.basename(exe)],
        cwd=os.path.dirname(exe),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write("\n".join(case.grid) + f"\n{case.n}\n")
    proc.stdin.flush()

    score = 0
    step = 0
    cur_dir = infer_dir(snake)
    final_output = []
    reason = "max_steps"

    for _ in range(max_steps):
        move = proc.stdout.readline()
        if not move:
            reason = "program_ended"
            break
        score_line = proc.stdout.readline()
        if not score_line:
            reason = "bad_output"
            break
        move = move.strip()
        reported_score = score_line.strip()
        if reported_score != str(score):
            reason = f"wrong_score_reported:{reported_score}:expected:{score}"
            proc.stdin.write("100 100\n")
            proc.stdin.flush()
            break

        result = simulate_move(base, snake, food, score, step, cur_dir, case.n, move)
        if result is None:
            reason = "dead"
            proc.stdin.write("100 100\n")
            proc.stdin.flush()
            break

        snake, food, score, step, cur_dir, ate = result
        current_grid = rebuild(base, snake, food)
        if ate:
            new_food = random_empty_cell(rng, current_grid)
            food = new_food
            proc.stdin.write(f"{new_food[0]} {new_food[1]}\n")
        else:
            proc.stdin.write("20 20\n")
        proc.stdin.flush()
    else:
        proc.stdin.write("100 100\n")
        proc.stdin.flush()

    try:
        for _ in range(21):
            line = proc.stdout.readline()
            if not line:
                break
            final_output.append(line.rstrip("\n"))
    except Exception:
        pass

    try:
        proc.kill()
    except Exception:
        pass
    _, err = proc.communicate(timeout=1)
    return score, reason, final_output, err


def weighted_total(scores, ns):
    total = 0.0
    for score, n in zip(scores, ns):
        total += score * (1.0 / (math.log2(n) + 1.0))
    return total


def base_grade(total):
    if total >= 500:
        return 50
    if total >= 400:
        return 45
    if total >= 300:
        return 40
    if total >= 200:
        return 35
    if total >= 100:
        return 30
    return 0


DEFAULT_STRATEGY = os.path.join("strategies", "bfs_dfs_survival")


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_strategy_dir(strategy: str):
    if os.path.isabs(strategy):
        return strategy
    return os.path.join(project_root(), strategy)


def compile_program(strategy_dir):
    exe_name = f"snake_oj_judge_{os.getpid()}.exe"
    exe = os.path.join(strategy_dir, exe_name)
    cmd = ["gcc", "snake_oj.c", "-O2", "-Wall", "-Wextra", "-o", exe_name]
    completed = subprocess.run(
        cmd,
        cwd=strategy_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise SystemExit("compile failed")
    if not os.path.exists(exe):
        raise SystemExit(f"compile finished but {exe_name} was not created")
    if completed.stderr.strip():
        print("compiler warnings:")
        print(completed.stderr)
    return exe


def parse_args():
    parser = argparse.ArgumentParser(description="Compile and locally judge a snake strategy.")
    parser.add_argument(
        "strategy",
        nargs="?",
        default=DEFAULT_STRATEGY,
        help=f"strategy directory, relative to project root or absolute; default: {DEFAULT_STRATEGY}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"base random seed for reproducible cases; default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="use a fresh random base seed for this run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_dir = resolve_strategy_dir(args.strategy)
    if not os.path.exists(os.path.join(strategy_dir, "snake_oj.c")):
        raise SystemExit(f"strategy source not found: {os.path.join(strategy_dir, 'snake_oj.c')}")

    exe = compile_program(strategy_dir)
    try:
        ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        base_seed = random.SystemRandom().randint(1, 2_147_483_647) if args.random else args.seed
        cases = [make_base(base_seed + i * 97, n) for i, n in enumerate(ns)]
        scores = []

        print(f"strategy={strategy_dir}", flush=True)
        print(f"base_seed={base_seed}", flush=True)
        print("case,N,raw_score,reason", flush=True)
        for i, case in enumerate(cases, 1):
            score, reason, _, err = run_case(exe, case)
            scores.append(score)
            print(f"{i},{case.n},{score},{reason}", flush=True)
            if err.strip():
                print(err, file=sys.stderr)

        total = weighted_total(scores, ns)
        print(f"weighted_total={total:.2f}", flush=True)
        print(f"base_grade={base_grade(total)}", flush=True)
        print("note=本地随机模拟分数只用于估计，OJ隐藏数据分数以正式提交为准。", flush=True)
    finally:
        try:
            os.remove(exe)
        except OSError:
            pass


if __name__ == "__main__":
    main()
