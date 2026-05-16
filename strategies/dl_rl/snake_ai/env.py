from __future__ import annotations

import copy
import random
from dataclasses import dataclass

SIZE = 20
NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
ACTION_CHARS = ["W", "A", "S", "D"]
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, -1),
    2: (1, 0),
    3: (0, 1),
}
CHAR_TO_ACTION = {ch: i for i, ch in enumerate(ACTION_CHARS)}
OPPOSITE = {0: 2, 2: 0, 1: 3, 3: 1}


@dataclass
class SnakeCase:
    grid: list[str]
    n: int
    seed: int


@dataclass
class StepResult:
    state: "SnakeEnv"
    reward: float
    done: bool
    info: dict


def random_empty_cell(rng: random.Random, grid: list[list[str]] | list[str]) -> tuple[int, int]:
    while True:
        r = rng.randint(1, 18)
        c = rng.randint(1, 18)
        if grid[r][c] == ".":
            return r, c


def make_case(seed: int, n: int) -> SnakeCase:
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
    obstacles: set[tuple[int, int]] = set()
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
    return SnakeCase(["".join(row) for row in grid], n, seed)


def parse_grid(grid: list[str]) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    body_cells = set()
    head = None
    food = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "H":
                head = (r, c)
            elif ch == "B":
                body_cells.add((r, c))
            elif ch == "F":
                food = (r, c)
    if head is None:
        raise ValueError("map has no snake head")

    snake = [head]
    prev = None
    cur = head
    for _ in range(len(body_cells)):
        found = None
        for dr, dc in ACTION_TO_DELTA.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in body_cells and nxt != prev:
                found = nxt
                break
        if found is None:
            break
        snake.append(found)
        prev, cur = cur, found
    return snake, food


def infer_dir(snake: list[tuple[int, int]]) -> int:
    head, neck = snake[0], snake[1]
    if neck == (head[0] - 1, head[1]):
        return 2
    if neck == (head[0] + 1, head[1]):
        return 0
    if neck == (head[0], head[1] - 1):
        return 3
    if neck == (head[0], head[1] + 1):
        return 1
    return 3


class SnakeEnv:
    def __init__(
        self,
        reward_food: float = 10.0,
        reward_death: float = -50.0,
        reward_step: float = -0.01,
        max_steps: int = 5000,
    ) -> None:
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.max_steps = max_steps
        self.rng = random.Random()
        self.base: list[list[str]] = []
        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] | None = None
        self.n = 32
        self.seed = 0
        self.score = 0
        self.steps = 0
        self.cur_dir = 3
        self.food_count = 0
        self.done = False
        self.death_reason = ""

    def reset(self, seed: int = 20260516, n: int | None = None, grid: list[str] | None = None) -> "SnakeEnv":
        self.seed = seed
        self.n = n if n is not None else NS[seed % len(NS)]
        case = SnakeCase(grid, self.n, seed) if grid is not None else make_case(seed, self.n)
        self.rng = random.Random(case.seed + 1000003)
        self.snake, self.food = parse_grid(case.grid)
        self.base = [["#" if ch == "#" else "O" if ch == "O" else "." for ch in row] for row in case.grid]
        self.score = 0
        self.steps = 0
        self.cur_dir = infer_dir(self.snake)
        self.food_count = 0
        self.done = False
        self.death_reason = ""
        return self

    def clone(self) -> "SnakeEnv":
        return copy.deepcopy(self)

    def render_grid(self) -> list[str]:
        grid = [row[:] for row in self.base]
        for r, c in self.snake[1:]:
            grid[r][c] = "B"
        hr, hc = self.snake[0]
        grid[hr][hc] = "H"
        if self.food is not None:
            grid[self.food[0]][self.food[1]] = "F"
        return ["".join(row) for row in grid]

    def legal_actions(self) -> list[int]:
        return [a for a in range(4) if self.clone()._move_no_food_spawn(a)[0]]

    def step(self, action: int) -> tuple["SnakeEnv", float, bool, dict]:
        if self.done:
            return self, 0.0, True, {"death_reason": self.death_reason}
        ok, ate, reason = self._move_no_food_spawn(action)
        reward = self.reward_step
        if not ok:
            self.done = True
            self.death_reason = reason
            return self, self.reward_death, True, {"death_reason": reason, "ate": False}
        if ate:
            self.score += 10
            self.food_count += 1
            reward += self.reward_food
            grid = self.render_grid()
            mutable = [list(row) for row in grid]
            self.food = random_empty_cell(self.rng, mutable)
        if self.steps >= self.max_steps:
            self.done = True
            self.death_reason = "max_steps"
        return self, reward, self.done, {"death_reason": self.death_reason, "ate": ate}

    def _move_no_food_spawn(self, action: int) -> tuple[bool, bool, str]:
        if action not in ACTION_TO_DELTA:
            return False, False, "bad_action"
        if OPPOSITE[self.cur_dir] == action:
            return False, False, "reverse"
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = self.snake[0][0] + dr, self.snake[0][1] + dc
        if not (0 <= nr < SIZE and 0 <= nc < SIZE):
            return False, False, "out_of_bounds"
        if self.base[nr][nc] in "#O":
            return False, False, "wall_or_obstacle"

        next_step = self.steps + 1
        ate = self.food == (nr, nc)
        grow = ate or (self.n > 0 and next_step % self.n == 0)
        tail = self.snake[-1]
        for body in self.snake[1:]:
            if body == (nr, nc) and (grow or body != tail):
                return False, False, "self"

        if grow:
            self.snake = [(nr, nc)] + self.snake
        else:
            self.snake = [(nr, nc)] + self.snake[:-1]
        self.steps = next_step
        self.cur_dir = action
        if ate:
            self.food = None
        return True, ate, ""

