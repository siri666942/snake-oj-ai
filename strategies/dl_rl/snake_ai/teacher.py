from __future__ import annotations

from collections import deque

from .env import ACTION_TO_DELTA, OPPOSITE, SIZE, SnakeEnv

INF = 10**9
DELTA = 4
MAX_CANDIDATES = 30
MAX_PATH_LEN = 80
MAX_DFS_NODES = 2500


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def passable(env: SnakeEnv, r: int, c: int, allow_tail: bool) -> bool:
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        return False
    if env.base[r][c] in "#O":
        return False
    tail = env.snake[-1]
    for body in env.snake[1:]:
        if body == (r, c):
            return allow_tail and body == tail
    return True


def bfs_distance(env: SnakeEnv, target: tuple[int, int], allow_tail: bool = True) -> tuple[int, int]:
    dist = [[-1] * SIZE for _ in range(SIZE)]
    first = [[-1] * SIZE for _ in range(SIZE)]
    q: deque[tuple[int, int]] = deque([env.snake[0]])
    hr, hc = env.snake[0]
    dist[hr][hc] = 0
    while q:
        r, c = q.popleft()
        if (r, c) == target:
            return dist[r][c], first[r][c]
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            if dist[r][c] == 0 and OPPOSITE[env.cur_dir] == action:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < SIZE and 0 <= nc < SIZE) or dist[nr][nc] != -1:
                continue
            if (nr, nc) != target and not passable(env, nr, nc, allow_tail):
                continue
            dist[nr][nc] = dist[r][c] + 1
            first[nr][nc] = action if dist[r][c] == 0 else first[r][c]
            q.append((nr, nc))
    return -1, -1


def reachable_space(env: SnakeEnv, allow_tail: bool = True) -> int:
    q: deque[tuple[int, int]] = deque([env.snake[0]])
    seen = {env.snake[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in ACTION_TO_DELTA.values():
            nr, nc = r + dr, c + dc
            if (nr, nc) in seen or not passable(env, nr, nc, allow_tail):
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return len(seen)


def evaluate_path_state(env: SnakeEnv, path_len: int) -> int:
    dist_tail, _ = bfs_distance(env, env.snake[-1], True)
    space = reachable_space(env, True)
    can_reach_tail = 1 if dist_tail >= 0 else 0
    trapped = 1 if space < len(env.snake) + 5 else 0
    return 10000 - 5 * path_len + 8 * space + 1000 * can_reach_tail - 3000 * trapped


def ordered_dirs(env: SnakeEnv) -> list[int]:
    if env.food is None:
        return list(range(4))
    return sorted(range(4), key=lambda a: manhattan((env.snake[0][0] + ACTION_TO_DELTA[a][0], env.snake[0][1] + ACTION_TO_DELTA[a][1]), env.food))


def choose_survival_action(env: SnakeEnv) -> int:
    _, first = bfs_distance(env, env.snake[-1], True)
    if first >= 0 and env.clone()._move_no_food_spawn(first)[0]:
        return first

    best_action = -1
    best_space = -1
    for action in range(4):
        nxt = env.clone()
        if not nxt._move_no_food_spawn(action)[0]:
            continue
        space = reachable_space(nxt, True)
        if space > best_space:
            best_space = space
            best_action = action
    if best_action >= 0:
        return best_action
    for action in range(4):
        if OPPOSITE[env.cur_dir] != action:
            return action
    return 0


def choose_teacher_action(env: SnakeEnv) -> int:
    if env.food is None:
        return choose_survival_action(env)
    dist, _ = bfs_distance(env, env.food, True)
    if dist <= 0:
        return choose_survival_action(env)
    max_depth = min(dist + DELTA, MAX_PATH_LEN)
    candidates: list[tuple[int, list[int]]] = []
    nodes = 0

    def dfs(state: SnakeEnv, depth: int, path: list[int]) -> None:
        nonlocal nodes
        if len(candidates) >= MAX_CANDIDATES:
            return
        nodes += 1
        if nodes > MAX_DFS_NODES:
            return
        if state.food is None:
            candidates.append((evaluate_path_state(state, depth), path[:]))
            return
        if depth >= max_depth:
            return
        need, _ = bfs_distance(state, state.food, True)
        if need < 0 or depth + need > max_depth:
            return
        for action in ordered_dirs(state):
            nxt = state.clone()
            ok, _ate, _reason = nxt._move_no_food_spawn(action)
            if not ok:
                continue
            path.append(action)
            dfs(nxt, depth + 1, path)
            path.pop()
            if len(candidates) >= MAX_CANDIDATES:
                return

    dfs(env.clone(), 0, [])
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1][0]
    return choose_survival_action(env)


def choose_fast_teacher_action(env: SnakeEnv) -> int:
    """Fast teacher for smoke tests and large cheap data passes."""
    if env.food is not None:
        _dist, first = bfs_distance(env, env.food, True)
        if first >= 0 and env.clone()._move_no_food_spawn(first)[0]:
            return first
    return choose_survival_action(env)
