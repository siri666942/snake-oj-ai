#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define SIZE 20
#define MAX_CELLS 400
#define INF 1000000000
#define DELTA 6
#define MAX_CANDIDATES 70
#define MAX_PATH_LEN 80
#define MAX_DFS_NODES 8500

typedef struct {
    int r;
    int c;
} Point;

typedef struct {
    char map[SIZE][SIZE + 1];
    Point snake[MAX_CELLS + 1];
    int len;
    int score;
    int step;
    int curDir;
    int foodR;
    int foodC;
} GameState;

typedef struct {
    int dirs[MAX_PATH_LEN];
    int len;
    int score;
} Path;

static const int dr[4] = {-1, 0, 1, 0};
static const int dc[4] = {0, -1, 0, 1};
static const char dirChar[4] = {'W', 'A', 'S', 'D'};

static char baseMap[SIZE][SIZE + 1];
static int growN;
static Path candidates[MAX_CANDIDATES];
static int candidateCount;
static int dfsNodeCount;

static int samePoint(Point a, Point b) {
    return a.r == b.r && a.c == b.c;
}

static int isOpposite(int a, int b) {
    return (a == 0 && b == 2) || (a == 2 && b == 0) ||
           (a == 1 && b == 3) || (a == 3 && b == 1);
}

static int inside(int r, int c) {
    return r >= 0 && r < SIZE && c >= 0 && c < SIZE;
}

static int manhattan(int r1, int c1, int r2, int c2) {
    int a = r1 - r2;
    int b = c1 - c2;
    return (a < 0 ? -a : a) + (b < 0 ? -b : b);
}

static void rebuildMap(GameState *s) {
    int r, c, i;
    for (r = 0; r < SIZE; r++) {
        for (c = 0; c < SIZE; c++) {
            if (baseMap[r][c] == '#' || baseMap[r][c] == 'O') {
                s->map[r][c] = baseMap[r][c];
            } else {
                s->map[r][c] = '.';
            }
        }
        s->map[r][SIZE] = '\0';
    }

    for (i = s->len - 1; i >= 1; i--) {
        s->map[s->snake[i].r][s->snake[i].c] = 'B';
    }
    s->map[s->snake[0].r][s->snake[0].c] = 'H';
    if (s->foodR >= 0 && s->foodC >= 0) {
        s->map[s->foodR][s->foodC] = 'F';
    }
}

static int simulateMove(GameState *s, int dir) {
    GameState old = *s;
    int nr, nc, nextStep, eatFood, naturalGrow, grow, i, oldLen;
    Point tail;

    if (isOpposite(s->curDir, dir)) {
        return 0;
    }

    nr = s->snake[0].r + dr[dir];
    nc = s->snake[0].c + dc[dir];
    if (!inside(nr, nc) || baseMap[nr][nc] == '#' || baseMap[nr][nc] == 'O') {
        return 0;
    }

    nextStep = s->step + 1;
    eatFood = (s->foodR >= 0 && nr == s->foodR && nc == s->foodC);
    naturalGrow = (growN > 0 && nextStep % growN == 0);
    grow = eatFood || naturalGrow;
    tail = s->snake[s->len - 1];

    for (i = 1; i < s->len; i++) {
        if (s->snake[i].r == nr && s->snake[i].c == nc) {
            if (!grow && samePoint(s->snake[i], tail)) {
                continue;
            }
            *s = old;
            return 0;
        }
    }

    oldLen = s->len;
    if (grow) {
        if (oldLen >= MAX_CELLS) {
            return 0;
        }
        for (i = oldLen; i >= 1; i--) {
            s->snake[i] = s->snake[i - 1];
        }
        s->len = oldLen + 1;
    } else {
        for (i = oldLen - 1; i >= 1; i--) {
            s->snake[i] = s->snake[i - 1];
        }
    }

    s->snake[0].r = nr;
    s->snake[0].c = nc;
    s->step = nextStep;
    s->curDir = dir;
    if (eatFood) {
        s->score += 10;
        s->foodR = -1;
        s->foodC = -1;
    }
    rebuildMap(s);
    return 1;
}

static int passableForBfs(const GameState *s, int r, int c, int allowTail) {
    int i;
    Point tail = s->snake[s->len - 1];
    if (!inside(r, c) || baseMap[r][c] == '#' || baseMap[r][c] == 'O') {
        return 0;
    }
    for (i = 1; i < s->len; i++) {
        if (s->snake[i].r == r && s->snake[i].c == c) {
            return allowTail && r == tail.r && c == tail.c;
        }
    }
    return 1;
}

static int bfsDistance(const GameState *s, Point target, int allowTail, int firstDirOut[1]) {
    int dist[SIZE][SIZE];
    int first[SIZE][SIZE];
    Point q[SIZE * SIZE];
    int head = 0, tail = 0, r, c, d, nr, nc;

    for (r = 0; r < SIZE; r++) {
        for (c = 0; c < SIZE; c++) {
            dist[r][c] = -1;
            first[r][c] = -1;
        }
    }

    q[tail++] = s->snake[0];
    dist[s->snake[0].r][s->snake[0].c] = 0;

    while (head < tail) {
        Point p = q[head++];
        if (p.r == target.r && p.c == target.c) {
            if (firstDirOut) {
                firstDirOut[0] = first[p.r][p.c];
            }
            return dist[p.r][p.c];
        }

        for (d = 0; d < 4; d++) {
            if (dist[p.r][p.c] == 0 && isOpposite(s->curDir, d)) {
                continue;
            }
            nr = p.r + dr[d];
            nc = p.c + dc[d];
            if (!inside(nr, nc) || dist[nr][nc] != -1) {
                continue;
            }
            if (nr == target.r && nc == target.c) {
                dist[nr][nc] = dist[p.r][p.c] + 1;
                first[nr][nc] = (dist[p.r][p.c] == 0 ? d : first[p.r][p.c]);
                q[tail++] = (Point){nr, nc};
                continue;
            }
            if (!passableForBfs(s, nr, nc, allowTail)) {
                continue;
            }
            dist[nr][nc] = dist[p.r][p.c] + 1;
            first[nr][nc] = (dist[p.r][p.c] == 0 ? d : first[p.r][p.c]);
            q[tail++] = (Point){nr, nc};
        }
    }
    if (firstDirOut) {
        firstDirOut[0] = -1;
    }
    return -1;
}

static int reachableSpace(const GameState *s, int allowTail) {
    int vis[SIZE][SIZE];
    Point q[SIZE * SIZE];
    int head = 0, tail = 0, d, nr, nc, cnt = 0;

    memset(vis, 0, sizeof(vis));
    q[tail++] = s->snake[0];
    vis[s->snake[0].r][s->snake[0].c] = 1;

    while (head < tail) {
        Point p = q[head++];
        cnt++;
        for (d = 0; d < 4; d++) {
            nr = p.r + dr[d];
            nc = p.c + dc[d];
            if (!inside(nr, nc) || vis[nr][nc]) {
                continue;
            }
            if (!passableForBfs(s, nr, nc, allowTail)) {
                continue;
            }
            vis[nr][nc] = 1;
            q[tail++] = (Point){nr, nc};
        }
    }
    return cnt;
}

static int evaluatePathState(const GameState *s, int pathLen) {
    Point tail = s->snake[s->len - 1];
    int space = reachableSpace(s, 1);
    int canReachTail = (bfsDistance(s, tail, 1, NULL) >= 0);
    int trapped = (space < s->len + 18);
    return 10000 - 0 * pathLen + 18 * space + 2200 * canReachTail - 8000 * trapped;
}

static void saveCandidate(const GameState *s, const int path[], int depth) {
    int i;
    if (candidateCount >= MAX_CANDIDATES || depth <= 0) {
        return;
    }
    candidates[candidateCount].len = depth;
    candidates[candidateCount].score = evaluatePathState(s, depth);
    for (i = 0; i < depth; i++) {
        candidates[candidateCount].dirs[i] = path[i];
    }
    candidateCount++;
}

static void orderedDirs(const GameState *s, int order[4]) {
    int used[4] = {0, 0, 0, 0};
    int i, d, bestD, bestScore, score;
    for (i = 0; i < 4; i++) {
        bestD = -1;
        bestScore = INF;
        for (d = 0; d < 4; d++) {
            if (used[d]) {
                continue;
            }
            score = manhattan(s->snake[0].r + dr[d], s->snake[0].c + dc[d], s->foodR, s->foodC);
            if (score < bestScore) {
                bestScore = score;
                bestD = d;
            }
        }
        order[i] = bestD;
        used[bestD] = 1;
    }
}

static void dfsPaths(GameState s, int depth, int maxDepth, int path[]) {
    int order[4];
    int i, dir, need;
    GameState next;

    if (candidateCount >= MAX_CANDIDATES) {
        return;
    }
    if (++dfsNodeCount > MAX_DFS_NODES) {
        return;
    }
    if (s.foodR < 0) {
        saveCandidate(&s, path, depth);
        return;
    }
    if (depth >= maxDepth) {
        return;
    }
    need = bfsDistance(&s, (Point){s.foodR, s.foodC}, 1, NULL);
    if (need < 0 || depth + need > maxDepth) {
        return;
    }

    orderedDirs(&s, order);
    for (i = 0; i < 4; i++) {
        dir = order[i];
        next = s;
        if (!simulateMove(&next, dir)) {
            continue;
        }
        path[depth] = dir;
        dfsPaths(next, depth + 1, maxDepth, path);
        if (candidateCount >= MAX_CANDIDATES) {
            return;
        }
    }
}

static int chooseSurvivalDir(const GameState *s) {
    Point tail = s->snake[s->len - 1];
    int firstDir = -1;
    int bestDir = -1;
    int bestSpace = -1;
    int d, space;
    GameState next;

    if (bfsDistance(s, tail, 1, &firstDir) >= 0 && firstDir >= 0) {
        next = *s;
        if (simulateMove(&next, firstDir)) {
            return firstDir;
        }
    }

    for (d = 0; d < 4; d++) {
        next = *s;
        if (!simulateMove(&next, d)) {
            continue;
        }
        space = reachableSpace(&next, 1);
        if (space > bestSpace) {
            bestSpace = space;
            bestDir = d;
        }
    }

    if (bestDir >= 0) {
        return bestDir;
    }
    for (d = 0; d < 4; d++) {
        if (!isOpposite(s->curDir, d)) {
            return d;
        }
    }
    return 0;
}

static int chooseDirection(const GameState *s) {
    Point food;
    int d, maxDepth, path[MAX_PATH_LEN], i, bestIdx = -1, bestScore = -INF;

    if (s->foodR >= 0) {
        food.r = s->foodR;
        food.c = s->foodC;
        d = bfsDistance(s, food, 1, NULL);
        if (d > 0) {
            maxDepth = d + DELTA;
            if (maxDepth > MAX_PATH_LEN) {
                maxDepth = MAX_PATH_LEN;
            }
            candidateCount = 0;
            dfsNodeCount = 0;
            dfsPaths(*s, 0, maxDepth, path);
            for (i = 0; i < candidateCount; i++) {
                if (candidates[i].score > bestScore) {
                    bestScore = candidates[i].score;
                    bestIdx = i;
                }
            }
            if (bestIdx >= 0) {
                return candidates[bestIdx].dirs[0];
            }
        }
    }
    return chooseSurvivalDir(s);
}

static void printFinalState(const GameState *s) {
    int r;
    for (r = 0; r < SIZE; r++) {
        printf("%s\n", s->map[r]);
    }
    printf("%d\n", s->score);
    fflush(stdout);
}

static int inferCurrentDir(Point head, Point neck) {
    if (neck.r == head.r - 1 && neck.c == head.c) {
        return 2;
    }
    if (neck.r == head.r + 1 && neck.c == head.c) {
        return 0;
    }
    if (neck.r == head.r && neck.c == head.c - 1) {
        return 3;
    }
    if (neck.r == head.r && neck.c == head.c + 1) {
        return 1;
    }
    return 3;
}

static void initState(GameState *s) {
    char inputMap[SIZE][SIZE + 1];
    int r, c, d;
    Point head = {-1, -1};

    memset(s, 0, sizeof(*s));
    s->foodR = -1;
    s->foodC = -1;
    s->score = 0;
    s->step = 0;

    for (r = 0; r < SIZE; r++) {
        if (scanf("%20s", inputMap[r]) != 1) {
            exit(0);
        }
    }
    scanf("%d", &growN);

    for (r = 0; r < SIZE; r++) {
        for (c = 0; c < SIZE; c++) {
            if (inputMap[r][c] == '#' || inputMap[r][c] == 'O') {
                baseMap[r][c] = inputMap[r][c];
            } else {
                baseMap[r][c] = '.';
            }
            if (inputMap[r][c] == 'H') {
                head = (Point){r, c};
            } else if (inputMap[r][c] == 'F') {
                s->foodR = r;
                s->foodC = c;
            }
        }
        baseMap[r][SIZE] = '\0';
    }

    s->snake[0] = head;
    s->len = 3;

    for (d = 0; d < 4; d++) {
        r = head.r + dr[d];
        c = head.c + dc[d];
        if (inside(r, c) && inputMap[r][c] == 'B') {
            s->snake[1] = (Point){r, c};
            break;
        }
    }
    for (d = 0; d < 4; d++) {
        r = s->snake[1].r + dr[d];
        c = s->snake[1].c + dc[d];
        if (inside(r, c) && inputMap[r][c] == 'B' && !(r == head.r && c == head.c)) {
            s->snake[2] = (Point){r, c};
            break;
        }
    }
    s->curDir = inferCurrentDir(s->snake[0], s->snake[1]);
    rebuildMap(s);
}

int main(void) {
    GameState state, oldState;
    int dir, r, c;

    initState(&state);

    while (1) {
        dir = chooseDirection(&state);
        oldState = state;
        printf("%c\n%d\n", dirChar[dir], state.score);
        fflush(stdout);

        simulateMove(&state, dir);

        if (scanf("%d %d", &r, &c) != 2) {
            break;
        }
        if (r == 100 && c == 100) {
            printFinalState(&oldState);
            break;
        }
        if (r > 0 && r < 19 && c > 0 && c < 19) {
            state.foodR = r;
            state.foodC = c;
            rebuildMap(&state);
        }
    }
    return 0;
}
