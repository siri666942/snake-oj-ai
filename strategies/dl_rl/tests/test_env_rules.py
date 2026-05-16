from strategies.dl_rl.snake_ai.env import SnakeEnv


def test_food_growth_scores_once_with_natural_growth() -> None:
    grid = [
        "####################",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#...BHF............#",
        "#...B..............#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "####################",
    ]
    env = SnakeEnv().reset(seed=1, n=1, grid=grid)
    _, reward, done, info = env.step(3)
    assert not done
    assert info["ate"]
    assert env.score == 10
    assert len(env.snake) == 4
    assert reward > 0


def test_reverse_move_dies() -> None:
    env = SnakeEnv().reset(seed=1, n=32)
    _, _, done, info = env.step(1)
    assert done
    assert info["death_reason"] == "reverse"


def test_tail_cell_allowed_only_without_growth() -> None:
    grid = [
        "####################",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#...BH.............#",
        "#...BB.............#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#..................#",
        "#.................F#",
        "#..................#",
        "####################",
    ]
    env = SnakeEnv().reset(seed=1, n=32, grid=grid)
    assert len(env.snake) == 4
    assert env.clone().step(2)[2] is False
    grow_env = SnakeEnv().reset(seed=1, n=1, grid=grid)
    assert grow_env.clone().step(2)[2] is True
