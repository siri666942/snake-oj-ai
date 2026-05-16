import numpy as np

from strategies.dl_rl.snake_ai.env import SnakeEnv
from strategies.dl_rl.snake_ai.value_data import episode_value_samples


def test_value_labels_keep_earned_score_after_death() -> None:
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
    env = SnakeEnv().reset(seed=1, n=32, grid=grid)
    actions = iter([3, 1])

    def policy(_env: SnakeEnv) -> int:
        return next(actions)

    features, scores, final_score = episode_value_samples(env, policy)
    labels = np.asarray([final_score - score for score in scores], dtype=np.float32)
    assert len(features) == 2
    assert scores == [0, 10]
    assert final_score == 10
    assert labels.tolist() == [10.0, 0.0]
    assert labels.min() >= 0.0
