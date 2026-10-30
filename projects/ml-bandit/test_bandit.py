import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location(
    "bandit", Path(__file__).with_name("01_bandit.py")
)
bandit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bandit)


def test_running_average():
    assert bandit.update_average(5, 2, 11) == 8


def test_best_action_without_exploration():
    rng = np.random.default_rng(1)
    assert all(bandit.choose_action(np.array([1., 5., 2.]), rng, epsilon=0) == 1
               for _ in range(50))


def test_reproducible_runs():
    for a, b in zip(bandit.one_run(5), bandit.one_run(5)):
        np.testing.assert_array_equal(a, b)


def test_learning_beats_random_on_average():
    runs = [bandit.one_run(seed) for seed in range(bandit.N_RUNS)]
    assert np.mean([x[0].sum() for x in runs]) > np.mean([x[1].sum() for x in runs])