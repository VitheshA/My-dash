from train import run


def test_model_beats_majority_baseline():
    metrics = run()
    assert 0 <= metrics["holdout_accuracy"] <= 1
    assert metrics["holdout_accuracy"] > metrics["majority_baseline_accuracy"]
    assert metrics["training_cv_accuracy"] > 0.6
