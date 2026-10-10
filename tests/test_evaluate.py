"""
Tests for evaluation metrics, calibration, and champion selection.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.modeling import (
    bootstrap_ci,
    calibration_table,
    cross_validate_models,
    evaluate_predictions,
    expected_calibration_error,
    gain_table,
    select_champion,
)


@pytest.fixture
def calibrated_sample():
    rng = np.random.default_rng(1)
    probability = rng.uniform(0.02, 0.98, 20000)
    return rng.binomial(1, probability), probability


def test_ece_is_small_for_calibrated_probabilities(calibrated_sample):
    y, p = calibrated_sample
    assert expected_calibration_error(y, p) < 0.02


def test_ece_is_large_for_overconfident_probabilities(calibrated_sample):
    y, p = calibrated_sample
    assert expected_calibration_error(y, np.clip(p * 1.5, 0, 1)) > 0.10


def test_calibration_table_covers_every_customer(calibrated_sample):
    y, p = calibrated_sample
    table = calibration_table(y, p, n_bins=10)
    assert len(table) == 10
    assert table["customers"].sum() == len(y)
    assert table["mean_predicted"].is_monotonic_increasing


def test_base_rate_predictor_has_zero_skill_and_chance_auc(calibrated_sample):
    y, _ = calibrated_sample
    metrics = evaluate_predictions(y, np.full(len(y), y.mean()))
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["brier_skill"] == pytest.approx(0.0, abs=1e-6)


def test_perfect_predictor_has_full_skill():
    y = np.array([0, 1] * 500)
    metrics = evaluate_predictions(y, y.astype(float))
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["brier_skill"] > 0.99


def test_bootstrap_interval_brackets_the_point_estimate(calibrated_sample):
    y, p = calibrated_sample
    low, high = bootstrap_ci(y, p, roc_auc_score, n_resamples=200)
    assert low < roc_auc_score(y, p) < high
    assert high - low < 0.05


def test_gain_table_reaches_all_churners_and_ranks_deciles(calibrated_sample):
    y, p = calibrated_sample
    table = gain_table(y, p)
    assert table["cumulative_churners_captured"].iloc[-1] == pytest.approx(1.0)
    assert table["cumulative_customers"].iloc[-1] == pytest.approx(1.0)
    assert table["lift"].iloc[0] > 1 > table["lift"].iloc[-1]
    assert table["churn_rate"].is_monotonic_decreasing


def test_cross_validation_reports_every_model_and_metric():
    X, y = make_classification(n_samples=600, n_features=6, weights=[0.7], random_state=0)
    X, y = pd.DataFrame(X), pd.Series(y)
    models = {
        "prior": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(max_iter=500),
    }
    summary, folds = cross_validate_models(models, X, y, n_splits=3, n_repeats=2, seed=0)

    assert set(summary["model"]) == {"prior", "logistic"}
    assert len(folds) == 2 * 3 * 2
    scores = summary.set_index("model")
    assert scores.loc["logistic", "roc_auc_mean"] > scores.loc["prior", "roc_auc_mean"] + 0.3
    assert scores.loc["logistic", "log_loss_mean"] < scores.loc["prior", "log_loss_mean"]


def _summary(rows):
    return pd.DataFrame(
        [
            {
                "model": m,
                "log_loss_mean": mean,
                "log_loss_se": se,
                "roc_auc_mean": 0.9,
                "roc_auc_se": 0.01,
            }
            for m, mean, se in rows
        ]
    )


SELECTION = {
    "metric": "log_loss",
    "one_se_rule": True,
    "complexity_order": ["prior", "logistic", "hist_gb"],
}


def test_one_se_rule_prefers_the_simpler_model_when_within_error():
    summary = _summary(
        [("prior", 0.58, 0.001), ("logistic", 0.340, 0.01), ("hist_gb", 0.335, 0.01)]
    )
    champion, rationale = select_champion(summary, SELECTION)
    assert champion == "logistic"
    assert "within 1 SE" in rationale


def test_clearly_better_complex_model_still_wins():
    summary = _summary(
        [("prior", 0.58, 0.001), ("logistic", 0.40, 0.005), ("hist_gb", 0.33, 0.005)]
    )
    champion, _ = select_champion(summary, SELECTION)
    assert champion == "hist_gb"
