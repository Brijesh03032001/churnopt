"""
End-to-end tests on a 1,500-customer sample: train, decide, experiment.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import stage_decide, stage_experiment, stage_train


@pytest.fixture
def trained(small_config):
    metrics = stage_train(small_config)
    return small_config, metrics


def test_training_writes_every_artifact(trained):
    config, _ = trained
    out = Path(config["paths"]["outputs"])
    for name in [
        "metrics.json",
        "model_comparison.csv",
        "leakage_audit.csv",
        "predictions_oof.csv",
        "predictions_holdout.csv",
    ]:
        assert (out / name).exists(), name
    assert (Path(config["paths"]["models"]) / "churn_model.joblib").exists()


def test_champion_beats_the_baselines_on_the_holdout(trained):
    _, metrics = trained
    holdout, baselines = metrics["holdout"], metrics["holdout_baselines"]
    assert holdout["roc_auc"] > 0.80
    assert holdout["roc_auc"] > baselines["contract_only"]["roc_auc"] + 0.05
    assert holdout["log_loss"] < baselines["prior"]["log_loss"]


def test_probabilities_are_calibrated_not_inflated(trained):
    """The first version's mean predicted probability was 18x the real churn rate."""
    config, metrics = trained
    holdout = pd.read_csv(Path(config["paths"]["outputs"]) / "predictions_holdout.csv")
    ratio = holdout["churn_probability"].mean() / holdout["churn"].mean()
    assert 0.85 < ratio < 1.15
    assert metrics["holdout"]["ece"] < 0.08
    assert metrics["holdout"]["brier_skill"] > 0.2


def test_out_of_fold_predictions_cover_every_customer_once(trained):
    config, metrics = trained
    oof = pd.read_csv(Path(config["paths"]["outputs"]) / "predictions_oof.csv")
    assert len(oof) == metrics["n_customers"]
    assert oof["customer_id"].is_unique
    assert oof["churn_probability"].between(0, 1).all()


def test_leakage_audit_shows_the_leaky_columns_inflate_auc(trained):
    config, _ = trained
    audit = pd.read_csv(Path(config["paths"]["outputs"]) / "leakage_audit.csv").set_index(
        "feature_set"
    )
    standard = audit.loc["Standard: project default", "roc_auc"]
    assert audit.loc["Standard + satisfaction_score", "roc_auc"] > standard + 0.05
    assert abs(audit.loc["Standard + cltv", "roc_auc"] - standard) < 0.02


def test_decision_stage_produces_a_budget_respecting_plan(trained, economics_config, tmp_path):
    config, _ = trained
    summary = stage_decide(config, economics_config)
    out = Path(config["paths"]["outputs"])

    plan = pd.read_csv(out / "action_plan.csv")
    budget = economics_config["budget"]["max_campaign_budget"]
    cost = economics_config["offer"]["cost_per_contact"]

    assert len(plan) == summary["contacts_recommended"]
    assert len(plan) * cost <= budget
    assert (plan["expected_value"] >= 0).all()  # CSV rounds to 0.1, so a $0.04 EV shows as 0.0
    assert plan["expected_value"].is_monotonic_decreasing
    assert summary["recommended"]["expected_roi"] > 0
    assert summary["recommended"]["breakeven_save_rate"] < economics_config["offer"]["save_rate"]


def test_model_policy_beats_random_contacts_in_the_backtest(trained, economics_config):
    config, _ = trained
    summary = stage_decide(config, economics_config)
    nets = {row["policy"]: row["net"] for row in summary["policies"]}
    assert nets["Model: rank by expected value"] > nets["Random contacts"] + 1000


def test_experiment_stage_sizes_the_holdout_test(trained, economics_config, experiment_config):
    config, _ = trained
    stage_decide(config, economics_config)
    summary = stage_experiment(config, economics_config, experiment_config)

    assert summary["n_treatment_to_detect_breakeven"] > 0
    assert 0.7 < summary["simulated_power_min"] <= summary["simulated_power_max"] < 0.9

    saved = json.loads((Path(config["paths"]["outputs"]) / "experiment_summary.json").read_text())
    assert saved["breakeven_save_rate"] == pytest.approx(summary["breakeven_save_rate"])


def test_every_stage_runs_and_produces_all_figures(
    small_config, economics_config, experiment_config
):
    """Full contract: sql -> train -> decide -> experiment -> explain -> report."""
    from src.pipeline import stage_explain, stage_report, stage_sql

    stage_sql(small_config)
    stage_train(small_config)
    stage_decide(small_config, economics_config)
    stage_experiment(small_config, economics_config, experiment_config)
    stage_explain(small_config)
    summary = stage_report(small_config, economics_config, experiment_config)

    out = Path(small_config["paths"]["outputs"])
    assert len(summary["figures"]) == 14
    for path in summary["figures"].values():
        assert Path(path).stat().st_size > 10_000, path

    saved = json.loads((out / "results_summary.json").read_text())
    assert saved["data"]["customers"] == 1500
    assert (
        saved["model"]["with_satisfaction_auc"] > saved["model"]["standard_feature_set_auc"] + 0.05
    )
    assert saved["decision"]["recommended"]["expected_roi"] > 0
