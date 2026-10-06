"""
Tests for feature preparation and the leakage guard.
"""

import copy

import pytest

from src.modeling import build_candidates, excluded_columns, prepare_features

LEAKY = [
    "satisfaction_score",
    "churn_score",
    "churn_reason",
    "churn_category",
    "customer_status",
    "churn_label",
    "cltv",
]


def test_leaky_columns_are_excluded_from_the_feature_list(config):
    excluded = excluded_columns(config)
    for column in LEAKY:
        assert column in excluded


@pytest.mark.parametrize("feature_set", ["standard", "strict"])
def test_model_inputs_never_contain_leaky_columns(customers, config, feature_set):
    """The guard that matters: nothing recorded at or after the outcome reaches the model."""
    cfg = copy.deepcopy(config)
    cfg["data"]["feature_set"] = feature_set
    X, _, _ = prepare_features(customers, cfg)
    assert set(LEAKY).isdisjoint(X.columns)
    assert "churn" not in X.columns
    assert "customer_id" not in X.columns


def test_strict_set_drops_lifetime_cumulative_columns(customers, config):
    cfg = copy.deepcopy(config)
    cfg["data"]["feature_set"] = "strict"
    strict, _, _ = prepare_features(customers, cfg)
    standard, _, _ = prepare_features(customers, config)

    for column in config["data"]["cumulative_columns"]:
        assert column in standard.columns
        assert column not in strict.columns


def test_extra_columns_can_be_added_back_for_the_audit(customers, config):
    X, numeric, _ = prepare_features(customers, config, extra_columns=["satisfaction_score"])
    assert "satisfaction_score" in X.columns
    assert "satisfaction_score" in numeric


def test_yes_no_columns_become_numeric_flags(customers, config):
    X, numeric, categorical = prepare_features(customers, config)
    assert "paperless_billing" in numeric
    assert set(X["paperless_billing"].unique()) <= {0, 1}
    assert "contract" in categorical
    assert "gender" in categorical


def test_no_missing_values_reach_the_model(customers, config):
    X, _, _ = prepare_features(customers, config)
    assert X.isna().sum().sum() == 0


def test_derived_addon_count_is_within_bounds(customers, config):
    X, _, _ = prepare_features(customers, config)
    assert X["num_addon_services"].between(0, 7).all()


def test_candidates_do_not_use_class_weights(customers, config):
    """Class weighting inflates probabilities; the decision math needs real ones."""
    X, numeric, categorical = prepare_features(customers, config)
    models = build_candidates(config, numeric, categorical)

    assert models["logistic"].named_steps["model"].class_weight is None
    assert models["hist_gb"].named_steps["model"].class_weight is None
    assert models["contract_only"].named_steps["model"].class_weight is None
