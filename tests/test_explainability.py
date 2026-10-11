"""
Tests for odds ratios, permutation importance, and SHAP aggregation.
"""

import numpy as np
import pytest
from sklearn.base import clone

from src.explainability import (
    odds_ratio_table,
    permutation_importance_table,
    shap_importance_table,
    shap_values_by_feature,
)
from src.modeling import build_candidates, prepare_features


@pytest.fixture(scope="module")
def odds(customers):
    return odds_ratio_table(customers)


def test_two_year_contract_sharply_lowers_churn_odds(odds):
    row = odds[odds["term"].str.startswith("Contract: Two Year")].iloc[0]
    assert row["odds_ratio"] < 0.15
    assert row["ci_high"] < 0.15
    assert row["significant"]


def test_longer_tenure_lowers_odds_and_higher_price_raises_them(odds):
    tenure = odds[odds["term"].str.startswith("Tenure")].iloc[0]
    price = odds[odds["term"].str.startswith("Monthly charge")].iloc[0]
    assert tenure["odds_ratio"] < 1 < price["odds_ratio"]


def test_confidence_intervals_contain_the_estimate(odds):
    assert (odds["ci_low"] <= odds["odds_ratio"]).all()
    assert (odds["odds_ratio"] <= odds["ci_high"]).all()


def test_confounded_columns_are_not_in_the_inferential_model(odds):
    """Offers are nearly a function of tenure band, so their coefficients would mislead."""
    assert not odds["term"].str.contains("Offer").any()
    assert not odds["term"].str.contains("referral", case=False).any()


@pytest.fixture(scope="module")
def fitted(customers, config):
    X, numeric, categorical = prepare_features(customers, config)
    model = clone(build_candidates(config, numeric, categorical)["hist_gb"]).fit(
        X, customers["churn"]
    )
    return model, X, customers["churn"]


def test_permutation_importance_ranks_contract_first(fitted):
    model, X, y = fitted
    table = permutation_importance_table(model, X, y, n_repeats=3)
    assert table["feature"].iloc[0] == "contract"
    assert table["auc_drop"].iloc[0] > 0.05


def test_shap_aggregates_one_column_per_original_feature(fitted):
    model, X, _ = fitted
    values = shap_values_by_feature(
        model, X.sample(100, random_state=0), X.sample(60, random_state=1)
    )

    assert list(values.columns) == list(X.columns)
    assert len(values) == 60
    assert np.isfinite(values.to_numpy()).all()

    ranking = shap_importance_table(values)
    assert ranking["feature"].iloc[0] == "contract"
