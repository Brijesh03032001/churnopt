"""
Tests for data loading, cleaning, validation, and splitting.
"""

import pandas as pd
import pytest

from src.data_ingestion import DataValidationError, TelcoDataLoader, to_snake_case


@pytest.fixture
def loader(config):
    return TelcoDataLoader(config)


def test_to_snake_case_handles_spaces_and_acronyms():
    assert to_snake_case("Avg Monthly GB Download") == "avg_monthly_gb_download"
    assert to_snake_case("Under 30") == "under_30"
    assert to_snake_case(" Customer ID ") == "customer_id"


def test_none_is_kept_as_a_category_not_missing(loader):
    """pandas parses the string 'None' as NaN by default; that silently corrupts two columns."""
    clean = loader.clean(loader.load_raw())
    assert clean["offer"].isna().sum() == 0
    assert clean["internet_type"].isna().sum() == 0
    assert (clean["offer"] == "None").sum() > 3000
    assert (clean["internet_type"] == "None").sum() == (clean["internet_service"] == "No").sum()


def test_target_is_binary_and_matches_customer_status(customers):
    assert set(customers["churn"].unique()) == {0, 1}
    assert ((customers["customer_status"] == "Churned") == (customers["churn"] == 1)).all()


def test_one_row_per_customer(customers):
    assert customers["customer_id"].is_unique
    assert len(customers) == 7043


def test_churn_rate_matches_published_dataset(customers):
    assert customers["churn"].mean() == pytest.approx(0.2654, abs=0.001)


def test_validation_rejects_duplicate_ids(loader, customers):
    broken = pd.concat([customers, customers.head(1)])
    with pytest.raises(DataValidationError, match="duplicate"):
        loader.validate(broken)


def test_validation_rejects_negative_tenure(loader, customers):
    broken = customers.copy()
    broken.loc[0, "tenure_months"] = -1
    with pytest.raises(DataValidationError, match="negative tenure"):
        loader.validate(broken)


def test_validation_rejects_label_status_disagreement(loader, customers):
    broken = customers.copy()
    flip = broken.index[broken["churn"] == 1][0]
    broken.loc[flip, "churn"] = 0
    with pytest.raises(DataValidationError, match="customer_status disagrees"):
        loader.validate(broken)


def test_split_is_disjoint_and_stratified(loader, customers):
    train, test = loader.split(customers)
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])
    assert len(train) + len(test) == len(customers)
    assert train["churn"].mean() == pytest.approx(test["churn"].mean(), abs=0.002)
    assert len(test) / len(customers) == pytest.approx(0.20, abs=0.005)


def test_split_is_reproducible(loader, customers):
    first, _ = loader.split(customers)
    second, _ = loader.split(customers)
    assert first["customer_id"].tolist() == second["customer_id"].tolist()
