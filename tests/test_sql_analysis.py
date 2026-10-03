"""
Tests for the SQL analysis layer (DuckDB over the raw CSV).
"""

import pytest

from src.analytics import query_files, run_sql_analysis


@pytest.fixture(scope="module")
def results(config):
    return run_sql_analysis(config, write_outputs=False)


def test_every_numbered_query_runs(config, results):
    assert len(query_files(config["paths"]["sql"])) == 6
    assert len(results) == 6
    assert all(len(frame) > 0 for frame in results.values())


def test_overview_matches_pandas(results, customers):
    overview = results["churn_overview"].iloc[0]
    assert overview["customers"] == len(customers)
    assert overview["churned_customers"] == customers["churn"].sum()
    assert overview["churn_rate_pct"] == pytest.approx(100 * customers["churn"].mean(), abs=0.1)
    lost = customers.loc[customers["churn"] == 1, "monthly_charge"].sum()
    assert overview["monthly_revenue_lost"] == pytest.approx(lost, abs=1)


def test_contract_tenure_shares_sum_to_one_hundred(results):
    table = results["churn_by_contract_and_tenure"]
    assert table["share_of_all_churners_pct"].sum() == pytest.approx(100, abs=0.5)
    assert table["churned"].sum() == 1869


def test_pareto_is_sorted_and_cumulates_to_one_hundred(results):
    pareto = results["revenue_at_risk_pareto"]
    assert pareto["monthly_revenue_lost"].is_monotonic_decreasing
    assert pareto["cumulative_share_pct"].iloc[-1] == pytest.approx(100, abs=0.2)
    top = pareto.iloc[0]
    assert (top["contract"], top["internet_type"]) == ("Month-to-Month", "Fiber Optic")


def test_offer_comparison_excludes_the_no_offer_group(results):
    offers = results["offer_confounding"]
    assert "None" not in set(offers["offer"])
    assert offers["offer"].nunique() == 5


def test_each_offer_sits_in_few_tenure_bands(results):
    """Why raw offer comparisons mislead: every offer lives in one or two tenure bands."""
    bands_per_offer = results["offer_confounding"].groupby("offer")["tenure_band"].nunique()
    assert bands_per_offer.max() <= 2
