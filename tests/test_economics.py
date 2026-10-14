"""
Tests for retention economics and the contact policy.

Several of these are regression tests for mistakes the first version of this
project made: reporting value/cost as ROI, and feeding uncalibrated
probabilities into dollar calculations.
"""

import numpy as np
import pandas as pd
import pytest

from src.decision_engine import (
    RetentionEconomics,
    baseline_rule_mask,
    breakeven_save_rate,
    budget_contacts,
    build_action_plan,
    compare_policies,
    optimal_contacts,
    optimal_net_profit,
    profit_curve,
    realized_net,
    score_customers,
    sensitivity_grid,
)


@pytest.fixture
def economics() -> RetentionEconomics:
    return RetentionEconomics(
        cost_per_contact=25.0,
        save_rate=0.30,
        gross_margin=0.40,
        horizon_months=12,
        monthly_discount_rate=0.0,
    )


def test_annuity_factor_equals_horizon_without_discounting(economics):
    assert economics.annuity_factor == pytest.approx(12)


def test_discounting_shrinks_customer_value(economics):
    discounted = economics.with_overrides(monthly_discount_rate=0.01)
    assert discounted.annuity_factor < economics.annuity_factor
    assert discounted.annuity_factor == pytest.approx(11.255, abs=0.001)


def test_expected_value_matches_hand_calculation(economics):
    # V = 100 * 0.40 * 12 = 480; EV = 0.5 * 0.30 * 480 - 25 = 47
    ev = economics.expected_value([0.5], [100.0])
    assert ev.iloc[0] == pytest.approx(47.0)


def test_expected_value_is_zero_at_the_breakeven_probability(economics):
    charge = pd.Series([40.0, 75.0, 110.0])
    breakeven = economics.breakeven_probability(charge)
    ev = economics.expected_value(breakeven.to_numpy(), charge.to_numpy())
    assert ev.abs().max() < 1e-9


def test_roi_is_net_over_cost_not_value_over_cost(economics):
    """The first version reported value/cost (105%) as ROI when net/cost was 4.6%."""
    df = pd.DataFrame(
        {"churn": [1, 1], "customer_value": [100.0 / economics.save_rate] * 2}
    )  # two churners, each worth exactly $100 once the save rate is applied
    result = realized_net(df, economics, np.array([0, 1]))
    assert result["cost"] == 50.0
    assert result["value"] == pytest.approx(200.0)
    assert result["net"] == pytest.approx(150.0)
    assert result["roi"] == pytest.approx(3.0)  # net / cost, not value / cost = 4.0


def test_cost_is_charged_for_every_contact_even_non_churners(economics):
    df = pd.DataFrame({"churn": [0, 0, 1], "customer_value": [500.0, 500.0, 500.0]})
    result = realized_net(df, economics, np.array([0, 1, 2]))
    assert result["cost"] == 75.0
    assert result["churners_reached"] == 1


def test_profit_peaks_where_marginal_value_turns_negative(toy_scored, economics):
    scored = score_customers(toy_scored, economics)
    curve = profit_curve(scored, economics)
    k = optimal_contacts(curve)

    assert k == int((scored["expected_value"] > 0).sum())
    assert 0 < k < len(scored)
    assert curve["expected_net"].iloc[k - 1] == curve["expected_net"].max()
    assert curve["expected_net"].iloc[-1] < curve["expected_net"].max()  # non-monotonic


def test_no_profitable_campaign_when_contact_is_too_expensive(toy_scored, economics):
    ruinous = economics.with_overrides(cost_per_contact=1000.0)
    curve = profit_curve(score_customers(toy_scored, ruinous), ruinous)
    assert optimal_contacts(curve) == 0


def test_expected_profit_tracks_realized_profit_when_probabilities_are_calibrated(
    toy_scored, economics
):
    scored = score_customers(toy_scored, economics)
    row = profit_curve(scored, economics).iloc[1999]
    assert row["realized_net"] == pytest.approx(row["expected_net"], rel=0.05)


def test_inflated_probabilities_overstate_expected_profit(toy_scored, economics):
    """What the first version's class-weighted model did to every dollar figure."""
    inflated = toy_scored.assign(
        churn_probability=np.clip(toy_scored["churn_probability"] * 1.6, 0, 1)
    )
    scored = score_customers(inflated, economics)
    row = profit_curve(scored, economics).iloc[1999]
    assert row["expected_net"] > 1.3 * row["realized_net"]


def test_budget_limits_contacts_by_cost_and_capacity(economics):
    assert budget_contacts(economics, {"max_campaign_budget": 20000.0, "max_contacts": None}) == 800
    assert budget_contacts(economics, {"max_campaign_budget": 20000.0, "max_contacts": 500}) == 500
    cheaper = economics.with_overrides(cost_per_contact=10.0)
    assert budget_contacts(cheaper, {"max_campaign_budget": 20000.0, "max_contacts": None}) == 2000


def test_policies_are_compared_at_the_same_number_of_contacts(toy_scored, economics):
    scored = score_customers(toy_scored, economics)
    rule = {"contract": "Month-to-Month", "internet_type": "Fiber Optic"}
    table = compare_policies(scored, economics, 800, rule)

    assert len(table) == 4
    assert (table["contacts"] == 800).all()
    by_policy = table.set_index("policy")["net"]
    assert by_policy["Model: rank by expected value"] > by_policy["Random contacts"]
    assert (
        by_policy["Model: rank by expected value"] >= by_policy["Model: rank by churn probability"]
    )


def test_baseline_rule_mask_selects_the_named_segment(toy_scored):
    rule = {"contract": "Month-to-Month", "internet_type": "Fiber Optic"}
    mask = baseline_rule_mask(toy_scored, rule)
    selected = toy_scored[mask]
    assert (selected["contract"] == "Month-to-Month").all()
    assert (selected["internet_type"] == "Fiber Optic").all()


def test_breakeven_save_rate_gives_zero_net_profit(toy_scored, economics):
    scored = score_customers(toy_scored, economics)
    breakeven = breakeven_save_rate(scored, economics)
    k = 800
    s_star = breakeven.loc[k - 1, "breakeven_save_rate"]

    at_breakeven = economics.with_overrides(save_rate=s_star)
    pv = (scored["churn_probability"] * scored["customer_value"]).sort_values(ascending=False)
    net = at_breakeven.save_rate * pv.iloc[:k].sum() - at_breakeven.cost_per_contact * k
    assert net == pytest.approx(0.0, abs=1e-6)


def test_sensitivity_profit_rises_with_save_rate_and_falls_with_cost(toy_scored, economics):
    budget = {"max_campaign_budget": 20000.0, "max_contacts": None}
    grid = sensitivity_grid(
        toy_scored.pipe(score_customers, economics),
        economics,
        [0.1, 0.3, 0.5],
        [15, 25, 60],
        budget,
    )
    pivot = grid.pivot(index="save_rate", columns="cost_per_contact", values="expected_net")

    assert (pivot.diff(axis=0).dropna() >= 0).all().all()  # more saves, more profit
    assert (pivot.diff(axis=1).dropna(axis=1) <= 0).all().all()  # higher cost, less profit


def test_optimal_profit_respects_the_contact_cap(toy_scored, economics):
    scored = score_customers(toy_scored, economics)
    capped = optimal_net_profit(scored, economics, max_contacts=300)
    uncapped = optimal_net_profit(scored, economics)
    assert capped["contacts"] == 300
    assert uncapped["expected_net"] > capped["expected_net"]


def test_action_plan_is_ranked_and_hides_the_outcome(toy_scored, economics):
    scored = score_customers(toy_scored, economics)
    plan = build_action_plan(scored, 50)

    assert plan["rank"].tolist() == list(range(1, 51))
    assert "churn" not in plan.columns  # a live plan would not know the outcome
    assert plan["expected_value"].is_monotonic_decreasing
    assert plan["suggested_action"].notna().all()
    assert plan["secondary_action"].notna().all()  # empty string when only one rule matches
    assert plan["risk_drivers"].notna().all()


def test_support_flags_work_with_yes_no_text_as_stored_in_the_predictions_file(toy_scored):
    """Regression: the rule once compared 'No' to 0, so it never fired on real data."""
    from src.decision_engine import add_risk_drivers

    customers = toy_scored.head(4).copy()
    customers["contract"] = "Two Year"
    customers["tenure_months"] = 30
    customers["payment_method"] = "Bank Withdrawal"
    customers["monthly_charge"] = 50.0
    customers["internet_type"] = ["DSL", "DSL", "DSL", "None"]
    customers["premium_tech_support"] = ["No", "Yes", "No", "No"]
    customers["online_security"] = ["No", "No", "Yes", "No"]

    result = add_risk_drivers(customers)
    flagged = result["risk_drivers"].str.contains("no tech support or security")
    assert flagged.tolist() == [
        True,
        False,
        False,
        False,
    ]  # has either, or no internet: not flagged
    assert result.loc[0, "suggested_action"].startswith("Offer a free tech-support")
