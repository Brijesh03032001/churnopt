"""
Action plan: who to contact, why they are at risk, and what to try first.

The drivers come from simple, readable flags tied to the strongest associations
in the odds-ratio analysis. The suggested actions are templates to TEST, not
proven treatments: the data is observational, and the save rate of each action
is exactly what the randomized holdout should measure.
"""

from typing import List

import pandas as pd


def _has(value) -> bool:
    """True for a Yes/No text flag or a 0/1 numeric flag."""
    return value in ("Yes", 1, True)


# (flag name, rule, suggested action). The first matching flag sets the action.
DRIVER_RULES = [
    (
        "month-to-month contract",
        lambda r: r["contract"] == "Month-to-Month",
        "Offer a discounted 12-month contract",
    ),
    (
        "no tech support or security",
        lambda r: r["internet_type"] != "None"
        and not _has(r["premium_tech_support"])
        and not _has(r["online_security"]),
        "Offer a free tech-support / security trial",
    ),
    (
        "tenure under 12 months",
        lambda r: r["tenure_months"] < 12,
        "Onboarding check-in call with a loyalty credit",
    ),
    (
        "mailed-check payment",
        lambda r: r["payment_method"] == "Mailed Check",
        "Offer a bill credit for switching to automatic payment",
    ),
    (
        "high monthly charge (over $85)",
        lambda r: r["monthly_charge"] > 85,
        "Review the plan for a cheaper bundle",
    ),
]
DEFAULT_ACTION = "Retention call"


def add_risk_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach readable risk drivers and a first action to each customer.

    Args:
        df: Customers with contract, internet_type, premium_tech_support,
            online_security, tenure_months, payment_method, monthly_charge

    Returns:
        Copy with `risk_drivers` (semicolon-separated), `suggested_action` (first
        matching rule) and `secondary_action` (second matching rule, if any)
    """
    df = df.copy()
    drivers: List[str] = []
    actions: List[str] = []
    secondary: List[str] = []

    for _, row in df.iterrows():
        matched = [(name, action) for name, rule, action in DRIVER_RULES if rule(row)]
        drivers.append("; ".join(name for name, _ in matched) or "none of the tracked flags")
        actions.append(matched[0][1] if matched else DEFAULT_ACTION)
        secondary.append(matched[1][1] if len(matched) > 1 else "")

    df["risk_drivers"] = drivers
    df["suggested_action"] = actions
    df["secondary_action"] = secondary
    return df


def build_action_plan(scored: pd.DataFrame, contacts: int) -> pd.DataFrame:
    """
    The contact list for the campaign.

    Args:
        scored: Customers scored by `score_customers` (sorted by expected value)
        contacts: Number of customers to contact

    Returns:
        Top `contacts` customers with rank, value, and suggested action.
        The `churn` outcome is dropped because a live plan would not have it.
    """
    plan = add_risk_drivers(scored.head(contacts))
    plan.insert(0, "rank", range(1, len(plan) + 1))
    columns = [
        "rank",
        "customer_id",
        "churn_probability",
        "monthly_charge",
        "customer_value",
        "expected_value",
        "breakeven_probability",
        "contract",
        "internet_type",
        "tenure_months",
        "payment_method",
        "risk_drivers",
        "suggested_action",
        "secondary_action",
    ]
    return plan[columns].round(
        {
            "churn_probability": 3,
            "customer_value": 1,
            "expected_value": 1,
            "breakeven_probability": 3,
        }
    )
