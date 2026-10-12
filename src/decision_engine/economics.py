"""
Retention economics: turn calibrated churn probabilities into contact decisions.

Expected value of contacting customer i:

    EV_i = p_i * save_rate * V_i - cost_per_contact

p_i is the calibrated probability the customer churns, V_i the margin kept if
they are saved, and the cost is paid for EVERY contacted customer, so offers
wasted on customers who would have stayed are already priced in.

The save rate and the cost are assumptions (configs/economics.yaml); the
dataset contains no campaign outcomes. The sensitivity functions show how far
the conclusions depend on them.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..utils import setup_logging

logger = setup_logging(__name__)


@dataclass(frozen=True)
class RetentionEconomics:
    """Assumptions of the retention campaign."""

    cost_per_contact: float
    save_rate: float
    gross_margin: float
    horizon_months: int
    monthly_discount_rate: float

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RetentionEconomics":
        """
        Build from the parsed economics.yaml.

        Args:
            config: Configuration dictionary from economics.yaml

        Returns:
            RetentionEconomics instance
        """
        return cls(
            cost_per_contact=float(config["offer"]["cost_per_contact"]),
            save_rate=float(config["offer"]["save_rate"]),
            gross_margin=float(config["value"]["gross_margin"]),
            horizon_months=int(config["value"]["horizon_months"]),
            monthly_discount_rate=float(config["value"]["monthly_discount_rate"]),
        )

    def with_overrides(self, **kwargs) -> "RetentionEconomics":
        """Copy with some assumptions changed, for sensitivity analysis."""
        return replace(self, **kwargs)

    @property
    def annuity_factor(self) -> float:
        """Present value of one unit of monthly margin over the horizon."""
        rate = self.monthly_discount_rate
        months = np.arange(1, self.horizon_months + 1)
        return float(np.sum((1 + rate) ** -months))

    def customer_value(self, monthly_charge) -> pd.Series:
        """Margin kept, in present-value terms, if the customer is saved."""
        return pd.Series(monthly_charge) * self.gross_margin * self.annuity_factor

    def expected_value(self, churn_probability, monthly_charge) -> pd.Series:
        """Expected net value of contacting each customer."""
        value = self.customer_value(monthly_charge)
        return pd.Series(churn_probability) * self.save_rate * value - self.cost_per_contact

    def breakeven_probability(self, monthly_charge) -> pd.Series:
        """Churn probability above which contacting the customer has positive EV."""
        value = self.customer_value(monthly_charge)
        return self.cost_per_contact / (self.save_rate * value)


def score_customers(df: pd.DataFrame, economics: RetentionEconomics) -> pd.DataFrame:
    """
    Attach value, expected value, and break-even probability to each customer.

    Args:
        df: Predictions with `churn_probability` and `monthly_charge`
        economics: Campaign assumptions

    Returns:
        Copy sorted by expected value, highest first
    """
    scored = df.copy()
    scored["customer_value"] = economics.customer_value(scored["monthly_charge"]).to_numpy()
    scored["expected_value"] = economics.expected_value(
        scored["churn_probability"].to_numpy(), scored["monthly_charge"].to_numpy()
    ).to_numpy()
    scored["breakeven_probability"] = economics.breakeven_probability(
        scored["monthly_charge"]
    ).to_numpy()
    return scored.sort_values("expected_value", ascending=False).reset_index(drop=True)


def realized_net(
    df: pd.DataFrame, economics: RetentionEconomics, selected: np.ndarray
) -> Dict[str, float]:
    """
    Backtest a contact list against the churn outcomes that actually happened.

    Customers who churned and were contacted contribute save_rate * V (the save
    rate is still an assumption); every contacted customer costs the contact fee.

    Args:
        df: Scored customers with `churn` and `customer_value`
        economics: Campaign assumptions
        selected: Positions (integer array) or boolean mask of contacted customers

    Returns:
        Dictionary with contacts, cost, value, net, roi, churners reached, precision
    """
    chosen = df.iloc[selected]
    contacts = len(chosen)
    cost = contacts * economics.cost_per_contact
    value = float((chosen["churn"] * economics.save_rate * chosen["customer_value"]).sum())
    net = value - cost
    return {
        "contacts": contacts,
        "cost": cost,
        "value": value,
        "net": net,
        "roi": net / cost if cost > 0 else 0.0,
        "churners_reached": int(chosen["churn"].sum()),
        "precision": float(chosen["churn"].mean()) if contacts else 0.0,
        "share_of_churners_reached": float(chosen["churn"].sum() / df["churn"].sum()),
    }


def profit_curve(df: pd.DataFrame, economics: RetentionEconomics) -> pd.DataFrame:
    """
    Cumulative expected and realized profit as contacts go down the ranked list.

    The curve is non-monotonic because each added contact costs the same but
    brings less expected value; it peaks where the next customer's EV turns
    negative. The expected and realized curves agreeing is a calibration check
    expressed in dollars.

    Args:
        df: Scored customers (output of score_customers)
        economics: Campaign assumptions

    Returns:
        DataFrame indexed by number of contacts
    """
    ranked = df.sort_values("expected_value", ascending=False).reset_index(drop=True)
    contacts = np.arange(1, len(ranked) + 1)
    realized_gain = ranked["churn"] * economics.save_rate * ranked["customer_value"]
    return pd.DataFrame(
        {
            "contacts": contacts,
            "cost": contacts * economics.cost_per_contact,
            "expected_net": ranked["expected_value"].cumsum().to_numpy(),
            "realized_net": (realized_gain - economics.cost_per_contact).cumsum().to_numpy(),
            "marginal_ev": ranked["expected_value"].to_numpy(),
        }
    )


def optimal_contacts(curve: pd.DataFrame) -> int:
    """Number of contacts that maximizes expected net profit (0 if none is profitable)."""
    if curve["expected_net"].max() <= 0:
        return 0
    return int(curve.loc[curve["expected_net"].idxmax(), "contacts"])


def budget_contacts(economics: RetentionEconomics, budget_config: Dict[str, Any]) -> int:
    """
    Contacts the campaign can afford and staff.

    Args:
        economics: Campaign assumptions
        budget_config: `budget` section of economics.yaml

    Returns:
        Maximum number of contacts allowed by budget and capacity
    """
    affordable = int(budget_config["max_campaign_budget"] // economics.cost_per_contact)
    capacity = budget_config.get("max_contacts")
    return min(affordable, capacity) if capacity else affordable


def baseline_rule_mask(df: pd.DataFrame, rule: Dict[str, str]) -> np.ndarray:
    """Customers a business would flag without a model (e.g. month-to-month fiber)."""
    return (
        (df["contract"] == rule["contract"]) & (df["internet_type"] == rule["internet_type"])
    ).to_numpy()


def compare_policies(
    df: pd.DataFrame, economics: RetentionEconomics, k: int, rule: Dict[str, str], seed: int = 42
) -> pd.DataFrame:
    """
    Backtest contact policies at the same number of contacts.

    Args:
        df: Scored customers with outcomes
        economics: Campaign assumptions
        k: Contacts allowed
        rule: Baseline rule from economics.yaml
        seed: Random seed for the random policy and the rule's tie-break

    Returns:
        DataFrame with one row per policy: realized net, ROI, precision
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    by_ev = df.sort_values("expected_value", ascending=False).index.to_numpy()
    by_prob = df.sort_values("churn_probability", ascending=False).index.to_numpy()
    rule_idx = np.flatnonzero(baseline_rule_mask(df, rule))
    rule_pick = rng.choice(rule_idx, size=min(k, len(rule_idx)), replace=False)

    # Random policy averaged over many draws, so one lucky draw does not mislead
    random_rows = [
        realized_net(df, economics, rng.choice(n, size=k, replace=False)) for _ in range(200)
    ]
    random_mean = {key: float(np.mean([r[key] for r in random_rows])) for key in random_rows[0]}

    policies = {
        "Model: rank by expected value": realized_net(df, economics, by_ev[:k]),
        "Model: rank by churn probability": realized_net(df, economics, by_prob[:k]),
        f"Rule: {rule['contract']} + {rule['internet_type']}": realized_net(
            df, economics, rule_pick
        ),
        "Random contacts": random_mean,
    }
    table = pd.DataFrame(policies).T.reset_index(names="policy")
    table["contacts"] = table["contacts"].round().astype(int)
    return table


def breakeven_save_rate(df: pd.DataFrame, economics: RetentionEconomics) -> pd.DataFrame:
    """
    Lowest save rate at which a top-k campaign still breaks even, for every k.

    Net(k) = s * sum_{i<=k}(p_i V_i) - c * k, so break-even s*(k) = c k / sum(p_i V_i).
    Customers are ranked by p * V, which does not depend on s or c.

    Args:
        df: Scored customers
        economics: Campaign assumptions

    Returns:
        DataFrame with contacts and the break-even save rate
    """
    pv = (df["churn_probability"] * df["customer_value"]).sort_values(ascending=False).to_numpy()
    contacts = np.arange(1, len(pv) + 1)
    return pd.DataFrame(
        {
            "contacts": contacts,
            "breakeven_save_rate": economics.cost_per_contact * contacts / np.cumsum(pv),
        }
    )


def optimal_net_profit(
    df: pd.DataFrame, economics: RetentionEconomics, max_contacts: Optional[int] = None
) -> Dict[str, float]:
    """
    Best achievable expected net profit under the given assumptions.

    Args:
        df: Scored customers
        economics: Campaign assumptions
        max_contacts: Optional cap on contacts

    Returns:
        Dictionary with contacts, expected net profit, and cost at the optimum
    """
    pv = (df["churn_probability"] * df["customer_value"]).sort_values(ascending=False).to_numpy()
    contacts = np.arange(1, len(pv) + 1)
    net = economics.save_rate * np.cumsum(pv) - economics.cost_per_contact * contacts
    if max_contacts is not None:
        net, contacts = net[:max_contacts], contacts[:max_contacts]
    best = int(np.argmax(net))
    if net[best] <= 0:
        return {"contacts": 0, "expected_net": 0.0, "cost": 0.0}
    return {
        "contacts": int(contacts[best]),
        "expected_net": float(net[best]),
        "cost": float(contacts[best] * economics.cost_per_contact),
    }


def sensitivity_grid(
    df: pd.DataFrame,
    economics: RetentionEconomics,
    save_rates,
    costs,
    budget_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Optimal expected profit over a grid of save rates and contact costs.

    The budget is a dollar amount, so a higher cost per contact means fewer
    contacts are affordable.

    Args:
        df: Scored customers
        economics: Base-case assumptions
        save_rates: Save rates to try
        costs: Cost per contact values to try
        budget_config: `budget` section of economics.yaml; None = unconstrained

    Returns:
        Long DataFrame: save_rate, cost_per_contact, contacts, expected_net
    """
    rows = []
    for save_rate in save_rates:
        for cost in costs:
            variant = economics.with_overrides(save_rate=save_rate, cost_per_contact=cost)
            limit = budget_contacts(variant, budget_config) if budget_config else None
            best = optimal_net_profit(df, variant, limit)
            rows.append({"save_rate": save_rate, "cost_per_contact": cost, **best})
    return pd.DataFrame(rows)


def tornado(
    df: pd.DataFrame,
    economics: RetentionEconomics,
    sensitivity_config: Dict[str, Any],
    budget_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    One-at-a-time sensitivity: how far does each assumption swing the profit?

    Args:
        df: Scored customers
        economics: Base-case assumptions
        sensitivity_config: `sensitivity` section of economics.yaml
        budget_config: `budget` section of economics.yaml; None = unconstrained

    Returns:
        DataFrame with low and high net profit per assumption, widest swing first.
        Customer value depends on margin and horizon, so the scored frame is
        rebuilt for each variant.
    """
    base_df = score_customers(df, economics)
    base_limit = budget_contacts(economics, budget_config) if budget_config else None
    base = optimal_net_profit(base_df, economics, base_limit)["expected_net"]

    parameters = {
        "save_rate": ("Save rate", sensitivity_config["save_rate"]),
        "cost_per_contact": ("Cost per contact", sensitivity_config["cost_per_contact"]),
        "gross_margin": ("Gross margin", sensitivity_config["gross_margin"]),
        "horizon_months": ("Value horizon", sensitivity_config["horizon_months"]),
    }

    rows = []
    for field, (label, values) in parameters.items():
        results = {}
        for value in (min(values), max(values)):
            variant = economics.with_overrides(**{field: value})
            variant_df = score_customers(df, variant)
            limit = budget_contacts(variant, budget_config) if budget_config else None
            results[value] = optimal_net_profit(variant_df, variant, limit)["expected_net"]
        low_value, high_value = min(values), max(values)
        rows.append(
            {
                "assumption": label,
                "low_value": low_value,
                "high_value": high_value,
                "net_at_low": results[low_value],
                "net_at_high": results[high_value],
                "base_net": base,
                "swing": abs(results[high_value] - results[low_value]),
            }
        )
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def bootstrap_policy_net(
    df: pd.DataFrame,
    economics: RetentionEconomics,
    k: int,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap interval for the realized net profit of the model's top-k policy.

    Customers are resampled with replacement and the top-k list is rebuilt from
    each resample's expected values, so the interval reflects both ranking and
    outcome noise.

    Args:
        df: Scored customers with outcomes
        economics: Campaign assumptions
        k: Contacts allowed
        n_resamples: Number of resamples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        Dictionary with mean, lower, and upper realized net profit, and the
        share of resamples with positive net profit
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    ev = df["expected_value"].to_numpy()
    gain = (df["churn"] * economics.save_rate * df["customer_value"]).to_numpy()

    nets = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        top = idx[np.argsort(-ev[idx])[:k]]
        nets[i] = gain[top].sum() - k * economics.cost_per_contact

    alpha = (1 - ci_level) / 2
    return {
        "mean": float(nets.mean()),
        "lower": float(np.quantile(nets, alpha)),
        "upper": float(np.quantile(nets, 1 - alpha)),
        "share_positive": float((nets > 0).mean()),
    }
