"""
Design of the randomized retention test.

The decision analysis assumes a save rate. This module sizes the experiment
that would measure it. It reports sample sizes and power, never outcomes:
there is no campaign data in this project, and simulated "results" would only
reproduce the assumptions that went into them.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from ..utils import setup_logging

logger = setup_logging(__name__)


def treated_churn_rate(baseline_rate: float, save_rate: float) -> float:
    """Churn rate among contacted customers if a share `save_rate` of churners is saved."""
    return baseline_rate * (1 - save_rate)


def required_sample_size(
    baseline_rate: float,
    save_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> int:
    """
    Customers needed in the treatment arm to detect a given save rate.

    Args:
        baseline_rate: Churn rate without intervention in the contacted segment
        save_rate: Share of would-be churners the offer retains
        alpha: Significance level
        power: Desired power
        alternative: 'two-sided' or 'larger'
        ratio: Control-to-treatment size ratio

    Returns:
        Treatment-arm sample size (control arm is `ratio` times this)
    """
    effect = proportion_effectsize(baseline_rate, treated_churn_rate(baseline_rate, save_rate))
    n = NormalIndPower().solve_power(
        effect_size=effect, alpha=alpha, power=power, ratio=ratio, alternative=alternative
    )
    return int(np.ceil(n))


def detectable_save_rate(
    n_treatment: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> float:
    """
    Smallest save rate a test of a given size can detect with the requested power.

    Inverts the Cohen's h effect size used by `required_sample_size`.

    Args:
        n_treatment: Treatment-arm size
        baseline_rate: Churn rate without intervention in the contacted segment
        alpha: Significance level
        power: Desired power
        alternative: 'two-sided' or 'larger'
        ratio: Control-to-treatment size ratio

    Returns:
        Minimum detectable save rate (share of would-be churners retained)
    """
    h = NormalIndPower().solve_power(
        nobs1=n_treatment, alpha=alpha, power=power, ratio=ratio, alternative=alternative
    )
    treated_rate = np.sin((2 * np.arcsin(np.sqrt(baseline_rate)) - h) / 2) ** 2
    return float(1 - treated_rate / baseline_rate)


def design_table(baseline_rate: float, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Sample sizes for each save rate the test should detect.

    Args:
        baseline_rate: Churn rate without intervention in the contacted segment
        config: Parsed experiment.yaml

    Returns:
        DataFrame with detectable difference and required customers per arm
    """
    test = config["test"]
    control_fraction = config["control_fraction"]
    ratio = control_fraction / (1 - control_fraction)

    rows = []
    for save_rate in config["save_rates_to_detect"]:
        n_treatment = required_sample_size(
            baseline_rate, save_rate, test["alpha"], test["power"], test["alternative"], ratio
        )
        n_control = int(np.ceil(n_treatment * ratio))
        rows.append(
            {
                "save_rate": save_rate,
                "churn_if_untreated": baseline_rate,
                "churn_if_treated": treated_churn_rate(baseline_rate, save_rate),
                "detectable_difference": baseline_rate * save_rate,
                "n_treatment": n_treatment,
                "n_control": n_control,
                "n_total": n_treatment + n_control,
            }
        )
    return pd.DataFrame(rows)


def simulate_power(
    n_treatment: int,
    n_control: int,
    baseline_rate: float,
    save_rate: float,
    alpha: float = 0.05,
    n_trials: int = 4000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo power of the two-proportion z-test for a known true effect.

    This validates the analytic sample size. It is a power check, not a result:
    the 'true' save rate is an input.

    Args:
        n_treatment: Treatment-arm size
        n_control: Control-arm size
        baseline_rate: Churn rate without intervention
        save_rate: Assumed true save rate
        alpha: Significance level
        n_trials: Number of simulated experiments
        seed: Random seed

    Returns:
        Share of simulated experiments that reject the null
    """
    rng = np.random.default_rng(seed)
    churned_treatment = rng.binomial(
        n_treatment, treated_churn_rate(baseline_rate, save_rate), n_trials
    )
    churned_control = rng.binomial(n_control, baseline_rate, n_trials)

    pooled = (churned_treatment + churned_control) / (n_treatment + n_control)
    standard_error = np.sqrt(pooled * (1 - pooled) * (1 / n_treatment + 1 / n_control))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (churned_control / n_control - churned_treatment / n_treatment) / standard_error
    p_values = 2 * stats.norm.sf(np.abs(np.nan_to_num(z)))
    return float((p_values < alpha).mean())


def validate_design(design: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Add simulated power next to the analytic target for every row of the design table.

    Args:
        design: Output of design_table
        config: Parsed experiment.yaml

    Returns:
        Design table with a `simulated_power` column
    """
    sim = config["simulation"]
    design = design.copy()
    design["simulated_power"] = [
        simulate_power(
            row.n_treatment,
            row.n_control,
            row.churn_if_untreated,
            row.save_rate,
            config["test"]["alpha"],
            sim["n_trials"],
            sim["seed"],
        )
        for row in design.itertuples()
    ]
    return design
