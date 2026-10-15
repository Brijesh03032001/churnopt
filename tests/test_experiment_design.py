"""
Tests for the randomized-test design calculations.
"""

import pytest

from src.experiments import (
    design_table,
    detectable_save_rate,
    required_sample_size,
    simulate_power,
    treated_churn_rate,
    validate_design,
)


def test_treated_churn_rate_scales_with_save_rate():
    assert treated_churn_rate(0.8, 0.25) == pytest.approx(0.6)
    assert treated_churn_rate(0.8, 0.0) == pytest.approx(0.8)


def test_sample_size_matches_textbook_value():
    # 50% vs 40% at 80% power, 5% two-sided: about 388 per arm
    n = required_sample_size(0.5, save_rate=0.2)
    assert 380 <= n <= 395


def test_smaller_effects_need_more_customers():
    sizes = [required_sample_size(0.6, s) for s in (0.30, 0.20, 0.10, 0.05)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 10 * sizes[0]


def test_detectable_save_rate_inverts_the_sample_size(economics_config=None):
    n = required_sample_size(0.7, save_rate=0.12)
    assert detectable_save_rate(n, 0.7) == pytest.approx(0.12, abs=0.005)


def test_simulated_power_matches_the_analytic_target():
    n = required_sample_size(0.7, save_rate=0.15)
    power = simulate_power(n, n, 0.7, 0.15, n_trials=4000, seed=1)
    assert power == pytest.approx(0.80, abs=0.03)


def test_simulated_power_is_near_alpha_when_there_is_no_effect():
    rate = simulate_power(500, 500, 0.7, 0.0, alpha=0.05, n_trials=4000, seed=1)
    assert rate == pytest.approx(0.05, abs=0.02)


def test_design_table_covers_every_requested_save_rate(experiment_config):
    table = design_table(0.8, experiment_config)
    assert table["save_rate"].tolist() == experiment_config["save_rates_to_detect"]
    assert (table["n_total"] == table["n_treatment"] + table["n_control"]).all()
    assert table["n_treatment"].is_monotonic_decreasing


def test_validated_design_hits_the_power_target(experiment_config):
    fast = {**experiment_config, "simulation": {"n_trials": 3000, "seed": 3}}
    table = validate_design(design_table(0.8, fast), fast)
    assert table["simulated_power"].between(0.76, 0.84).all()
