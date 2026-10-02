"""
Pytest configuration and fixtures.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Config paths are relative to the project root
os.chdir(PROJECT_ROOT)

from src.data_ingestion import TelcoDataLoader  # noqa: E402
from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config() -> dict:
    """Parsed configs/project.yaml."""
    return load_config("configs/project.yaml")


@pytest.fixture(scope="session")
def economics_config() -> dict:
    """Parsed configs/economics.yaml."""
    return load_config("configs/economics.yaml")


@pytest.fixture(scope="session")
def experiment_config() -> dict:
    """Parsed configs/experiment.yaml."""
    return load_config("configs/experiment.yaml")


@pytest.fixture(scope="session")
def customers(config) -> pd.DataFrame:
    """The real cleaned dataset (7,043 customers)."""
    return TelcoDataLoader(config).run()


@pytest.fixture
def small_config(config, tmp_path, customers) -> dict:
    """
    Config pointing at a 1,500-customer stratified sample and a temp output folder,
    with a single CV repeat, so end-to-end tests stay fast.
    """
    raw = pd.read_csv(config["paths"]["raw_data"], keep_default_na=False, na_values=[""])
    sample = raw.groupby("Churn Label", group_keys=False).sample(
        frac=1500 / len(raw), random_state=0
    )
    sample_path = tmp_path / "telco_sample.csv"
    sample.to_csv(sample_path, index=False)

    small = {**config}
    small["paths"] = {
        **config["paths"],
        "raw_data": str(sample_path),
        "outputs": str(tmp_path / "outputs"),
        "figures": str(tmp_path / "outputs" / "figures"),
        "models": str(tmp_path / "models"),
    }
    small["cv"] = {"n_splits": 3, "n_repeats": 1}
    small["evaluation"] = {**config["evaluation"], "bootstrap_resamples": 100}
    return small


@pytest.fixture
def toy_scored() -> pd.DataFrame:
    """
    Scored customers with calibrated probabilities and outcomes drawn from them.

    Because outcomes are sampled from the probabilities, expected and realized
    profit should agree up to sampling noise.
    """
    rng = np.random.default_rng(0)
    n = 20000
    probability = rng.beta(0.8, 1.6, n)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "churn_probability": probability,
            "churn": rng.binomial(1, probability),
            "monthly_charge": rng.uniform(20, 110, n),
            "contract": rng.choice(
                ["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.2]
            ),
            "internet_type": rng.choice(["Fiber Optic", "DSL", "None"], n, p=[0.45, 0.35, 0.2]),
            "tenure_months": rng.integers(1, 72, n),
            "payment_method": rng.choice(["Bank Withdrawal", "Mailed Check", "Credit Card"], n),
            "premium_tech_support": rng.integers(0, 2, n),
            "online_security": rng.integers(0, 2, n),
        }
    )
