"""
Candidate churn models.

Candidates run from trivial baselines to a gradient-boosted model, so every
gain in complexity has to be justified against a simpler alternative. Class
weights are deliberately NOT used: reweighting classes inflates predicted
probabilities, and the decision engine needs probabilities that mean what they
say.
"""

from typing import Any, Dict, List

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..utils import setup_logging
from .features import build_preprocessor

logger = setup_logging(__name__)

MODEL_DESCRIPTIONS = {
    "prior": "Predicts the base churn rate for everyone",
    "contract_only": "Logistic regression on contract type alone (the 'obvious rule')",
    "logistic": "Regularized logistic regression, all features",
    "hist_gb": "Gradient-boosted trees, all features",
    "hist_gb_calibrated": "Gradient-boosted trees with probability calibration",
}


def build_candidates(
    config: Dict[str, Any], numeric: List[str], categorical: List[str]
) -> Dict[str, BaseEstimator]:
    """
    Build the unfitted candidate models.

    Args:
        config: Configuration dictionary from project.yaml
        numeric: Numeric column names
        categorical: Categorical column names

    Returns:
        Dictionary mapping model name to an unfitted estimator
    """
    seed = config["random_seed"]
    logistic_cfg = config["models"]["logistic"]
    hist_gb_cfg = config["models"]["hist_gb"]
    calibration_cfg = config["calibration"]

    contract_only = Pipeline(
        [
            (
                "pre",
                ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["contract"])]),
            ),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    logistic = Pipeline(
        [
            ("pre", build_preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegressionCV(
                    Cs=logistic_cfg["Cs"],
                    cv=5,
                    scoring="neg_log_loss",
                    max_iter=logistic_cfg["max_iter"],
                    random_state=seed,
                ),
            ),
        ]
    )

    def make_hist_gb() -> Pipeline:
        return Pipeline(
            [
                ("pre", build_preprocessor(numeric, categorical)),
                ("model", HistGradientBoostingClassifier(random_state=seed, **hist_gb_cfg)),
            ]
        )

    return {
        "prior": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "contract_only": contract_only,
        "logistic": logistic,
        "hist_gb": make_hist_gb(),
        "hist_gb_calibrated": CalibratedClassifierCV(
            estimator=make_hist_gb(),
            method=calibration_cfg["method"],
            cv=calibration_cfg["cv"],
        ),
    }


def build_class_weighted_reference(
    config: Dict[str, Any], numeric: List[str], categorical: List[str]
) -> BaseEstimator:
    """
    Logistic regression with class_weight="balanced", kept only as a cautionary reference.

    Balancing classes raises ranking sensitivity but inflates every predicted
    probability, so the output cannot be used as a probability. It is never a
    candidate; the calibration figure compares the champion against it.

    Args:
        config: Configuration dictionary from project.yaml
        numeric: Numeric column names
        categorical: Categorical column names

    Returns:
        Unfitted pipeline
    """
    return Pipeline(
        [
            ("pre", build_preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=config["models"]["logistic"]["max_iter"],
                    random_state=config["random_seed"],
                ),
            ),
        ]
    )
