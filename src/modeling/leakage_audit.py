"""
Leakage audit: what each suspect column would do to the model if it were allowed in.

A very high score is not good news. Satisfaction Score, Churn Score and Churn
Reason are recorded at or after the outcome, so a model that uses them looks
brilliant on paper and is useless for deciding whom to contact beforehand.
This audit puts a number on that, one feature set at a time.
"""

import copy
from typing import Any, Dict, List

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..data_ingestion import TARGET
from ..utils import setup_logging
from .evaluate import evaluate_predictions
from .features import prepare_features
from .models import build_candidates

logger = setup_logging(__name__)

# (label, feature_set, extra columns allowed back in, verdict)
AUDIT_SETS = [
    ("Strict: no lifetime-cumulative columns", "strict", [], "conservative sensitivity check"),
    ("Standard: project default", "standard", [], "used"),
    ("Standard + cltv", "standard", ["cltv"], "excluded: adds nothing, not reproducible"),
    (
        "Standard + satisfaction_score",
        "standard",
        ["satisfaction_score"],
        "excluded: recorded around exit",
    ),
    (
        "Standard + churn_score",
        "standard",
        ["churn_score"],
        "excluded: another model's output",
    ),
    (
        "Standard + satisfaction_score + churn_score",
        "standard",
        ["satisfaction_score", "churn_score"],
        "excluded: leakage",
    ),
]


def run_leakage_audit(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Cross-validate a logistic regression on progressively leakier feature sets.

    Args:
        df: Cleaned customer dataframe (training portion)
        config: Configuration dictionary from project.yaml

    Returns:
        DataFrame with one row per feature set: metrics, the change in AUC
        versus the standard set, and a verdict
    """
    seed = config["random_seed"]
    y = df[TARGET]
    cv = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=seed)

    rows: List[Dict[str, Any]] = []
    for label, feature_set, extra, verdict in AUDIT_SETS:
        audit_config = copy.deepcopy(config)
        audit_config["data"]["feature_set"] = feature_set

        X, numeric, categorical = prepare_features(df, audit_config, extra_columns=extra)
        model = build_candidates(audit_config, numeric, categorical)["logistic"]
        proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]

        metrics = evaluate_predictions(y, proba)
        rows.append(
            {
                "feature_set": label,
                "n_features": X.shape[1],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "brier_score": metrics["brier_score"],
                "verdict": verdict,
            }
        )
        logger.info(f"Leakage audit: {label}: AUC {metrics['roc_auc']:.4f}")

    audit = pd.DataFrame(rows)
    standard_auc = audit.loc[audit["feature_set"] == "Standard: project default", "roc_auc"].iloc[0]
    audit["auc_vs_standard"] = audit["roc_auc"] - standard_auc
    return audit
