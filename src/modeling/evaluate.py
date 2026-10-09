"""
Model evaluation: discrimination, calibration, ranking value, and uncertainty.

Discrimination (AUC) says whether the model ranks churners above stayers.
Calibration says whether "30% risk" really means 30 in 100 customers churn.
Ranking metrics (gain, lift) say how much a capacity-limited campaign gains.
The decision engine needs all three, so all three are measured.
"""

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from ..utils import setup_logging

logger = setup_logging(__name__)

PROB_CLIP = 1e-6


def calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """
    Reliability table with equal-frequency bins.

    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        DataFrame with mean predicted probability, observed churn rate, and
        customer count per bin
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), n_bins, labels=False)
    table = (
        frame.groupby("bin")
        .agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), customers=("y", "size"))
        .reset_index(drop=True)
    )
    return table


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected calibration error: customer-weighted gap between predicted and observed rates.

    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of equal-frequency bins

    Returns:
        ECE in probability units (0.05 means predictions are off by 5 points on average)
    """
    table = calibration_table(y_true, y_prob, n_bins)
    gap = (table["mean_predicted"] - table["observed_rate"]).abs()
    return float((gap * table["customers"]).sum() / table["customers"].sum())


def evaluate_predictions(y_true, y_prob, n_bins: int = 10) -> Dict[str, float]:
    """
    Compute the standard metric set for a vector of predicted probabilities.

    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        n_bins: Bins for the calibration error

    Returns:
        Dictionary with roc_auc, pr_auc, brier_score, brier_skill, log_loss, ece
    """
    y = np.asarray(y_true)
    p = np.clip(np.asarray(y_prob, dtype=float), PROB_CLIP, 1 - PROB_CLIP)

    brier = brier_score_loss(y, p)
    brier_reference = brier_score_loss(y, np.full(len(y), y.mean()))

    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier_score": brier,
        "brier_skill": 1 - brier / brier_reference,
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
    }


def bootstrap_ci(
    y_true,
    y_prob,
    metric: Callable,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap confidence interval, resampling customers.

    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        metric: Function of (y_true, y_prob) returning a float
        n_resamples: Number of bootstrap resamples
        ci_level: Confidence level
        seed: Random seed

    Returns:
        Tuple of (lower, upper) bounds
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    rng = np.random.default_rng(seed)

    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], p[idx]))

    alpha = (1 - ci_level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def gain_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """
    Gain and lift by risk decile (decile 1 = highest predicted risk).

    Args:
        y_true: Binary outcomes
        y_prob: Predicted probabilities
        n_bins: Number of equal-size groups

    Returns:
        DataFrame with churn rate, lift over the base rate, and the cumulative
        share of all churners captured by each decile
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame = frame.sort_values("p", ascending=False).reset_index(drop=True)
    frame["decile"] = np.floor(np.arange(len(frame)) * n_bins / len(frame)).astype(int) + 1

    table = (
        frame.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum")).reset_index()
    )
    base_rate = frame["y"].mean()
    table["churn_rate"] = table["churners"] / table["customers"]
    table["lift"] = table["churn_rate"] / base_rate
    table["cumulative_churners_captured"] = table["churners"].cumsum() / frame["y"].sum()
    table["cumulative_customers"] = table["customers"].cumsum() / len(frame)
    return table


def cross_validate_models(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Repeated stratified k-fold comparison of candidate models.

    Every metric is computed on rows the fold's model never saw. Preprocessing
    lives inside each model pipeline, so it is refitted per fold too.

    Args:
        models: Dictionary mapping name to unfitted estimator
        X: Feature frame
        y: Binary target
        n_splits: Folds per repeat
        n_repeats: Number of repeats with different shuffles
        seed: Random seed
        n_bins: Bins for the calibration error

    Returns:
        Tuple of (summary, folds). Summary has mean, std, and standard error
        per model and metric; folds has one row per model and fold.
    """
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(splitter.split(X, y))

    rows = []
    for name, estimator in models.items():
        logger.info(f"Cross-validating {name} ({len(splits)} folds)")
        for fold, (train_idx, test_idx) in enumerate(splits):
            model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            metrics = evaluate_predictions(y.iloc[test_idx], proba, n_bins)
            rows.append({"model": name, "fold": fold, **metrics})

    folds = pd.DataFrame(rows)
    metric_cols = [col for col in folds.columns if col not in ("model", "fold")]

    grouped = folds.groupby("model", sort=False)[metric_cols]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    # Folds overlap in their training rows, so they are not independent draws.
    # Dividing by sqrt(n_splits) instead of sqrt(n_splits * n_repeats) keeps the
    # standard error wider than the naive one (conservative).
    for col in metric_cols:
        summary[f"{col}_se"] = summary[f"{col}_std"] / np.sqrt(n_splits)

    return summary.reset_index(), folds


def select_champion(summary: pd.DataFrame, selection_config: Dict) -> Tuple[str, str]:
    """
    Pick the champion model from cross-validated results.

    The best model by the selection metric wins, unless the one-standard-error
    rule is on and a simpler model scores within one standard error of it.

    Args:
        summary: Output of cross_validate_models
        selection_config: `selection` section of project.yaml

    Returns:
        Tuple of (champion_name, rationale)
    """
    metric = selection_config["metric"]
    order = selection_config["complexity_order"]
    lower_is_better = metric in ("log_loss", "brier_score", "ece")

    mean = summary.set_index("model")[f"{metric}_mean"]
    se = summary.set_index("model")[f"{metric}_se"]
    best = mean.idxmin() if lower_is_better else mean.idxmax()

    if not selection_config.get("one_se_rule", True):
        return best, f"best cross-validated {metric}"

    if lower_is_better:
        within = [m for m in mean.index if mean[m] <= mean[best] + se[best]]
    else:
        within = [m for m in mean.index if mean[m] >= mean[best] - se[best]]

    champion = min(within, key=lambda m: order.index(m))
    if champion == best:
        return champion, f"best cross-validated {metric}"
    return (
        champion,
        f"simplest model within 1 SE of the best ({best}) on {metric}",
    )
