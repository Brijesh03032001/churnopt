"""
Explaining churn: what drives it, and what the model relies on.

Three complementary views:
1. Odds ratios from a logistic regression with confidence intervals: the
   analyst's view, "holding everything else fixed, how does X change the odds?"
2. Permutation importance on the held-out test set: how much does the champion
   model's AUC drop when a feature is shuffled?
3. SHAP values for the champion, summed across one-hot columns so each original
   feature gets one number per customer.

All three describe association in observational data. None of them says that
changing a feature would change churn; only a randomized test can.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.inspection import permutation_importance

from ..data_ingestion import TARGET
from ..modeling.features import add_derived_features
from ..utils import setup_logging

logger = setup_logging(__name__)

# (column, reference level) for categorical terms, in display order
CATEGORICAL_TERMS = [
    ("contract", "Month-to-Month"),
    ("internet_type", "None"),
    ("payment_method", "Bank Withdrawal"),
]
BINARY_TERMS = [
    "premium_tech_support",
    "online_security",
    "online_backup",
    "device_protection_plan",
    "streaming_tv",
    "paperless_billing",
    "senior_citizen",
    "dependents",
]
# (column, divisor, label): continuous terms rescaled so one unit is meaningful
CONTINUOUS_TERMS = [
    ("tenure_months", 12.0, "Tenure (per +12 months)"),
    ("monthly_charge", 10.0, "Monthly charge (per +$10)"),
]


def odds_ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit an interpretable logistic regression and report odds ratios.

    Left out on purpose: cumulative billing totals (near-linear combinations of
    tenure and monthly charge, so coefficients would be unstable); offer (each
    offer goes almost entirely to one tenure band, so its coefficient mixes up
    the offer with the customer's tenure); referral count (non-monotonic and
    tied to tenure); gender (not a lever).

    Args:
        df: Cleaned customer dataframe

    Returns:
        DataFrame with term, odds ratio, 95% CI, and p-value, sorted by effect size
    """
    data = add_derived_features(df)
    columns = {}
    labels = {}

    for col, divisor, label in CONTINUOUS_TERMS:
        columns[col] = data[col] / divisor
        labels[col] = label

    for col in BINARY_TERMS:
        binary = data[col]
        columns[col] = (
            (binary == "Yes").astype(float) if binary.dtype == object else binary.astype(float)
        )
        labels[col] = col.replace("_", " ").capitalize()

    for col, reference in CATEGORICAL_TERMS:
        for level in sorted(data[col].unique()):
            if level == reference:
                continue
            name = f"{col}={level}"
            columns[name] = (data[col] == level).astype(float)
            labels[name] = f"{col.replace('_', ' ').capitalize()}: {level} (vs {reference})"

    X = sm.add_constant(pd.DataFrame(columns))
    fit = sm.Logit(data[TARGET], X).fit(disp=0)

    ci = fit.conf_int()
    table = pd.DataFrame(
        {
            "term": [labels[name] for name in fit.params.index if name != "const"],
            "odds_ratio": np.exp(fit.params.drop("const")).to_numpy(),
            "ci_low": np.exp(ci.drop("const")[0]).to_numpy(),
            "ci_high": np.exp(ci.drop("const")[1]).to_numpy(),
            "p_value": fit.pvalues.drop("const").to_numpy(),
        }
    )
    table["significant"] = table["p_value"] < 0.05
    table["log_distance"] = np.abs(np.log(table["odds_ratio"]))
    return (
        table.sort_values("log_distance", ascending=False)
        .drop(columns="log_distance")
        .reset_index(drop=True)
    )


def permutation_importance_table(
    model, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 15, seed: int = 42
) -> pd.DataFrame:
    """
    Drop in test ROC-AUC when each raw feature is shuffled.

    Features are shuffled before preprocessing, so a one-hot encoded feature is
    permuted as a whole.

    Args:
        model: Fitted pipeline that accepts the raw feature frame
        X_test: Held-out features
        y_test: Held-out outcomes
        n_repeats: Shuffles per feature
        seed: Random seed

    Returns:
        DataFrame with mean and std AUC drop per feature, largest first
    """
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats, random_state=seed
    )
    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "auc_drop": result.importances_mean,
            "auc_drop_std": result.importances_std,
        }
    )
    return table.sort_values("auc_drop", ascending=False).reset_index(drop=True)


def _fitted_pipeline(model):
    """Return a fitted preprocessing + estimator pipeline, unwrapping calibration."""
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def shap_values_by_feature(
    model, X_background: pd.DataFrame, X_explain: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """
    SHAP values for the champion, aggregated back to the original features.

    SHAP values are additive, so summing the values of a feature's one-hot
    columns gives that feature's total contribution.

    Args:
        model: Fitted champion (pipeline, or calibrated wrapper of one)
        X_background: Reference sample for the masker
        X_explain: Customers to explain
        seed: Random seed

    Returns:
        DataFrame with one column per original feature, one row per explained
        customer, in probability units (change from the average prediction)
    """
    import shap

    pipeline = _fitted_pipeline(model)
    preprocessor, estimator = pipeline.named_steps["pre"], pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    background = preprocessor.transform(X_background)
    explained = preprocessor.transform(X_explain)

    explainer = shap.explainers.Permutation(
        lambda A: estimator.predict_proba(A)[:, 1],
        shap.maskers.Independent(background, max_samples=len(background)),
        seed=seed,
    )
    values = explainer(explained, max_evals=2 * len(names) + 1, silent=True).values

    def source_feature(name: str) -> str:
        stem = name.split("__", 1)[1]
        for col in X_explain.columns:
            if stem == col or stem.startswith(f"{col}_"):
                return col
        return stem

    sources = np.array([source_feature(n) for n in names])
    aggregated = {
        feature: values[:, sources == feature].sum(axis=1) for feature in X_explain.columns
    }
    return pd.DataFrame(aggregated, index=X_explain.index)


def shap_importance_table(shap_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, largest first."""
    table = shap_frame.abs().mean().rename("mean_abs_shap").reset_index()
    table.columns = ["feature", "mean_abs_shap"]
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
