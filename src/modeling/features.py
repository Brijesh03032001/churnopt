"""
Feature preparation for customer-level churn modeling.

Critical design principle: the feature list is derived from an explicit
exclusion list in configs/project.yaml, never from "everything left over".
Columns recorded at or after the outcome (satisfaction survey, IBM churn score,
churn reason) cannot reach the model unless a caller asks for them by name,
which only the leakage audit does.
"""

from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data_ingestion import TARGET
from ..utils import setup_logging

logger = setup_logging(__name__)

ADDON_SERVICES = [
    "online_security",
    "online_backup",
    "device_protection_plan",
    "premium_tech_support",
    "streaming_tv",
    "streaming_movies",
    "streaming_music",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretable derived features.

    Args:
        df: Cleaned customer dataframe

    Returns:
        Copy of the dataframe with `num_addon_services` (count of add-on
        services the customer pays for)
    """
    df = df.copy()
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)
    return df


def excluded_columns(config: Dict[str, Any]) -> List[str]:
    """
    Columns that must never be model inputs for the configured feature set.

    Args:
        config: Configuration dictionary from project.yaml

    Returns:
        Sorted list of column names to exclude
    """
    data_config = config["data"]
    excluded = set(data_config["excluded_columns"])
    excluded |= {data_config["id_column"], TARGET}

    if data_config.get("feature_set", "standard") == "strict":
        excluded |= set(data_config["cumulative_columns"])

    return sorted(excluded)


def prepare_features(
    df: pd.DataFrame, config: Dict[str, Any], extra_columns: Iterable[str] = ()
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build the model input frame and split columns into numeric and categorical.

    Yes/No columns become 0/1 numeric flags. Everything else that is text is
    treated as categorical and one-hot encoded by the preprocessor.

    Args:
        df: Cleaned customer dataframe
        config: Configuration dictionary from project.yaml
        extra_columns: Columns to add back despite being excluded. Only the
            leakage audit should use this.

    Returns:
        Tuple of (X, numeric_columns, categorical_columns)
    """
    df = add_derived_features(df)

    excluded = set(excluded_columns(config)) - set(extra_columns)
    columns = [col for col in df.columns if col not in excluded]
    X = df[columns].copy()

    categorical = []
    for col in columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "string":
            if set(X[col].unique()) <= {"Yes", "No"}:
                X[col] = (X[col] == "Yes").astype(int)
            else:
                categorical.append(col)

    numeric = [col for col in columns if col not in categorical]
    return X, numeric, categorical


def build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    """
    Scale numeric columns and one-hot encode categorical columns.

    The preprocessor is part of every model pipeline, so it is fitted inside
    each cross-validation fold and never sees held-out rows.

    Args:
        numeric: Numeric column names
        categorical: Categorical column names

    Returns:
        Unfitted ColumnTransformer
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
