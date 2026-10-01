"""
Data ingestion and cleaning for the IBM Telco customer churn dataset.

This module handles:
1. Loading raw data without letting pandas turn the string "None" into NaN
2. Standardizing column names and building the churn target
3. Validating the cleaned data against explicit expectations
4. Stratified train/test splitting at the customer level
"""

import re
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import setup_logging

logger = setup_logging(__name__)

TARGET = "churn"


class DataValidationError(ValueError):
    """Raised when the cleaned data violates an expectation."""


def to_snake_case(name: str) -> str:
    """Convert a raw column name such as 'Avg Monthly GB Download' to snake_case."""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


class TelcoDataLoader:
    """
    Loader for the IBM Telco customer churn dataset.

    One row per customer. The dataset has no history, so the unit of analysis
    is the customer and the target means "churned during the observed quarter".
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader.

        Args:
            config: Configuration dictionary from project.yaml
        """
        self.config = config
        self.raw_data_path = config["paths"]["raw_data"]
        self.data_config = config["data"]
        self.logger = logger

    def load_raw(self) -> pd.DataFrame:
        """
        Load the raw telco CSV.

        "None" is a real category in Offer and Internet Type (no offer, no
        internet), so default NA parsing is disabled and only empty cells are
        treated as missing.

        Returns:
            Raw dataframe
        """
        self.logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path, keep_default_na=False, na_values=[""])
        self.logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize names, fix types, and build the churn target.

        Args:
            df: Raw dataframe

        Returns:
            Cleaned dataframe with snake_case columns and a binary `churn` column
        """
        df = df.copy()
        df.columns = [to_snake_case(col) for col in df.columns]

        text_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in text_cols:
            df[col] = df[col].str.strip()

        for col in ["total_charges", "monthly_charge", "tenure_in_months"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={"tenure_in_months": "tenure_months"})

        positive = self.data_config["positive_label"]
        df[TARGET] = (df[self.data_config["target_source"]] == positive).astype(int)

        df = df.drop_duplicates(subset=[self.data_config["id_column"]])
        return df.reset_index(drop=True)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check the cleaned data against explicit expectations.

        Args:
            df: Cleaned dataframe

        Returns:
            Dictionary of summary statistics

        Raises:
            DataValidationError: If any expectation fails
        """
        id_col = self.data_config["id_column"]
        problems = []

        if df[id_col].duplicated().any():
            problems.append("duplicate customer ids")

        required = [id_col, TARGET, "tenure_months", "monthly_charge", "contract"]
        for col in required:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                problems.append(f"{n_missing} missing values in {col}")

        if (df["tenure_months"] < 0).any():
            problems.append("negative tenure")
        if (df["monthly_charge"] <= 0).any():
            problems.append("non-positive monthly charge")

        # The label and the status column must agree: Churned <=> Yes
        status_mismatch = int(((df["customer_status"] == "Churned") != (df[TARGET] == 1)).sum())
        if status_mismatch:
            problems.append(
                f"{status_mismatch} rows where customer_status disagrees with the label"
            )

        # No-internet customers must have the 'None' category, not a missing value
        no_internet = df["internet_service"] == "No"
        if (df.loc[no_internet, "internet_type"] != "None").any():
            problems.append("internet_type is not 'None' for customers without internet")

        churn_rate = float(df[TARGET].mean())
        if not 0.01 < churn_rate < 0.60:
            problems.append(f"implausible churn rate {churn_rate:.3f}")

        if problems:
            raise DataValidationError("; ".join(problems))

        report = {
            "n_customers": int(len(df)),
            "n_churned": int(df[TARGET].sum()),
            "churn_rate": churn_rate,
            "n_columns": int(len(df.columns)),
        }
        self.logger.info(
            f"Validation passed: {report['n_customers']:,} customers, "
            f"{report['churn_rate']:.1%} churned"
        )
        return report

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stratified customer-level train/test split.

        Each customer is one row, so no customer appears on both sides.

        Args:
            df: Cleaned dataframe

        Returns:
            Tuple of (train, test) dataframes
        """
        train, test = train_test_split(
            df,
            test_size=self.config["split"]["test_size"],
            stratify=df[TARGET],
            random_state=self.config["random_seed"],
        )
        self.logger.info(f"Split: {len(train):,} train, {len(test):,} test")
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        """
        Load, clean, and validate the dataset.

        Returns:
            Cleaned and validated dataframe
        """
        df = self.clean(self.load_raw())
        self.validate(df)
        return df
