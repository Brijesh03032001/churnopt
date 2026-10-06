"""
Modeling module: features, candidate models, evaluation, training.
"""

from .evaluate import (
    bootstrap_ci,
    calibration_table,
    cross_validate_models,
    evaluate_predictions,
    expected_calibration_error,
    gain_table,
    select_champion,
)
from .features import add_derived_features, build_preprocessor, excluded_columns, prepare_features
from .models import MODEL_DESCRIPTIONS, build_candidates, build_class_weighted_reference

__all__ = [
    "add_derived_features",
    "build_candidates",
    "build_class_weighted_reference",
    "build_preprocessor",
    "bootstrap_ci",
    "calibration_table",
    "cross_validate_models",
    "evaluate_predictions",
    "excluded_columns",
    "expected_calibration_error",
    "gain_table",
    "prepare_features",
    "select_champion",
    "MODEL_DESCRIPTIONS",
]
