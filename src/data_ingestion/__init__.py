"""
Data ingestion module.
"""

from .loader import TARGET, DataValidationError, TelcoDataLoader, to_snake_case

__all__ = ["TelcoDataLoader", "DataValidationError", "TARGET", "to_snake_case"]
