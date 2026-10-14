"""
Experiment design module.
"""

from .design import (
    design_table,
    detectable_save_rate,
    required_sample_size,
    simulate_power,
    treated_churn_rate,
    validate_design,
)

__all__ = [
    "design_table",
    "detectable_save_rate",
    "required_sample_size",
    "simulate_power",
    "treated_churn_rate",
    "validate_design",
]
