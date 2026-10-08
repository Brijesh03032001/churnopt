"""
Training pipeline: model comparison, champion selection, honest evaluation.

Evaluation protocol:
1. Customer-level stratified split. The test set is not touched until the end.
2. Repeated stratified cross-validation on the training set compares all
   candidates and selects the champion.
3. The champion is fitted on the training set and scored once on the test set,
   with bootstrap confidence intervals.
4. Out-of-fold predictions for ALL customers feed the decision analysis, so no
   customer is ever scored by a model that saw their label.
5. A final model is fitted on all customers for scoring new data.
"""

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..data_ingestion import TARGET, TelcoDataLoader
from ..utils import ensure_dir, save_json, setup_logging
from .evaluate import (
    bootstrap_ci,
    cross_validate_models,
    evaluate_predictions,
    select_champion,
)
from .features import prepare_features
from .leakage_audit import run_leakage_audit
from .models import MODEL_DESCRIPTIONS, build_candidates

logger = setup_logging(__name__)

# Columns carried alongside predictions so the decision analysis and the
# dashboard need nothing but the predictions file.
CONTEXT_COLUMNS = [
    "customer_id",
    "contract",
    "internet_type",
    "payment_method",
    "offer",
    "tenure_months",
    "monthly_charge",
    "num_addon_services",
    "premium_tech_support",
    "online_security",
    "senior_citizen",
    "gender",
    "dependents",
]


class ChurnModelTrainer:
    """
    Train and evaluate churn models with a leakage-safe protocol.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary from project.yaml
        """
        self.config = config
        self.seed = config["random_seed"]
        self.output_dir = ensure_dir(config["paths"]["outputs"])
        self.model_dir = ensure_dir(config["paths"]["models"])
        self.logger = logger

    def _context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Customer attributes used downstream, with derived features added."""
        from .features import add_derived_features

        context = add_derived_features(df)[CONTEXT_COLUMNS].reset_index(drop=True)
        for column in ("premium_tech_support", "online_security"):
            context[column] = (context[column] == "Yes").astype(int)
        return context

    def run(self) -> Dict[str, Any]:
        """
        Execute the full training and evaluation protocol.

        Returns:
            Dictionary with the champion name, fitted model, and all metrics
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING PIPELINE")
        self.logger.info("=" * 60)

        loader = TelcoDataLoader(self.config)
        df = loader.run()
        train, test = loader.split(df)

        X_train, numeric, categorical = prepare_features(train, self.config)
        X_test, _, _ = prepare_features(test, self.config)
        y_train, y_test = train[TARGET], test[TARGET]

        # 1. Compare candidates on the training set only
        candidates = build_candidates(self.config, numeric, categorical)
        cv_cfg = self.config["cv"]
        n_bins = self.config["evaluation"]["calibration_bins"]
        summary, folds = cross_validate_models(
            candidates, X_train, y_train, cv_cfg["n_splits"], cv_cfg["n_repeats"], self.seed, n_bins
        )
        summary.insert(1, "description", summary["model"].map(MODEL_DESCRIPTIONS))
        champion_name, rationale = select_champion(summary, self.config["selection"])
        self.logger.info(f"Champion: {champion_name} ({rationale})")

        # 2. Fit on train, score the untouched test set once
        champion = clone(candidates[champion_name]).fit(X_train, y_train)
        test_proba = champion.predict_proba(X_test)[:, 1]
        holdout = evaluate_predictions(y_test, test_proba, n_bins)

        eval_cfg = self.config["evaluation"]
        metric_fns = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}
        holdout_ci = {
            name: bootstrap_ci(
                y_test,
                test_proba,
                fn,
                eval_cfg["bootstrap_resamples"],
                eval_cfg["ci_level"],
                self.seed,
            )
            for name, fn in metric_fns.items()
        }

        # Baselines scored on the same test set, for context
        baselines = {}
        for name in ("prior", "contract_only"):
            baseline = clone(candidates[name]).fit(X_train, y_train)
            baselines[name] = evaluate_predictions(
                y_test, baseline.predict_proba(X_test)[:, 1], n_bins
            )

        # Conservative check: the same champion without lifetime-cumulative columns
        strict_config = copy.deepcopy(self.config)
        strict_config["data"]["feature_set"] = "strict"
        X_train_strict, strict_numeric, strict_categorical = prepare_features(train, strict_config)
        X_test_strict, _, _ = prepare_features(test, strict_config)
        strict_model = clone(
            build_candidates(strict_config, strict_numeric, strict_categorical)[champion_name]
        ).fit(X_train_strict, y_train)
        holdout_strict = evaluate_predictions(
            y_test, strict_model.predict_proba(X_test_strict)[:, 1], n_bins
        )

        # 3. Leakage audit on the training set
        audit = run_leakage_audit(train, self.config)

        # 4. Out-of-fold predictions for every customer
        X_all, _, _ = prepare_features(df, self.config)
        y_all = df[TARGET]
        oof_proba = cross_val_predict(
            clone(candidates[champion_name]),
            X_all,
            y_all,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed),
            method="predict_proba",
        )[:, 1]
        oof_metrics = evaluate_predictions(y_all, oof_proba, n_bins)

        # 5. Final model on all customers for scoring new data
        final_model = clone(candidates[champion_name]).fit(X_all, y_all)

        # Persist
        oof = self._context(df)
        oof.insert(1, TARGET, y_all.to_numpy())
        oof["churn_probability"] = oof_proba
        holdout_predictions = self._context(test)[["customer_id"]]
        holdout_predictions[TARGET] = y_test.to_numpy()
        holdout_predictions["churn_probability"] = test_proba

        summary.to_csv(self.output_dir / "model_comparison.csv", index=False)
        folds.to_csv(self.output_dir / "cv_folds.csv", index=False)
        audit.to_csv(self.output_dir / "leakage_audit.csv", index=False)
        oof.to_csv(self.output_dir / "predictions_oof.csv", index=False)
        holdout_predictions.to_csv(self.output_dir / "predictions_holdout.csv", index=False)

        champion_row = summary.set_index("model").loc[champion_name]
        metrics = {
            "champion": champion_name,
            "selection_rationale": rationale,
            "n_customers": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "churn_rate": float(y_all.mean()),
            "feature_set": self.config["data"]["feature_set"],
            "n_features": int(X_train.shape[1]),
            "cv": {
                "n_splits": cv_cfg["n_splits"],
                "n_repeats": cv_cfg["n_repeats"],
                "champion": {
                    col.removesuffix("_mean"): float(champion_row[col])
                    for col in champion_row.index
                    if col.endswith("_mean")
                },
            },
            "holdout": holdout,
            "holdout_ci": {k: list(v) for k, v in holdout_ci.items()},
            "holdout_baselines": baselines,
            "holdout_strict_features": holdout_strict,
            "out_of_fold_all_customers": oof_metrics,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(metrics, str(self.output_dir / "metrics.json"))

        model_path = Path(self.model_dir) / "churn_model.joblib"
        joblib.dump(
            {
                "model": final_model,
                "champion": champion_name,
                "feature_columns": list(X_all.columns),
                "numeric": numeric,
                "categorical": categorical,
                "config_feature_set": self.config["data"]["feature_set"],
            },
            model_path,
        )
        self.logger.info(f"Saved model to {model_path}")

        self.logger.info(
            f"Holdout ROC-AUC {holdout['roc_auc']:.4f} "
            f"[{holdout_ci['roc_auc'][0]:.4f}, {holdout_ci['roc_auc'][1]:.4f}], "
            f"Brier {holdout['brier_score']:.4f}, ECE {holdout['ece']:.4f}"
        )
        self.logger.info("=" * 60)

        return {
            "champion": champion_name,
            "model": final_model,
            "metrics": metrics,
            "comparison": summary,
            "leakage_audit": audit,
            "holdout_model": champion,
            "train": train,
            "test": test,
        }
