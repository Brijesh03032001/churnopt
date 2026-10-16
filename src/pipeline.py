"""
Pipeline stages.

Each stage reads the outputs of the stages before it from outputs/, so any
stage can be rerun on its own:

    sql         SQL business analysis (DuckDB)
    train       model comparison, holdout evaluation, leakage audit, predictions
    decide      retention economics, policy backtest, sensitivity, action plan
    experiment  design of the randomized holdout test
    explain     odds ratios, permutation importance, SHAP
    report      figures and the results summary
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.base import clone

from .analytics import run_sql_analysis
from .data_ingestion import TARGET, TelcoDataLoader
from .decision_engine import (
    RetentionEconomics,
    baseline_rule_mask,
    bootstrap_policy_net,
    breakeven_save_rate,
    budget_contacts,
    build_action_plan,
    compare_policies,
    optimal_contacts,
    optimal_net_profit,
    profit_curve,
    score_customers,
    sensitivity_grid,
    tornado,
)
from .experiments import design_table, detectable_save_rate, required_sample_size, validate_design
from .explainability import (
    odds_ratio_table,
    permutation_importance_table,
    shap_importance_table,
    shap_values_by_feature,
)
from .modeling import (
    build_candidates,
    build_class_weighted_reference,
    calibration_table,
    prepare_features,
)
from .modeling.train import ChurnModelTrainer
from .reporting import figures
from .reporting.style import save
from .utils import ensure_dir, load_config, save_json, setup_logging

logger = setup_logging(__name__)

PROJECT_CONFIG = "configs/project.yaml"
ECONOMICS_CONFIG = "configs/economics.yaml"
EXPERIMENT_CONFIG = "configs/experiment.yaml"
STAGES = ["sql", "train", "decide", "experiment", "explain", "report"]


def _out(config: Dict[str, Any]) -> Path:
    return Path(ensure_dir(config["paths"]["outputs"]))


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def stage_sql(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the SQL analysis queries."""
    results = run_sql_analysis(config)
    return {"queries": list(results)}


def stage_train(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compare models, evaluate on the holdout, audit leakage, write predictions."""
    result = ChurnModelTrainer(config).run()
    return result["metrics"]


def stage_decide(config: Dict[str, Any], economics_config: Dict[str, Any]) -> Dict[str, Any]:
    """Turn out-of-fold probabilities into a campaign: value, backtest, sensitivity, plan."""
    out = _out(config)
    economics = RetentionEconomics.from_config(economics_config)
    oof = pd.read_csv(out / "predictions_oof.csv")
    scored = score_customers(oof, economics)

    curve = profit_curve(scored, economics)
    k_budget = budget_contacts(economics, economics_config["budget"])
    k_optimal = optimal_contacts(curve)
    k = min(k_budget, k_optimal)

    rule = economics_config["baseline_rule"]
    policies = compare_policies(scored, economics, k, rule, config["random_seed"]) if k else None
    breakeven = breakeven_save_rate(scored, economics)
    bootstrap = economics_config["bootstrap"]
    interval = bootstrap_policy_net(
        scored, economics, k, bootstrap["n_resamples"], bootstrap["ci_level"], config["random_seed"]
    )

    sens = economics_config["sensitivity"]
    grid = sensitivity_grid(
        scored, economics, sens["save_rate"], sens["cost_per_contact"], economics_config["budget"]
    )
    swings = tornado(oof, economics, sens, economics_config["budget"])
    plan = build_action_plan(scored, k)

    curve.to_csv(out / "profit_curve.csv", index=False)
    policies.to_csv(out / "policy_comparison.csv", index=False)
    breakeven.to_csv(out / "breakeven_save_rate.csv", index=False)
    grid.to_csv(out / "sensitivity_grid.csv", index=False)
    swings.to_csv(out / "sensitivity_tornado.csv", index=False)
    plan.to_csv(out / "action_plan.csv", index=False)

    at_k = curve.iloc[k - 1]
    at_optimum = curve.iloc[k_optimal - 1]
    unconstrained = optimal_net_profit(scored, economics)
    model_row = policies[policies["policy"] == "Model: rank by expected value"].iloc[0]
    rule_row = policies[policies["policy"].str.startswith("Rule:")].iloc[0]

    summary = {
        "assumptions": asdict(economics),
        "budget": economics_config["budget"],
        "contacts_budget_limit": int(k_budget),
        "contacts_profit_maximizing": int(k_optimal),
        "contacts_recommended": int(k),
        "customers_with_positive_ev": int((scored["expected_value"] > 0).sum()),
        "recommended": {
            "contacts": int(k),
            "cost": float(at_k["cost"]),
            "expected_net": float(at_k["expected_net"]),
            "realized_net": float(at_k["realized_net"]),
            "expected_roi": float(at_k["expected_net"] / at_k["cost"]),
            "realized_roi": float(at_k["realized_net"] / at_k["cost"]),
            "precision": float(model_row["precision"]),
            "share_of_churners_reached": float(model_row["share_of_churners_reached"]),
            "breakeven_save_rate": float(breakeven.loc[k - 1, "breakeven_save_rate"]),
            "bootstrap_realized_net": interval,
            "model_vs_rule_net_ratio": float(model_row["net"] / rule_row["net"]),
        },
        "profit_maximizing": {
            "contacts": int(k_optimal),
            "cost": float(at_optimum["cost"]),
            "expected_net": float(at_optimum["expected_net"]),
            "realized_net": float(at_optimum["realized_net"]),
            "expected_roi": float(at_optimum["expected_net"] / at_optimum["cost"]),
            "breakeven_save_rate": float(breakeven.loc[k_optimal - 1, "breakeven_save_rate"]),
        },
        "unconstrained_optimum": unconstrained,
        "baseline_rule_size": int(baseline_rule_mask(scored, rule).sum()),
        "policies": policies.round(3).to_dict(orient="records"),
    }
    save_json(summary, str(out / "decision_summary.json"))
    logger.info(
        f"Campaign: {k:,} contacts, expected net ${summary['recommended']['expected_net']:,.0f}, "
        f"realized ${summary['recommended']['realized_net']:,.0f}"
    )
    return summary


def stage_experiment(
    config: Dict[str, Any], economics_config: Dict[str, Any], experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Size the randomized holdout test that would measure the save rate."""
    out = _out(config)
    decision = _read_json(out / "decision_summary.json")
    economics = RetentionEconomics.from_config(economics_config)
    scored = score_customers(pd.read_csv(out / "predictions_oof.csv"), economics)

    contacts = decision["contacts_recommended"]
    baseline_rate = float(scored.head(contacts)[TARGET].mean())
    breakeven = decision["recommended"]["breakeven_save_rate"]

    test = experiment_config["test"]
    control_fraction = experiment_config["control_fraction"]
    ratio = control_fraction / (1 - control_fraction)

    design = validate_design(design_table(baseline_rate, experiment_config), experiment_config)
    design.to_csv(out / "experiment_design.csv", index=False)

    n_breakeven = required_sample_size(
        baseline_rate, breakeven, test["alpha"], test["power"], test["alternative"], ratio
    )
    n_treatment_budget = int(round(contacts * (1 - control_fraction)))
    summary = {
        "baseline_churn_rate_contacted": baseline_rate,
        "breakeven_save_rate": breakeven,
        "assumed_save_rate": economics.save_rate,
        "n_treatment_to_detect_breakeven": int(n_breakeven),
        "n_total_to_detect_breakeven": int(round(n_breakeven * (1 + ratio))),
        "contacts_in_budget": int(contacts),
        "n_treatment_if_budget_list_is_split": n_treatment_budget,
        "detectable_save_rate_with_budget_list": detectable_save_rate(
            n_treatment_budget,
            baseline_rate,
            test["alpha"],
            test["power"],
            test["alternative"],
            ratio,
        ),
        "simulated_power_min": float(design["simulated_power"].min()),
        "simulated_power_max": float(design["simulated_power"].max()),
    }
    save_json(summary, str(out / "experiment_summary.json"))
    return summary


def stage_explain(config: Dict[str, Any]) -> Dict[str, Any]:
    """Odds ratios, permutation importance, and SHAP values."""
    out = _out(config)
    loader = TelcoDataLoader(config)
    df = loader.run()
    train, test = loader.split(df)
    champion_name = _read_json(out / "metrics.json")["champion"]

    X_train, numeric, categorical = prepare_features(train, config)
    X_test, _, _ = prepare_features(test, config)
    model = clone(build_candidates(config, numeric, categorical)[champion_name]).fit(
        X_train, train[TARGET]
    )

    odds = odds_ratio_table(df)
    permutation = permutation_importance_table(
        model, X_test, test[TARGET], seed=config["random_seed"]
    )
    shap_frame = shap_values_by_feature(
        model,
        X_train.sample(200, random_state=config["random_seed"]),
        X_test.sample(min(400, len(X_test)), random_state=config["random_seed"]),
        seed=config["random_seed"],
    )
    shap_table = shap_importance_table(shap_frame)

    odds.to_csv(out / "odds_ratios.csv", index=False)
    permutation.to_csv(out / "permutation_importance.csv", index=False)
    shap_table.to_csv(out / "shap_importance.csv", index=False)
    return {
        "top_permutation": permutation["feature"].head(5).tolist(),
        "top_shap": shap_table["feature"].head(5).tolist(),
    }


def stage_report(
    config: Dict[str, Any], economics_config: Dict[str, Any], experiment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Build every figure and the one-file results summary."""
    out = _out(config)
    figure_dir = str(ensure_dir(config["paths"]["figures"]))
    metrics = _read_json(out / "metrics.json")
    decision = _read_json(out / "decision_summary.json")
    experiment = _read_json(out / "experiment_summary.json")
    economics = RetentionEconomics.from_config(economics_config)

    loader = TelcoDataLoader(config)
    df = loader.run()
    train, test = loader.split(df)
    oof = pd.read_csv(out / "predictions_oof.csv")
    holdout = pd.read_csv(out / "predictions_holdout.csv")
    audit = pd.read_csv(out / "leakage_audit.csv")
    comparison = pd.read_csv(out / "model_comparison.csv")
    pareto = pd.read_csv(out / "sql" / "03_revenue_at_risk_pareto.csv")
    overview = pd.read_csv(out / "sql" / "01_churn_overview.csv").iloc[0].to_dict()

    # Reference: what class weighting does to probabilities
    X_train, numeric, categorical = prepare_features(train, config)
    X_test, _, _ = prepare_features(test, config)
    reference = build_class_weighted_reference(config, numeric, categorical).fit(
        X_train, train[TARGET]
    )
    reference_proba = reference.predict_proba(X_test)[:, 1]
    n_bins = config["evaluation"]["calibration_bins"]
    reference_table = calibration_table(test[TARGET], reference_proba, n_bins)
    champion_table = calibration_table(holdout[TARGET], holdout["churn_probability"], n_bins)
    overestimate = float(reference_proba.mean() / test[TARGET].mean())

    rule = economics_config["baseline_rule"]
    rule_label = f"Rule: {rule['contract']} + {rule['internet_type']}"
    curve = pd.read_csv(out / "profit_curve.csv")
    policies = pd.read_csv(out / "policy_comparison.csv")
    grid = pd.read_csv(out / "sensitivity_grid.csv")
    swings = pd.read_csv(out / "sensitivity_tornado.csv")
    design = pd.read_csv(out / "experiment_design.csv")
    odds = pd.read_csv(out / "odds_ratios.csv")
    shap_table = pd.read_csv(out / "shap_importance.csv")
    permutation = pd.read_csv(out / "permutation_importance.csv")

    saved = {
        "01_churn_by_contract_tenure": save(
            figures.churn_by_contract_and_tenure(df), figure_dir, "01_churn_by_contract_tenure"
        ),
        "02_revenue_pareto": save(figures.revenue_pareto(pareto), figure_dir, "02_revenue_pareto"),
        "03_leakage_audit": save(figures.leakage_audit(audit), figure_dir, "03_leakage_audit"),
        "04_model_comparison": save(
            figures.model_comparison(comparison, metrics["champion"]),
            figure_dir,
            "04_model_comparison",
        ),
        "05_calibration": save(
            figures.calibration_curve(
                champion_table, reference_table, metrics["holdout"]["ece"], overestimate
            ),
            figure_dir,
            "05_calibration",
        ),
        "06_gain_curve": save(
            figures.gain_curve(oof, baseline_rule_mask(oof, rule), rule_label),
            figure_dir,
            "06_gain_curve",
        ),
        "07_profit_curve": save(
            figures.profit_curve_chart(
                curve, decision["contacts_budget_limit"], decision["contacts_profit_maximizing"]
            ),
            figure_dir,
            "07_profit_curve",
        ),
        "08_policy_comparison": save(
            figures.policy_comparison(policies, decision["contacts_recommended"]),
            figure_dir,
            "08_policy_comparison",
        ),
        "09_sensitivity_heatmap": save(
            figures.sensitivity_heatmap(grid, economics.save_rate, economics.cost_per_contact),
            figure_dir,
            "09_sensitivity_heatmap",
        ),
        "10_tornado": save(figures.tornado_chart(swings), figure_dir, "10_tornado"),
        "11_odds_ratios": save(figures.odds_ratio_forest(odds), figure_dir, "11_odds_ratios"),
        "12_shap_importance": save(
            figures.importance_bars(
                shap_table,
                "mean_abs_shap",
                "Contract and referrals lead, then tenure and household",
                "Mean absolute SHAP value, champion model, 400 held-out customers (change in churn probability)",
                "Mean |SHAP| (probability points)",
            ),
            figure_dir,
            "12_shap_importance",
        ),
        "13_permutation_importance": save(
            figures.importance_bars(
                permutation,
                "auc_drop",
                "Shuffling contract type hurts the model most",
                "Drop in held-out ROC-AUC when one feature is shuffled (15 repeats)",
                "ROC-AUC drop",
            ),
            figure_dir,
            "13_permutation_importance",
        ),
        "14_experiment_power": save(
            figures.power_curve(
                design,
                experiment["baseline_churn_rate_contacted"],
                experiment["breakeven_save_rate"],
                economics.save_rate,
                experiment["n_treatment_to_detect_breakeven"],
                experiment["n_treatment_if_budget_list_is_split"],
                experiment["detectable_save_rate_with_budget_list"],
            ),
            figure_dir,
            "14_experiment_power",
        ),
    }

    strict = audit[audit["feature_set"].str.startswith("Strict")].iloc[0]
    standard = audit[audit["feature_set"].str.startswith("Standard: project")].iloc[0]
    sat = audit[audit["feature_set"] == "Standard + satisfaction_score"].iloc[0]
    both = audit[audit["feature_set"].str.contains("satisfaction_score \\+ churn_score")].iloc[0]

    summary = {
        "data": {
            "customers": int(overview["customers"]),
            "churn_rate": float(overview["churn_rate_pct"]) / 100,
            "monthly_revenue_lost_pct": float(overview["revenue_lost_pct"]),
            "monthly_recurring_revenue": float(overview["monthly_recurring_revenue"]),
            "monthly_revenue_lost": float(overview["monthly_revenue_lost"]),
        },
        "model": {
            "champion": metrics["champion"],
            "holdout": metrics["holdout"],
            "holdout_ci": metrics["holdout_ci"],
            "cv_champion": metrics["cv"]["champion"],
            "baselines": metrics["holdout_baselines"],
            "holdout_strict_features": metrics["holdout_strict_features"],
            "strict_feature_set_auc": float(strict["roc_auc"]),
            "standard_feature_set_auc": float(standard["roc_auc"]),
            "with_satisfaction_auc": float(sat["roc_auc"]),
            "with_satisfaction_and_churn_score_auc": float(both["roc_auc"]),
            "class_weighted_overestimate_factor": overestimate,
        },
        "decision": decision,
        "experiment": experiment,
        "figures": saved,
    }
    save_json(summary, str(out / "results_summary.json"))
    return summary


def run_stage(name: str) -> Dict[str, Any]:
    """Run one pipeline stage by name."""
    config = load_config(PROJECT_CONFIG)
    economics_config = load_config(ECONOMICS_CONFIG)
    experiment_config = load_config(EXPERIMENT_CONFIG)

    logger.info(f"=== STAGE: {name} ===")
    if name == "sql":
        return stage_sql(config)
    if name == "train":
        return stage_train(config)
    if name == "decide":
        return stage_decide(config, economics_config)
    if name == "experiment":
        return stage_experiment(config, economics_config, experiment_config)
    if name == "explain":
        return stage_explain(config)
    if name == "report":
        return stage_report(config, economics_config, experiment_config)
    raise ValueError(f"Unknown stage {name!r}; choose from {STAGES}")


def run_all() -> None:
    """Run every stage in order."""
    for name in STAGES:
        run_stage(name)
