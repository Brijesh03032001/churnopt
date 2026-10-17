"""
Retention targeting simulator.

Run from the project root after the pipeline has produced outputs/:

    streamlit run app/dashboard.py

Every number is recomputed live from the out-of-fold churn probabilities, so
moving a slider re-ranks customers, re-sizes the campaign, and re-runs the
backtest. The economic inputs are assumptions; the dashboard says so.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.decision_engine import (  # noqa: E402
    RetentionEconomics,
    breakeven_save_rate,
    build_action_plan,
    compare_policies,
    optimal_contacts,
    profit_curve,
    score_customers,
)
from src.utils import load_config  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="ChurnOpt", page_icon=None, layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "predictions_oof.csv")


@st.cache_data
def load_summary() -> dict:
    return json.loads((OUTPUTS / "results_summary.json").read_text())


@st.cache_data
def load_economics_defaults() -> dict:
    return load_config(str(ROOT / "configs" / "economics.yaml"))


@st.cache_data
def simulate(assumptions: tuple, budget: float, cap_contacts: bool) -> dict:
    """Score customers and size the campaign for one set of assumptions."""
    cost, save_rate, margin, horizon = assumptions
    defaults = load_economics_defaults()
    economics = RetentionEconomics(
        cost_per_contact=cost,
        save_rate=save_rate,
        gross_margin=margin,
        horizon_months=horizon,
        monthly_discount_rate=defaults["value"]["monthly_discount_rate"],
    )
    scored = score_customers(load_predictions(), economics)
    curve = profit_curve(scored, economics)
    k_optimal = optimal_contacts(curve)
    k_budget = int(budget // cost) if cap_contacts else len(scored)
    k = min(k_optimal, k_budget)

    result = {
        "economics": economics,
        "scored": scored,
        "curve": curve,
        "k": k,
        "k_optimal": k_optimal,
        "k_budget": k_budget,
    }
    if k:
        rule = defaults["baseline_rule"]
        result["policies"] = compare_policies(scored, economics, k, rule)
        result["breakeven"] = breakeven_save_rate(scored, economics)
        result["plan"] = build_action_plan(scored, k)
    return result


def money(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def main() -> None:
    if not (OUTPUTS / "predictions_oof.csv").exists():
        st.error("No predictions found. Run `python run_pipeline.py` first.")
        return

    defaults = load_economics_defaults()
    summary = load_summary()

    st.title("ChurnOpt: retention targeting simulator")
    st.caption(
        "IBM Telco sample data (7,043 customers). Churn probabilities are out-of-fold, so no "
        "customer is scored by a model that saw their outcome. Dollar inputs are assumptions."
    )

    with st.sidebar:
        st.header("Campaign assumptions")
        save_rate = st.slider(
            "Save rate",
            0.05,
            0.60,
            float(defaults["offer"]["save_rate"]),
            0.01,
            help="Share of would-be churners who stay after being contacted. "
            "Unknown until a randomized test measures it.",
            format="%.2f",
        )
        cost = st.slider(
            "Cost per contact ($)", 5, 100, int(defaults["offer"]["cost_per_contact"]), 1
        )
        margin = st.slider(
            "Gross margin", 0.20, 0.70, float(defaults["value"]["gross_margin"]), 0.01
        )
        horizon = st.slider(
            "Months of margin kept per saved customer",
            3,
            36,
            int(defaults["value"]["horizon_months"]),
        )
        st.divider()
        cap_contacts = st.checkbox("Limit spend to a budget", value=True)
        budget = st.number_input(
            "Campaign budget ($)",
            1000,
            200000,
            int(defaults["budget"]["max_campaign_budget"]),
            1000,
            disabled=not cap_contacts,
        )

    result = simulate((float(cost), save_rate, margin, int(horizon)), float(budget), cap_contacts)
    curve, k = result["curve"], result["k"]

    tab_campaign, tab_segments, tab_list, tab_model = st.tabs(
        ["Campaign", "Segments", "Contact list", "Model and data"]
    )

    with tab_campaign:
        if k == 0:
            st.warning(
                "No profitable campaign under these assumptions: even the highest-value, "
                "highest-risk customer does not pay back the contact cost."
            )
        else:
            at_k = curve.iloc[k - 1]
            breakeven = result["breakeven"].loc[k - 1, "breakeven_save_rate"]
            reached = result["policies"].set_index("policy").iloc[0]["share_of_churners_reached"]

            columns = st.columns(5)
            columns[0].metric("Customers to contact", f"{k:,}")
            columns[1].metric("Campaign cost", money(at_k["cost"]))
            columns[2].metric("Expected net profit", money(at_k["expected_net"]))
            columns[3].metric(
                "ROI (net / cost)",
                f"{at_k['expected_net'] / at_k['cost']:.0%}",
                help="Net profit divided by cost, not value divided by cost.",
            )
            columns[4].metric(
                "Break-even save rate",
                f"{breakeven:.1%}",
                delta=f"{(save_rate - breakeven) * 100:+.1f} pts vs assumed",
                help="Lowest save rate at which this list still pays for itself.",
            )
            st.caption(
                f"Contact list reaches {reached:.0%} of all churners. "
                f"Budget allows {result['k_budget']:,} contacts; "
                f"the profit-maximizing list has {result['k_optimal']:,}."
            )

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Cumulative profit as the list grows")
            chart = curve[["contacts", "expected_net", "realized_net"]].rename(
                columns={
                    "expected_net": "Expected (calibrated probabilities)",
                    "realized_net": "Realized (actual churn outcomes)",
                }
            )
            st.line_chart(chart, x="contacts", color=[BLUE, ORANGE], height=360)
        with right:
            st.subheader("Same number of contacts, four policies")
            if k:
                policies = result["policies"][["policy", "net", "precision"]].copy()
                policies["precision"] = (policies["precision"] * 100).round(0).astype(int).astype(
                    str
                ) + "%"
                policies["net"] = policies["net"].map(money)
                st.dataframe(
                    policies.rename(
                        columns={
                            "policy": "Policy",
                            "net": "Realized net profit",
                            "precision": "Contacts who churn",
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption("Backtest on actual outcomes, using the assumed save rate.")

    with tab_segments:
        scored = result["scored"]
        scored = scored.assign(
            tenure_band=pd.cut(
                scored["tenure_months"],
                [0, 12, 24, 48, 1000],
                labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
            )
        )
        dimension = st.selectbox(
            "Group customers by",
            ["contract", "internet_type", "payment_method", "tenure_band", "offer"],
        )
        table = (
            scored.groupby(dimension, observed=True)
            .agg(
                customers=("churn", "size"),
                churn_rate=("churn", "mean"),
                avg_predicted_risk=("churn_probability", "mean"),
                avg_monthly_charge=("monthly_charge", "mean"),
                profitable_to_contact=("expected_value", lambda s: (s > 0).mean()),
            )
            .reset_index()
            .sort_values("churn_rate", ascending=False)
        )
        table["churn_rate"] = (table["churn_rate"] * 100).round(1)
        table["avg_predicted_risk"] = (table["avg_predicted_risk"] * 100).round(1)
        table["profitable_to_contact"] = (table["profitable_to_contact"] * 100).round(0)
        table["avg_monthly_charge"] = table["avg_monthly_charge"].round(2)
        st.dataframe(
            table.rename(
                columns={
                    "churn_rate": "Churn rate (%)",
                    "avg_predicted_risk": "Avg predicted risk (%)",
                    "avg_monthly_charge": "Avg monthly charge ($)",
                    "profitable_to_contact": "Positive EV (% of customers)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        if dimension == "offer":
            st.info(
                "Do not read this table as 'which offer works'. Each offer goes almost entirely "
                "to one tenure band, so offers differ mainly in customer age. See "
                "sql/06_offer_confounding.sql for the like-for-like comparison."
            )

    with tab_list:
        if k == 0:
            st.info("No customers to contact under these assumptions.")
        else:
            plan = result["plan"]
            contracts = st.multiselect(
                "Filter by contract", sorted(plan["contract"].unique()), default=[]
            )
            shown = plan[plan["contract"].isin(contracts)] if contracts else plan
            st.dataframe(shown, hide_index=True, use_container_width=True, height=380)
            st.download_button(
                "Download contact list (CSV)",
                plan.to_csv(index=False),
                file_name="action_plan.csv",
                mime="text/csv",
            )
            st.caption(
                "Suggested actions are templates to test. Whether any of them changes churn is "
                "exactly what the randomized holdout is for."
            )

            st.subheader("Look up a customer")
            customer = st.selectbox("Customer", plan["customer_id"].head(300))
            row = plan[plan["customer_id"] == customer].iloc[0]
            a, b, c = st.columns(3)
            a.metric("Predicted churn probability", f"{row['churn_probability']:.0%}")
            b.metric("Break-even probability", f"{row['breakeven_probability']:.0%}")
            c.metric("Expected value of contact", money(row["expected_value"]))
            st.write(f"**Risk drivers:** {row['risk_drivers']}")
            st.write(f"**Suggested first action:** {row['suggested_action']}")
            if row["secondary_action"]:
                st.write(f"**Then try:** {row['secondary_action']}")

    with tab_model:
        model = summary["model"]
        holdout, interval = model["holdout"], model["holdout_ci"]
        columns = st.columns(4)
        columns[0].metric(
            "Holdout ROC-AUC",
            f"{holdout['roc_auc']:.3f}",
            help=f"95% CI {interval['roc_auc'][0]:.3f} to {interval['roc_auc'][1]:.3f}",
        )
        columns[1].metric("Holdout PR-AUC", f"{holdout['pr_auc']:.3f}")
        columns[2].metric("Calibration error (ECE)", f"{holdout['ece']:.3f}")
        columns[3].metric("Brier skill vs base rate", f"{holdout['brier_skill']:.2f}")

        st.subheader("Leakage audit")
        audit = pd.read_csv(OUTPUTS / "leakage_audit.csv")[
            ["feature_set", "roc_auc", "verdict"]
        ].round({"roc_auc": 3})
        st.dataframe(audit, hide_index=True, use_container_width=True)
        st.caption(
            "Satisfaction score and churn score are recorded at or after the outcome. "
            "A 0.99 AUC is the signature of leakage, not of a good model."
        )

        figures = summary["figures"]
        left, right = st.columns(2)
        left.image(str(ROOT / figures["05_calibration"]))
        right.image(str(ROOT / figures["06_gain_curve"]))


main()
