"""
Figures for the README, notebooks and dashboard.

Every function takes plain DataFrames and returns a matplotlib Figure; `save`
in style.py writes it. Titles state the finding, not the chart type.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .style import (
    BLUE,
    BLUE_RAMP,
    BLUE_SEQUENTIAL,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    NEUTRAL,
    ORANGE,
    RED,
    SURFACE,
    apply_style,
    footnote,
    headline,
)

apply_style()


def _new(figsize=(9.5, 5.4), top: float = 0.80, left: float = 0.11, bottom: float = 0.13):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=top, left=left, bottom=bottom, right=0.97)
    return fig, ax


MODEL_NAMES = {
    "prior": "Base rate only",
    "contract_only": "Contract type only",
    "logistic": "Logistic regression",
    "hist_gb": "Gradient boosting",
    "hist_gb_calibrated": "Gradient boosting + calibration",
}


def _money(value: float) -> str:
    """Dollar label; the backslash stops matplotlib reading paired $ signs as math."""
    sign = "-" if value < 0 else ""
    value = abs(value)
    return f"{sign}\\${value / 1000:,.0f}K" if value >= 1000 else f"{sign}\\${value:,.0f}"


def churn_by_contract_and_tenure(df: pd.DataFrame):
    """Churn rate by contract type and tenure band (ordered groups, blue ramp)."""
    bands = pd.cut(
        df["tenure_months"],
        [0, 12, 24, 48, 1000],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
    )
    table = (
        df.assign(band=bands)
        .groupby(["contract", "band"], observed=True)["churn"]
        .agg(["mean", "size"])
    )
    contracts = ["Month-to-Month", "One Year", "Two Year"]
    band_names = list(bands.cat.categories)

    fig, ax = _new()
    width = 0.19
    for i, band in enumerate(band_names):
        rates = [table.loc[(c, band), "mean"] * 100 for c in contracts]
        positions = np.arange(len(contracts)) + (i - 1.5) * (width + 0.015)
        ax.bar(
            positions,
            rates,
            width,
            color=BLUE_RAMP[i],
            label=band,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        for x, rate in zip(positions, rates):
            ax.text(x, rate + 1, f"{rate:.0f}%", ha="center", va="bottom", fontsize=8.5, color=INK)

    ax.set_xticks(range(len(contracts)))
    ax.set_xticklabels(contracts)
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, 62)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Tenure", ncol=4, loc="upper right", fontsize=9, title_fontsize=9)
    headline(
        fig,
        "Churn is a month-to-month problem, and worst in the first year",
        f"Share of customers who churned, by contract and tenure band ({len(df):,} customers)",
    )
    footnote(fig, "Source: IBM Telco sample dataset. Observed quarter, cross-sectional.")
    return fig


def revenue_pareto(pareto: pd.DataFrame, top_n: int = 6):
    """Monthly revenue lost to churn by contract x internet segment."""
    data = pareto.head(top_n).iloc[::-1]
    labels = [f"{c}, {i}" for c, i in zip(data["contract"], data["internet_type"])]

    fig, ax = _new(top=0.80, left=0.30)
    colors = [BLUE if i == len(data) - 1 else NEUTRAL for i in range(len(data))]
    ax.barh(labels, data["monthly_revenue_lost"], color=colors, height=0.62, edgecolor=SURFACE)
    for y, (lost, share, cum) in enumerate(
        zip(data["monthly_revenue_lost"], data["share_of_loss_pct"], data["cumulative_share_pct"])
    ):
        ax.text(
            lost + data["monthly_revenue_lost"].max() * 0.01,
            y,
            f"{_money(lost)}/mo  ({share:.0f}% of loss)",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, data["monthly_revenue_lost"].max() * 1.45)
    ax.set_xlabel("Monthly recurring revenue lost to churn")
    ax.xaxis.set_major_formatter(lambda v, _: _money(v))
    ax.grid(axis="y", visible=False)
    top_share = pareto.iloc[0]["share_of_loss_pct"]
    top3 = pareto.iloc[2]["cumulative_share_pct"]
    headline(
        fig,
        f"One segment, month-to-month fiber, is {top_share:.0f}% of the revenue lost to churn",
        f"Top three segments account for {top3:.0f}% of the loss",
    )
    footnote(fig, "Source: sql/03_revenue_at_risk_pareto.sql")
    return fig


def leakage_audit(audit: pd.DataFrame):
    """Cross-validated AUC for progressively leakier feature sets (lollipop from chance)."""
    data = audit.iloc[::-1].reset_index(drop=True)
    excluded = data["verdict"].str.startswith("excluded") & data["feature_set"].str.contains(
        "satisfaction|churn_score"
    )

    fig, ax = _new(figsize=(10, 5.4), left=0.40, bottom=0.2)
    for y, (auc, bad) in enumerate(zip(data["roc_auc"], excluded)):
        color = ORANGE if bad else BLUE
        ax.hlines(y, 0.5, auc, color=color, linewidth=2.5)
        ax.plot(
            auc, y, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5
        )
        ax.text(auc + 0.006, y, f"{auc:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["feature_set"], fontsize=9)
    ax.set_xlim(0.5, 1.03)
    ax.set_xlabel("Cross-validated ROC-AUC (0.5 = chance)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, linestyle="", label="Usable at decision time"),
            Line2D(
                [0],
                [0],
                marker="o",
                color=ORANGE,
                linestyle="",
                label="Recorded at or after the outcome",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=9,
    )
    headline(
        fig,
        "Leaky columns add nearly ten AUC points and would sink the model in use",
        "Same logistic regression, only the allowed columns change. A 0.99 AUC here is a warning, not a result.",
    )
    footnote(
        fig, "5-fold cross-validation on the training split. Source: outputs/leakage_audit.csv"
    )
    return fig


def model_comparison(summary: pd.DataFrame, champion: str):
    """Cross-validated ROC-AUC and log loss per candidate, with 95% intervals."""
    data = summary.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    fig.subplots_adjust(top=0.76, left=0.22, bottom=0.14, right=0.97, wspace=0.08)

    for ax, metric, label, fmt in [
        (axes[0], "roc_auc", "ROC-AUC (higher is better)", "{:.3f}"),
        (axes[1], "log_loss", "Log loss (lower is better)", "{:.3f}"),
    ]:
        for y, row in data.iterrows():
            mean, se = row[f"{metric}_mean"], row[f"{metric}_se"]
            color = BLUE if row["model"] == champion else INK_MUTED
            ax.hlines(y, mean - 1.96 * se, mean + 1.96 * se, color=color, linewidth=2.5)
            ax.plot(
                mean,
                y,
                "o",
                color=color,
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
            ax.text(mean, y + 0.28, fmt.format(mean), ha="center", fontsize=8.5, color=INK)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)

    axes[0].set_yticks(range(len(data)))
    axes[0].set_yticklabels(data["model"].map(MODEL_NAMES))
    headline(
        fig,
        "Gradient boosting edges out logistic regression; both far beat the simple rules",
        f"Repeated 5-fold CV on the training split, mean with 95% interval. Champion (blue): {MODEL_NAMES[champion]}",
    )
    return fig


def calibration_curve(
    champion_table: pd.DataFrame,
    reference_table: pd.DataFrame,
    ece: float,
    overestimate: float,
    reference_label: str = "Class-weighted logistic regression",
):
    """Reliability diagram: champion versus a class-weighted model."""
    fig, ax = _new(figsize=(8.6, 6.4), top=0.83, left=0.11)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.text(0.80, 0.745, "perfect", fontsize=8.5, color=INK_SECONDARY, rotation=33)

    ax.plot(
        reference_table["mean_predicted"],
        reference_table["observed_rate"],
        "-o",
        color=ORANGE,
        markersize=6,
        markeredgecolor=SURFACE,
    )
    ax.plot(
        champion_table["mean_predicted"],
        champion_table["observed_rate"],
        "-o",
        color=BLUE,
        markersize=6,
        markeredgecolor=SURFACE,
    )
    ax.text(
        0.56,
        0.12,
        f"{reference_label}\npredicts {overestimate:.1f}x the real risk on average",
        color=INK,
        fontsize=9,
        va="center",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted churn probability (decile mean)")
    ax.set_ylabel("Observed churn rate")
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, marker="o", label=f"Champion (ECE {ece:.3f})"),
            Line2D([0], [0], color=ORANGE, marker="o", label=reference_label),
        ],
        loc="upper left",
        fontsize=9,
    )
    headline(
        fig,
        "Predicted risk matches observed churn, so it is safe to use in dollar math",
        "Held-out customers grouped into risk deciles",
    )
    footnote(
        fig,
        f"Decisions multiply probability by value; a probability {overestimate:.1f}x too high "
        f"overstates every payoff by {overestimate:.1f}x.",
    )
    return fig


def gain_curve(oof: pd.DataFrame, rule_mask: np.ndarray, rule_label: str):
    """Share of all churners reached as the contact list grows."""
    ranked = oof.sort_values("churn_probability", ascending=False)
    total = ranked["churn"].sum()
    reached = ranked["churn"].cumsum().to_numpy() / total
    share = np.arange(1, len(ranked) + 1) / len(ranked)

    fig, ax = _new(figsize=(8.2, 5.8), top=0.82)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle="--", linewidth=1.2)
    ax.plot(share, reached, color=BLUE, linewidth=2.5)
    rule_share = rule_mask.mean()
    rule_reached = oof.loc[rule_mask, "churn"].sum() / total
    ax.plot(
        rule_share,
        rule_reached,
        "o",
        color=ORANGE,
        markersize=10,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )

    k20 = int(0.2 * len(ranked))
    ax.plot(
        0.2,
        reached[k20 - 1],
        "o",
        color=BLUE,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    ax.text(
        0.215,
        reached[k20 - 1] - 0.08,
        f"Top 20% by risk reach {reached[k20 - 1]:.0%} of churners",
        fontsize=9.5,
        color=INK,
        va="center",
    )
    ax.text(
        rule_share + 0.025,
        rule_reached + 0.06,
        f"{rule_label}\n{rule_share:.0%} of customers reach {rule_reached:.0%} of churners",
        fontsize=9,
        color=INK,
        va="center",
    )
    ax.text(0.74, 0.62, "random", fontsize=8.5, color=INK_SECONDARY, rotation=38)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Share of customers contacted (highest risk first)")
    ax.set_ylabel("Share of all churners reached")
    headline(
        fig,
        "Contacting the riskiest 20% reaches over half of all churners",
        "Cumulative gain, out-of-fold predictions for all customers",
    )
    return fig


def profit_curve_chart(curve: pd.DataFrame, k_budget: int, k_optimal: int):
    """Expected versus realized net profit as the contact list grows."""
    fig, ax = _new(figsize=(9.5, 5.4), top=0.80)
    ax.plot(curve["contacts"], curve["expected_net"], color=BLUE, linewidth=2.5)
    ax.plot(
        curve["contacts"], curve["realized_net"], color=ORANGE, linewidth=2.5, linestyle=(0, (5, 2))
    )
    ax.axhline(0, color=INK_MUTED, linewidth=1)

    for k, label in [(k_budget, "Budget limit"), (k_optimal, "Profit-maximizing list")]:
        if k <= 0:
            continue
        ax.axvline(k, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.plot(
            k,
            curve.loc[k - 1, "expected_net"],
            "o",
            color=BLUE,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.annotate(
            f"{label}: {k:,} contacts\nexpected {_money(curve.loc[k - 1, 'expected_net'])}",
            xy=(k, curve.loc[k - 1, "expected_net"]),
            xytext=(
                k + len(curve) * 0.03,
                curve.loc[k - 1, "expected_net"] * (0.55 if k == k_budget else 1.03),
            ),
            fontsize=9,
            color=INK,
        )
    ax.set_xlabel("Customers contacted (ranked by expected value)")
    ax.set_ylabel("Cumulative net profit")
    ax.yaxis.set_major_formatter(lambda v, _: _money(v))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlim(0, len(curve) * 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=BLUE, linewidth=2.5, label="Expected (calibrated probabilities)"
            ),
            Line2D(
                [0],
                [0],
                color=ORANGE,
                linewidth=2.5,
                linestyle=(0, (5, 2)),
                label="Realized (actual churn outcomes)",
            ),
        ],
        loc="upper right",
        fontsize=9,
    )
    headline(
        fig,
        "Profit rises, peaks, then falls, and the expected curve tracks the realized one",
        "Each extra contact costs the same but brings less expected value. Save rate and costs are assumptions.",
    )
    footnote(
        fig,
        "Out-of-fold predictions, all customers. Realized = actual churners x assumed save rate x customer value, minus contact cost.",
    )
    return fig


def policy_comparison(policies: pd.DataFrame, contacts: int):
    """Realized net profit of each contact policy at the same number of contacts."""
    data = policies.iloc[::-1].reset_index(drop=True)
    fig, ax = _new(figsize=(9.5, 4.6), top=0.78, left=0.34)
    colors = [BLUE if p.startswith("Model: rank by expected") else NEUTRAL for p in data["policy"]]
    ax.barh(data["policy"], data["net"], color=colors, height=0.6, edgecolor=SURFACE)
    for y, (net, precision) in enumerate(zip(data["net"], data["precision"])):
        ax.text(
            net + data["net"].max() * 0.01,
            y,
            f"{_money(net)}  ({precision:.0%} are churners)",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, data["net"].max() * 1.42)
    ax.set_xlabel("Realized net profit")
    ax.xaxis.set_major_formatter(lambda v, _: _money(v))
    ax.grid(axis="y", visible=False)
    headline(
        fig,
        "The model's list earns more than the obvious rule at the same spend",
        f"Same {contacts:,} contacts for every policy. Backtest on out-of-fold predictions and actual outcomes.",
    )
    return fig


def sensitivity_heatmap(grid: pd.DataFrame, base_save: float, base_cost: float):
    """Optimal expected profit across save rates and contact costs."""
    pivot = grid.pivot(
        index="save_rate", columns="cost_per_contact", values="expected_net"
    ).sort_index(ascending=False)
    values = pivot.to_numpy()
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQUENTIAL)
    cmap.set_bad(NEUTRAL)
    masked = np.ma.masked_where(values <= 0, values)

    fig, ax = _new(figsize=(9.6, 5.8), top=0.82, left=0.12)
    ax.grid(False)
    ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = "none" if values[i, j] <= 0 else _money(values[i, j])
            dark = values[i, j] > 0 and values[i, j] > np.nanmax(values) * 0.5
            ax.text(
                j, i, text, ha="center", va="center", fontsize=9, color=SURFACE if dark else INK
            )
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels([_money(c) for c in pivot.columns])
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels([f"{s:.0%}" for s in pivot.index])
    ax.set_xlabel("Cost per contact")
    ax.set_ylabel("Save rate (share of would-be churners retained)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if base_save in list(pivot.index) and base_cost in list(pivot.columns):
        i = list(pivot.index).index(base_save)
        j = list(pivot.columns).index(base_cost)
        ax.add_patch(
            plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2.2)
        )
    headline(
        fig,
        "The campaign pays off under most assumptions, but not in one corner",
        "Best expected net profit within the budget. Gray = no profitable campaign. Outlined cell = base case.",
    )
    footnote(
        fig, "Source: configs/economics.yaml grid. All values are assumptions, not observations."
    )
    return fig


def tornado_chart(tornado: pd.DataFrame):
    """One-at-a-time swing in profit for each assumption."""
    data = tornado.iloc[::-1].reset_index(drop=True)
    base = data["base_net"].iloc[0]

    fig, ax = _new(figsize=(9.8, 5.2), top=0.78, left=0.30, bottom=0.2)
    for y, row in data.iterrows():
        ax.barh(
            y, row["net_at_low"] - base, left=base, color=ORANGE, height=0.55, edgecolor=SURFACE
        )
        ax.barh(y, row["net_at_high"] - base, left=base, color=BLUE, height=0.55, edgecolor=SURFACE)
        for value, x in (
            (row["low_value"], row["net_at_low"]),
            (row["high_value"], row["net_at_high"]),
        ):
            side = "left" if x < base else "right"
            offset = -1 if x < base else 1
            ax.text(
                x + offset * base * 0.01,
                y,
                f"{_money(x)}",
                ha="right" if side == "left" else "left",
                va="center",
                fontsize=8.5,
                color=INK,
            )
    ax.axvline(base, color=INK, linewidth=1.2)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(
        [
            f"{a}  ({_range_label(a, lo, hi)})"
            for a, lo, hi in zip(data["assumption"], data["low_value"], data["high_value"])
        ],
        fontsize=9,
    )
    ax.set_xlabel("Expected net profit")
    ax.xaxis.set_major_formatter(lambda v, _: _money(v))
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Patch(color=ORANGE, label="Assumption at low end of range"),
            Patch(color=BLUE, label="Assumption at high end of range"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.14),
        ncol=2,
        fontsize=9,
    )
    first, second = tornado["assumption"].iloc[0], tornado["assumption"].iloc[1]
    headline(
        fig,
        f"{first} and {second.lower()} swing profit the most",
        f"Base case {_money(base)} with the budget fixed in dollars. Each assumption moved alone across its range.",
    )
    footnote(
        fig,
        "Only the save rate can be measured directly, with a randomized holdout (see the experiment design).",
    )
    return fig


def _range_label(assumption: str, low: float, high: float) -> str:
    if "rate" in assumption.lower() or "margin" in assumption.lower():
        return f"{low:.0%} to {high:.0%}"
    if "Cost" in assumption:
        return f"{_money(low)} to {_money(high)}"
    return f"{low:g} to {high:g} months"


def odds_ratio_forest(table: pd.DataFrame, top_n: int = 14):
    """Adjusted odds ratios with 95% intervals (log scale)."""
    data = table.head(top_n).iloc[::-1].reset_index(drop=True)
    fig, ax = _new(figsize=(10, 6.6), top=0.85, left=0.40, bottom=0.17)
    for y, row in data.iterrows():
        if not row["significant"]:
            color = NEUTRAL
        else:
            color = RED if row["odds_ratio"] > 1 else BLUE
        ax.hlines(y, row["ci_low"], row["ci_high"], color=color, linewidth=2.5)
        ax.plot(
            row["odds_ratio"],
            y,
            "o",
            color=color,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ax.text(
            row["ci_high"] * 1.08,
            y,
            f"{row['odds_ratio']:.2f}",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    ax.axvline(1, color=INK_MUTED, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4])
    ax.get_xaxis().set_major_formatter(lambda v, _: f"{v:g}")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["term"], fontsize=9)
    ax.set_xlabel("Adjusted odds ratio for churn (log scale; 1 = no effect)")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=RED, linestyle="", label="Higher churn odds"),
            Line2D([0], [0], marker="o", color=BLUE, linestyle="", label="Lower churn odds"),
            Line2D(
                [0], [0], marker="o", color=NEUTRAL, linestyle="", label="Not significant at 5%"
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.1),
        ncol=3,
        fontsize=9,
    )
    headline(
        fig,
        "Contract type dwarfs everything else once other factors are held fixed",
        "Logistic regression odds ratios with 95% confidence intervals. Association, not proof of cause.",
    )
    return fig


def importance_bars(
    table: pd.DataFrame, column: str, title: str, subtitle: str, xlabel: str, top_n: int = 12
):
    """Horizontal bars of feature importance (single hue)."""
    data = table.head(top_n).iloc[::-1]
    fig, ax = _new(figsize=(9, 5.4), top=0.82, left=0.27)
    ax.barh(
        data["feature"].str.replace("_", " "),
        data[column],
        color=BLUE,
        height=0.62,
        edgecolor=SURFACE,
    )
    for y, value in enumerate(data[column]):
        ax.text(
            value + data[column].max() * 0.01, y, f"{value:.3f}", va="center", fontsize=9, color=INK
        )
    ax.set_xlim(0, data[column].max() * 1.15)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    headline(fig, title, subtitle)
    return fig


def power_curve(
    design: pd.DataFrame,
    baseline_rate: float,
    breakeven_save_rate: float,
    assumed_save_rate: float,
    n_for_breakeven: int,
    n_budget_arm: int,
    detectable_with_budget: float,
):
    """Customers per arm needed to detect each save rate."""
    fig, ax = _new(figsize=(9.6, 5.4), top=0.80)
    ax.plot(
        design["save_rate"],
        design["n_treatment"],
        "-o",
        color=BLUE,
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    for _, row in design.iterrows():
        ax.text(
            row["save_rate"],
            row["n_treatment"] * 1.18,
            f"{int(row['n_treatment']):,}",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yscale("log")
    ax.set_xlabel("True save rate the test should detect")
    ax.set_ylabel("Customers per arm (log scale)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlim(design["save_rate"].min() - 0.02, design["save_rate"].max() + 0.03)
    ax.grid(axis="x", visible=False)
    bottom = ax.get_ylim()[0]

    ax.axvline(assumed_save_rate, color=INK_MUTED, linestyle=":")
    ax.text(
        assumed_save_rate - 0.004,
        ax.get_ylim()[1] * 0.92,
        "assumed save rate",
        fontsize=8.5,
        color=INK_SECONDARY,
        rotation=90,
        va="top",
        ha="right",
    )
    ax.axvline(breakeven_save_rate, color=ORANGE, linewidth=1.5)
    ax.text(
        breakeven_save_rate + 0.004,
        bottom * 1.15,
        f"break-even save rate ({breakeven_save_rate:.1%})",
        fontsize=8.5,
        color=INK_SECONDARY,
        rotation=90,
        va="bottom",
    )

    ax.axhline(n_budget_arm, color=INK_MUTED, linestyle=":")
    ax.text(
        ax.get_xlim()[1] - 0.004,
        n_budget_arm * 0.92,
        f"Budget list split 50/50 ({n_budget_arm:,} per arm) detects {detectable_with_budget:.0%} or more",
        fontsize=8.5,
        color=INK_SECONDARY,
        ha="right",
        va="top",
    )
    headline(
        fig,
        f"Detecting the break-even save rate takes about {n_for_breakeven:,} customers per arm",
        f"80% power, 5% two-sided test, {baseline_rate:.0%} churn among contacted customers without an offer",
    )
    footnote(fig, "Design only: no campaign data exists, so no treatment effect is reported.")
    return fig
