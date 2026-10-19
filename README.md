# ChurnOpt: churn prediction you can put dollar figures on

A churn model, a leakage audit, a SQL analysis, and a profit-based contact policy, built on the IBM Telco customer churn sample dataset (7,043 customers). The aim is not a high AUC. It is a set of probabilities that mean what they say, so that "who should we contact, and is it worth it?" has an answer someone can check.

## Results at a glance

| | |
|---|---|
| **Model quality** | Holdout ROC-AUC **0.911** (95% CI 0.895 to 0.927), PR-AUC 0.788 against a 26.5% base rate. The one-feature "contract type" rule scores 0.756 |
| **Calibration** | Expected calibration error **0.020**. Predicted risk matches observed churn, so it is safe to multiply by dollars |
| **Leakage caught** | `Satisfaction Score` and `Churn Score` are recorded at or after the outcome and would lift AUC from 0.899 to **0.998**. They are excluded and the effect is measured |
| **Where the loss is** | **30.5%** of monthly revenue is lost to churn. One segment, month-to-month fiber, is **69%** of it |
| **Campaign** (assumed 30% save rate, \$25 per contact, \$20K budget) | 800 contacts, expected net profit **\$59K**, ROI (net / cost) 297%. Backtested on actual outcomes: \$62K, **1.9x** the month-to-month-fiber rule at the same spend |
| **How fragile is that?** | The campaign still breaks even if only **7.6%** of would-be churners are saved. 45 of 48 cells in the sensitivity grid are profitable |

> **Every dollar figure rests on assumptions** (save rate, contact cost, margin, horizon). The dataset contains no campaign outcomes. The break-even rate, the sensitivity analysis and the experiment design exist because of that.

![Profit curve](outputs/figures/07_profit_curve.png)

## What makes this different from a typical churn notebook

1. **A leakage audit with numbers.** The dataset ships columns that are really outcomes (a satisfaction survey, another model's churn score, the churn reason). Each exclusion is documented in [configs/project.yaml](configs/project.yaml) and a test fails if any of them reaches the model.
2. **Calibration before dollars.** Class weighting, the common fix for imbalance, inflates probabilities (here by 1.45x). Decisions multiply probability by value, so the model is trained without it and checked with a reliability curve.
3. **A decision, not a score.** Expected value per customer, a budget-limited contact list, a backtest against four policies, a break-even save rate, and sensitivity analysis. ROI is net over cost.
4. **Uncertainty everywhere.** Repeated cross-validation, bootstrap intervals on the holdout and on profit, a one-standard-error rule for model choice.
5. **SQL, not just pandas.** Six business questions answered in DuckDB with CTEs and window functions, including a like-for-like offer comparison.
6. **No invented evidence.** There is no A/B result here because there is no experiment. The project designs the randomized test that would measure the save rate, and validates the sample sizes by simulation.
7. **Reproducible and tested.** One command rebuilds every number and figure from the raw CSV in about a minute. 81 tests, 96% coverage, CI runs the full pipeline.

## Findings

![Churn by contract and tenure](outputs/figures/01_churn_by_contract_tenure.png)

- **Month-to-month customers in their first year are 55% of all churners**, and 54% of them leave.
- **Churners pay more** (\$74 against \$61 a month), so lost revenue (30.5%) is larger than lost customers (26.5%).
- **The top three segments are 85% of the revenue lost** ([outputs/figures/02_revenue_pareto.png](outputs/figures/02_revenue_pareto.png)).
- **Contract type dominates** in every view: odds ratios, SHAP and permutation importance. Two-year contracts cut the odds of churn by about 93% with other factors held fixed.
- **Tech support or online security** goes with 17 to 25 points less churn within the same plan on month-to-month contracts. Association only.
- **The retention offers cannot be ranked from this data.** Each offer goes almost entirely to one tenure band, so raw rates compare customer age, not offers. Offer E looks like 53% churn; against no-offer customers in the same tenure band it is still 8 points worse, but nobody randomized who got what.

## Model

