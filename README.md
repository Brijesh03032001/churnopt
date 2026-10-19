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
