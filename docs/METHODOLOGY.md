# Methodology

How the numbers in the README were produced, what they rest on, and where they stop being reliable.

## 1. Problem framing

**Business question.** Given a limited retention budget, which customers should be contacted, and is the campaign worth running?

**Unit of analysis.** The customer. The IBM Telco sample has one row per customer, so each customer appears once and no customer can sit on both sides of a train/test split.

**Target.** `Churn Label = Yes`: the customer left during the observed quarter (1,869 of 7,043, 26.5%). The dataset has no timestamps, so a probability here means "churns during this quarter", not "churns next month". There is no forecasting horizon and no way to test drift over time.

**Data quality checks** (`TelcoDataLoader.validate`): unique ids, no missing values in required columns, non-negative tenure, positive charges, a plausible churn rate, `Customer Status` agreeing with the label, and `Internet Type = None` wherever there is no internet service. pandas reads the text `None` as missing by default; here it is a real category (no offer, no internet), so the loader reads with `keep_default_na=False`. A test guards this.

## 2. Leakage audit

Columns recorded at or after the outcome would make a model look excellent and be useless for choosing whom to contact beforehand.

| Column | Evidence | Treatment |
|---|---|---|
| `satisfaction_score` | Scores 1 and 2 churn 100%, scores 4 and 5 churn 0% | Excluded |
| `churn_score` | Output of IBM's own churn model, correlation 0.66 with the label | Excluded |
| `churn_reason`, `churn_category` | Populated only for customers who already churned (1,869 of 1,869, 0 of the rest) | Excluded |
| `customer_status`, `churn_label` | Define the target | Excluded |
| `cltv` | Opaque corporate formula, cannot be reproduced, adds no predictive lift | Excluded |
| `country`, `state`, `quarter` | Constant | Excluded |
| `city`, `zip_code`, `latitude`, `longitude` | High-cardinality geography, overfitting risk | Excluded |

`outputs/leakage_audit.csv` holds the same logistic regression cross-validated on progressively leakier feature sets:

| Feature set | ROC-AUC |
|---|---|
| Strict: no lifetime-cumulative columns | 0.878 |
| Standard (project default) | 0.899 |
| Standard + `cltv` | 0.898 |
| Standard + `satisfaction_score` | 0.992 |
| Standard + `churn_score` | 0.981 |
| Standard + both | 0.998 |

**Cumulative columns.** Billing totals and the referral count are measured up to the observation date, which for a churner is the exit date. They carry a small look-ahead: a customer's lifetime total grows until they leave. They stay in the standard set because a CRM would know them at scoring time, and because the strict set (which drops them) is reported next to it: champion holdout ROC-AUC 0.911 standard, 0.887 strict.

The exclusion list is the only way columns are removed; the feature list is never "whatever is left". `tests/test_features.py` fails if any excluded column reaches the model in either feature set.

## 3. Validation protocol

1. **Split.** Stratified 80/20 on the customer, fixed seed. The 20% test set is untouched until step 4.
2. **Comparison.** Five candidates are compared by repeated stratified 5-fold cross-validation (3 repeats, 15 folds) on the training set only. Preprocessing lives inside each model pipeline, so scalers and encoders are refitted per fold and never see held-out rows.
3. **Selection.** The selection metric is log loss, a proper scoring rule that rewards calibrated probabilities. The one-standard-error rule applies: if a simpler model is within one standard error of the best, the simpler one wins. Folds overlap in their training rows, so they are not independent draws; the standard error divides the fold standard deviation by the square root of the folds per repeat (√5) rather than the square root of all 15 folds, which keeps it conservatively wide.
4. **Test.** The champion is fitted on the training set and scored once on the test set, with percentile bootstrap intervals (1,000 resamples of customers).
5. **Out-of-fold predictions.** A fresh 5-fold run produces a prediction for every customer from a model that never saw that customer. These feed the decision analysis and the dashboard.
6. **Final model.** Fitted on all customers for scoring new data (`models/churn_model.joblib`).

Candidates: base rate only; logistic regression on contract type alone (the obvious rule); regularized logistic regression with the regularization strength chosen by internal cross-validation; histogram gradient boosting with shallow trees and early stopping; and the boosted model with sigmoid calibration. Hyperparameters for the boosted model are fixed and conservative rather than tuned, so cross-validation scores are not inflated by a search.

## 4. Calibration

Decisions multiply probability by dollars, so a probability that runs 1.5x high overstates every payoff by 1.5x.

- No candidate uses class weights. The reference model in `build_class_weighted_reference` exists only to show the damage: on the test set it predicts 1.45x the real churn rate on average.
- Calibration is measured, not assumed: expected calibration error (equal-frequency bins), reliability curves, and Brier skill against the base-rate predictor. The champion's holdout ECE is 0.020.
- The dollar version of the same check is in the profit curve: expected profit (from the probabilities) and realized profit (from what customers did) agree to within 5%.

## 5. Economics

Expected value of contacting customer *i*:

```
EV_i = p_i x save_rate x V_i - cost_per_contact
V_i  = monthly_charge_i x gross_margin x sum_{t=1..H} (1 + d)^-t
```

- `p_i`: calibrated, out-of-fold churn probability.
- `save_rate`: share of would-be churners who stay after contact. **Unobserved.** The base case is 30%.
- `cost_per_contact`: charged for every contacted customer, which prices in offers wasted on customers who would have stayed. Base case \$25.
- `gross_margin` 40%, horizon `H` 12 months, monthly discount rate `d` 1%: together a saved customer is worth about 4.5 months of charge, around \$290 on average.
- Break-even probability for a customer: `cost / (save_rate x V_i)`. Contact only above it.

| Assumption | Base | Range tested | Status |
|---|---|---|---|
| Save rate | 30% | 10% to 50% | Unknown until a randomized test measures it |
| Cost per contact | \$25 | \$10 to \$80 | Placeholder for CRM and finance actuals |
| Gross margin | 40% | 25% to 55% | Placeholder |
| Value horizon | 12 months | 6 to 24 | Placeholder; survival data would pin it down |
| Budget | \$20,000 | n/a | Illustrative |

**ROI** is net profit divided by cost. **Realized profit** (the backtest) applies the assumed save rate to customers who actually churned and charges the contact cost for everyone contacted, so it uses real outcomes but not a real save rate.

**Policies compared at the same number of contacts:** rank by expected value; rank by churn probability; the rule "month-to-month contract with fiber optic"; random contacts (mean of 200 draws).

**Break-even save rate** for a top-*k* list: `cost x k / sum(p_i x V_i)` over the list. At the 800-contact budget it is 7.6%.

**Sensitivity.** A grid over save rate and cost (the budget is in dollars, so a higher cost means fewer contacts), and a one-at-a-time swing for each assumption. The interval on realized profit comes from bootstrapping customers and rebuilding the top-*k* list in every resample.

## 6. Explanations

- **Odds ratios** from a logistic regression with 95% intervals, on interpretable terms. Left out on purpose: cumulative billing totals (near-linear in tenure and charge), offers (each offer is nearly a function of tenure band, so its coefficient would mix up the offer and the customer's age), referral count (non-monotonic and tied to tenure), and gender.
- **Permutation importance** on the test set, shuffling raw columns before preprocessing so one-hot columns move together.
- **SHAP** values for the champion on 400 test customers, summed across one-hot columns so each original feature has one number.

All three describe association in observational data. None says that changing a feature would change churn.

## 7. Experiment design

[src/experiments/design.py](../src/experiments/design.py) sizes the randomized holdout that would measure the save rate. A save rate `s` takes churn among contacted customers from `b` to `b(1 - s)`, so the detectable difference is `b x s`. Sample sizes use Cohen's h with a two-sided 5% test and 80% power, and a Monte Carlo simulation of the two-proportion test confirms each row (simulated power 0.80 to 0.81). No treatment effect is reported because there is no campaign data.

## 8. Reproducibility

- Fixed seeds throughout; a from-scratch run reproduces every headline number exactly.
- `pandas` is capped below 3 because pandas 3 changes string dtypes. Other packages use minimum versions.
- One command, `python run_pipeline.py`, rebuilds every table and figure from the raw CSV in about a minute.
- 81 tests (96% coverage) cover data validation, the leakage guard, metrics and calibration, the economics (including ROI as net over cost), experiment sizing, the SQL results, and an end-to-end run of every stage on a sample. CI runs them and also checks that the headline AUC and calibration error have not moved.

## 9. Known limitations

- Synthetic, single-quarter, single-region sample data.
- No time axis: no horizon, no drift test.
- The model ranks by churn risk, not by how persuadable a customer is. Without treatment data, uplift cannot be modeled; contacting a customer who would stay anyway, or leave anyway, is wasted spend.
- Dollar results depend on the assumed save rate, cost, margin and horizon.
- The offer comparison is observational and bands are coarse.
- `Customer Status = Joined` customers (454) joined during the quarter and cannot have churned in it, which slightly favors short-tenure non-churners.

## 10. Changes from the first version

The first version of this project reported 659 customers targeted, "105% ROI", a \$1.24M unconstrained profit, and a 0.91 ROC-AUC. A review found that none of them held up:

| First version | Problem | Now |
|---|---|---|
| Customer-month "snapshots" with randomly assigned churn dates | The time axis was fabricated; 84% of test customers also appeared in training | One row per customer, customer-level split |
| Class-weighted probabilities used as true probabilities | Mean predicted risk was 18x the real rate; after calibration no row had positive expected value | No class weights, calibration measured, dollars checked against outcomes |
| "659 customers", "25,100 customers" | These were customer-months; there are 7,043 customers | Counts are customers |
| "ROI 105%" | That was value divided by cost; net over cost was 4.6% | ROI is net / cost |
| Rolling "trend" features | Constant for large datasets | Removed |
| Behavioral features (tickets, late payments) | Random numbers | Removed |
| A/B test with p = 1e-33 | Outcomes were drawn from the model's own probabilities | An experiment design, no fabricated result |
| `pip install` on a fresh machine | pandas 3 broke the loader; xgboost needed a system library | Capped pandas; no system dependencies |

The old code is not part of this repository.
