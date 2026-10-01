# Data

`raw/telco.csv` is the IBM Telco customer churn sample dataset: 7,043 synthetic
customers of a fictional California telco, one observed quarter, 50 columns.

- **Cross-sectional.** Each customer appears once. There is no month-by-month
  history, so there is no forecasting horizon and no way to measure drift.
- **Not real customers.** It is a sample dataset published by IBM for
  demonstrating analytics tools. Nothing here describes real people.
- **Target.** `Churn Label` = `Yes` means the customer left during the observed
  quarter (1,869 of 7,043, 26.5%).
- **Columns recorded at or after the outcome.** `Satisfaction Score`,
  `Churn Score`, `Churn Category`, `Churn Reason`, `Customer Status` and `CLTV`
  are never used as model inputs. `configs/project.yaml` lists each exclusion
  with its reason, and `outputs/leakage_audit.csv` measures what they would do.

Check the dataset's current terms of use before redistributing it in a fork.
