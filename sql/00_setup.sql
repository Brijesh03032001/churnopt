-- Setup: expose the raw CSV as a clean `customers` view with snake_case columns.
-- {{raw_csv}} is replaced with the path from configs/project.yaml by the runner.
-- nullstr = '' keeps the literal string 'None' (no offer / no internet) as a category.
CREATE OR REPLACE VIEW customers AS
SELECT
    "Customer ID"                       AS customer_id,
    "Gender"                            AS gender,
    "Senior Citizen"                    AS senior_citizen,
    "Dependents"                        AS dependents,
    "Number of Referrals"               AS number_of_referrals,
    "Tenure in Months"                  AS tenure_months,
    "Offer"                             AS offer,
    "Internet Type"                     AS internet_type,
    "Contract"                          AS contract,
    "Payment Method"                    AS payment_method,
    "Paperless Billing"                 AS paperless_billing,
    "Online Security"                   AS online_security,
    "Premium Tech Support"              AS tech_support,
    "Monthly Charge"                    AS monthly_charge,
    "Customer Status"                   AS customer_status,
    CASE WHEN "Churn Label" = 'Yes' THEN 1 ELSE 0 END AS churn
FROM read_csv('{{raw_csv}}', header = true, nullstr = '');
