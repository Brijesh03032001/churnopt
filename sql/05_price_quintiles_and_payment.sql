-- Question: Is churn driven by price level, and does payment method matter?
-- Why it matters: Separates "expensive plans churn" from "month-to-month plans
-- churn". Quintiles are computed with NTILE over monthly charge, then split by
-- contract so price is compared inside each contract type.
WITH priced AS (
    SELECT
        contract,
        payment_method,
        monthly_charge,
        churn,
        NTILE(5) OVER (PARTITION BY contract ORDER BY monthly_charge) AS price_quintile
    FROM customers
)
SELECT
    contract,
    price_quintile,
    COUNT(*)                                  AS customers,
    ROUND(MIN(monthly_charge), 2)             AS min_charge,
    ROUND(MAX(monthly_charge), 2)             AS max_charge,
    ROUND(100.0 * AVG(churn), 1)              AS churn_rate_pct,
    ROUND(100.0 * AVG(churn) FILTER (WHERE payment_method = 'Mailed Check'), 1)    AS churn_pct_mailed_check,
    ROUND(100.0 * AVG(churn) FILTER (WHERE payment_method = 'Bank Withdrawal'), 1) AS churn_pct_bank_withdrawal
FROM priced
GROUP BY contract, price_quintile
ORDER BY contract, price_quintile;
