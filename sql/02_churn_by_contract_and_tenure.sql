-- Question: Where does churn concentrate across contract type and tenure?
-- Why it matters: Shows how much of churn sits in the first year on month-to-month
-- plans, which is where a retention program has the most to win.
WITH banded AS (
    SELECT
        contract,
        CASE
            WHEN tenure_months <= 12 THEN '1: 0-12 months'
            WHEN tenure_months <= 24 THEN '2: 13-24 months'
            WHEN tenure_months <= 48 THEN '3: 25-48 months'
            ELSE                          '4: 49+ months'
        END AS tenure_band,
        churn
    FROM customers
)
SELECT
    contract,
    tenure_band,
    COUNT(*)                                                    AS customers,
    CAST(SUM(churn) AS INTEGER)                                 AS churned,
    ROUND(100.0 * AVG(churn), 1)                                AS churn_rate_pct,
    ROUND(100.0 * SUM(churn) / SUM(SUM(churn)) OVER (), 1)      AS share_of_all_churners_pct
FROM banded
GROUP BY contract, tenure_band
ORDER BY share_of_all_churners_pct DESC;
