-- Question: Do customers with tech support and online security churn less?
-- Why it matters: These add-ons are a cheap, actionable retention lever, if the
-- gap survives comparing like with like. The comparison is made WITHIN internet
-- type and contract so the add-ons are not just standing in for plan differences.
WITH internet_customers AS (
    SELECT
        internet_type,
        contract,
        CASE WHEN tech_support = 'Yes' OR online_security = 'Yes'
             THEN 'has support or security' ELSE 'neither' END  AS protection,
        churn
    FROM customers
    WHERE internet_type <> 'None'
),
rates AS (
    SELECT internet_type, contract, protection,
           COUNT(*) AS customers, AVG(churn) AS churn_rate
    FROM internet_customers
    GROUP BY ALL
)
SELECT
    w.internet_type,
    w.contract,
    w.customers                                       AS customers_with,
    n.customers                                       AS customers_without,
    ROUND(100 * w.churn_rate, 1)                      AS churn_pct_with,
    ROUND(100 * n.churn_rate, 1)                      AS churn_pct_without,
    ROUND(100 * (n.churn_rate - w.churn_rate), 1)     AS gap_points
FROM rates w
JOIN rates n
  ON w.internet_type = n.internet_type AND w.contract = n.contract
WHERE w.protection = 'has support or security' AND n.protection = 'neither'
ORDER BY gap_points DESC;
