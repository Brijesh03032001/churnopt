-- Question: Which segments account for most of the revenue lost to churn?
-- Why it matters: A Pareto view tells the retention team where to focus first.
-- Segment = contract x internet type. Cumulative share uses a running window sum.
WITH segments AS (
    SELECT
        contract,
        internet_type,
        COUNT(*)                                                AS customers,
        ROUND(100.0 * AVG(churn), 1)                            AS churn_rate_pct,
        ROUND(SUM(monthly_charge) FILTER (WHERE churn = 1), 0)  AS monthly_revenue_lost
    FROM customers
    GROUP BY contract, internet_type
)
SELECT
    contract,
    internet_type,
    customers,
    churn_rate_pct,
    monthly_revenue_lost,
    ROUND(100.0 * monthly_revenue_lost / SUM(monthly_revenue_lost) OVER (), 1) AS share_of_loss_pct,
    ROUND(100.0 * SUM(monthly_revenue_lost) OVER (ORDER BY monthly_revenue_lost DESC)
                / SUM(monthly_revenue_lost) OVER (), 1)                        AS cumulative_share_pct
FROM segments
WHERE monthly_revenue_lost > 0
ORDER BY monthly_revenue_lost DESC;
