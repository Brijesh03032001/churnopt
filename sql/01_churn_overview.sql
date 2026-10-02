-- Question: How big is the churn problem in customers and in recurring revenue?
-- Why it matters: Revenue lost is the number finance cares about; the customer
-- count alone understates it when churners pay more than the average customer.
SELECT
    COUNT(*)                                                        AS customers,
    CAST(SUM(churn) AS INTEGER)                                     AS churned_customers,
    ROUND(100.0 * AVG(churn), 1)                                    AS churn_rate_pct,
    ROUND(SUM(monthly_charge), 0)                                   AS monthly_recurring_revenue,
    ROUND(SUM(monthly_charge) FILTER (WHERE churn = 1), 0)          AS monthly_revenue_lost,
    ROUND(100.0 * SUM(monthly_charge) FILTER (WHERE churn = 1)
                / SUM(monthly_charge), 1)                           AS revenue_lost_pct,
    ROUND(AVG(monthly_charge) FILTER (WHERE churn = 1), 2)          AS avg_charge_churners,
    ROUND(AVG(monthly_charge) FILTER (WHERE churn = 0), 2)          AS avg_charge_stayers
FROM customers;
