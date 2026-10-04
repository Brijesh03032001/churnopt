-- Question: Which retention offers work?
-- Why it matters: The naive answer is wrong. Each offer goes almost entirely to
-- ONE tenure band, so comparing offers raw compares customer age, not offers.
-- Step 1 shows the raw churn rate. Step 2 compares each offer with customers who
-- got NO offer in the SAME tenure band. Where no like-for-like group exists the
-- comparison is blank, which is the honest answer: the data cannot say.
WITH banded AS (
    SELECT
        offer,
        CASE
            WHEN tenure_months <= 6  THEN '1: 0-6 months'
            WHEN tenure_months <= 12 THEN '2: 7-12 months'
            WHEN tenure_months <= 24 THEN '3: 13-24 months'
            WHEN tenure_months <= 48 THEN '4: 25-48 months'
            ELSE                          '5: 49+ months'
        END AS tenure_band,
        churn
    FROM customers
),
rates AS (
    SELECT offer, tenure_band, COUNT(*) AS customers, AVG(churn) AS churn_rate
    FROM banded
    GROUP BY ALL
),
no_offer AS (
    SELECT tenure_band, customers AS no_offer_customers, churn_rate AS no_offer_churn
    FROM rates
    WHERE offer = 'None'
)
SELECT
    r.offer,
    r.tenure_band,
    r.customers,
    ROUND(100 * r.churn_rate, 1)                              AS churn_pct,
    ROUND(100 * SUM(r.customers * r.churn_rate) OVER (PARTITION BY r.offer)
                / SUM(r.customers) OVER (PARTITION BY r.offer), 1) AS offer_raw_churn_pct,
    n.no_offer_customers,
    ROUND(100 * n.no_offer_churn, 1)                          AS no_offer_churn_pct,
    ROUND(100 * (r.churn_rate - n.no_offer_churn), 1)         AS gap_vs_no_offer_points
FROM rates r
LEFT JOIN no_offer n USING (tenure_band)
WHERE r.offer <> 'None'
ORDER BY r.offer, r.tenure_band;
