-- ================= American Express Data Analytics: SQL analysis =================
-- Tables: customers(customer_id, age, gender, city, card_type, annual_income_lakh, tenure_months,
--                   credit_limit, num_support_calls, has_autopay)
--         transactions(txn_id, customer_id, txn_date, category, amount, channel, is_international)

-- Q1. Monthly spend volume aur active customers
SELECT substr(txn_date, 1, 7)          AS month,
       COUNT(*)                        AS txn_count,
       COUNT(DISTINCT customer_id)     AS active_customers,
       ROUND(SUM(amount) / 1e7, 2)     AS spend_crore
FROM transactions
GROUP BY month ORDER BY month;

-- Q2. Category-wise spend share aur average ticket size
SELECT category,
       COUNT(*)                                                  AS txns,
       ROUND(SUM(amount) / 1e7, 2)                               AS spend_crore,
       ROUND(AVG(amount), 0)                                     AS avg_ticket,
       ROUND(100.0 * SUM(amount) / SUM(SUM(amount)) OVER (), 1)  AS spend_share_pct
FROM transactions
GROUP BY category ORDER BY spend_crore DESC;

-- Q3. Card tier ke hisaab se per-customer spend (JOIN + aggregation)
SELECT c.card_type,
       COUNT(DISTINCT c.customer_id)                              AS customers,
       ROUND(SUM(t.amount) / COUNT(DISTINCT c.customer_id), 0)    AS spend_per_customer,
       ROUND(AVG(t.amount), 0)                                    AS avg_ticket,
       ROUND(100.0 * AVG(t.is_international), 1)                  AS intl_txn_pct
FROM customers c JOIN transactions t ON t.customer_id = c.customer_id
GROUP BY c.card_type ORDER BY spend_per_customer DESC;

-- Q4. Top 10 customers by spend + unke andar rank (window function)
SELECT * FROM (
    SELECT c.customer_id, c.card_type, c.city,
           ROUND(SUM(t.amount), 0) AS total_spend,
           RANK() OVER (ORDER BY SUM(t.amount) DESC) AS spend_rank
    FROM customers c JOIN transactions t USING (customer_id)
    GROUP BY c.customer_id
) WHERE spend_rank <= 10;

-- Q5. Har customer ki top category (ROW_NUMBER with PARTITION BY)
WITH cat_spend AS (
    SELECT customer_id, category, SUM(amount) AS spend,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY SUM(amount) DESC) AS rn
    FROM transactions GROUP BY customer_id, category
)
SELECT category AS top_category, COUNT(*) AS customers
FROM cat_spend WHERE rn = 1
GROUP BY category ORDER BY customers DESC;

-- Q6. Inactive customers (last 90 days me koi transaction nahi) = churn risk
SELECT COUNT(*) AS inactive_customers,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 1) AS pct
FROM (
    SELECT customer_id, MAX(txn_date) AS last_txn
    FROM transactions GROUP BY customer_id
) WHERE julianday((SELECT MAX(txn_date) FROM transactions)) - julianday(last_txn) > 90;

-- Q7. City-wise spend aur MoM growth (LAG)
WITH city_month AS (
    SELECT c.city, substr(t.txn_date, 1, 7) AS month, SUM(t.amount) AS spend
    FROM transactions t JOIN customers c USING (customer_id)
    GROUP BY c.city, month
)
SELECT city, month, ROUND(spend / 1e5, 1) AS spend_lakh,
       ROUND(100.0 * (spend - LAG(spend) OVER (PARTITION BY city ORDER BY month))
             / LAG(spend) OVER (PARTITION BY city ORDER BY month), 1) AS mom_growth_pct
FROM city_month
WHERE month >= '2026-01'
ORDER BY city, month;
