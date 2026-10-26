# Power BI Stakeholder Dashboard – steps

**Load karo (Get Data → Text/CSV):**
- `outputs/customer_scores_for_powerbi.csv` (1 row = 1 customer, churn score + segment + action)
- `data/transactions.csv` (transaction level)
- `outputs/rfm_segment_summary.csv`

**Relationship:** `customer_scores[customer_id]` 1 → * `transactions[customer_id]`

**DAX measures:**
```DAX
Total Spend        = SUM(transactions[amount])
Active Customers   = DISTINCTCOUNT(transactions[customer_id])
Avg Ticket         = DIVIDE([Total Spend], COUNTROWS(transactions))
High Risk Count    = CALCULATE(COUNTROWS(customer_scores), customer_scores[risk_band] = "High")
Revenue at Risk    = CALCULATE(SUM(customer_scores[total_spend]), customer_scores[risk_band] = "High")
Avg Churn Prob     = AVERAGE(customer_scores[churn_probability])
Intl Txn %         = DIVIDE(CALCULATE(COUNTROWS(transactions), transactions[is_international] = 1), COUNTROWS(transactions))
```

**Page 1 – Executive overview:** KPI cards (Total Spend, Active Customers, Avg Ticket, Revenue at Risk), monthly spend line, category bar, card tier column chart, slicers (city, card type).

**Page 2 – Customer segments:** RFM segment donut (customers), segment vs spend share bar, scatter (txn_count vs total_spend, legend = segment).

**Page 3 – Churn risk:** risk band funnel/donut, Avg Churn Prob by card_type & city (matrix with conditional formatting), table: top 50 high-risk customers + `recommended_action` (marketing team yahi list export karegi).

**Page 4 – Marketing actions:** bar chart `recommended_action` count, drill-through to customer list.

Save as `Amex_Analytics_Dashboard.pbix`, screenshot README me lagao.
