# American Express Data Analytics – Spending Patterns, Segments & Churn Risk
**Tech:** Python · Pandas · Scikit-learn · SQL · Power BI

5,000 card members ke ~5.4 lakh transactions (Jan 2025 – Jun 2026) analyse karke:
1. **Spending patterns** nikale (monthly trend, category, card tier, city, seasonality) – SQL + Python
2. **Customer segments** banaye – RFM (Recency, Frequency, Monetary) scoring
3. **Churn risk predict** kiya – Logistic Regression / Random Forest / Gradient Boosting
4. **Stakeholder dashboard + targeted marketing recommendations** – Power BI ke liye scored file

> Data note: Real Amex customer data public nahi hai, isliye `src/generate_data.py` realistic simulated data banata hai (card tiers, Indian cities, categories, churn behaviour). Code Kaggle ke kisi bhi credit card transactions dataset pe bhi chal jayega — bas column names match karne hain.

## Run
```bash
pip install -r requirements.txt
python src/generate_data.py    # data/customers.csv, data/transactions.csv
python src/analysis.py         # SQL + EDA + RFM + churn model + exports  (~1 min)
```

## Pipeline
```
customers.csv + transactions.csv
   │
   ├─ Data quality check (nulls, duplicates, negative amounts)
   ├─ SQLite me load → 7 SQL queries (CTE, JOIN, RANK, ROW_NUMBER, LAG)    → outputs/sql_*.csv
   ├─ EDA charts (trend, category, tier, seasonality heatmap)              → outputs/charts/01-04
   ├─ RFM segmentation (quintile scores → 6 segments)                     → outputs/rfm_segment_summary.csv
   ├─ Churn model
   │     observation window: Jan 2025 – Mar 2026  (features)
   │     label window:       Apr – Jun 2026       (koi txn nahi = churn)
   │     features: recency, frequency, spend, 3m-vs-3m spend trend, active months,
   │               category shares, utilisation, tenure, support calls, autopay, card tier
   │     3 models, 5-fold stratified CV, ROC-AUC, threshold tuning (max F1)  → charts 06-07
   └─ Risk band + recommended marketing action per customer               → customer_scores_for_powerbi.csv
```

## Results (is run ke)
| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | 0.907 | 0.884 |
| Random Forest | 0.904 | 0.891 |
| Gradient Boosting | 0.906 | 0.890 |

- Churn rate ~10.8%. Best (CV) model Logistic Regression: churn class recall **66%**, precision **62%** at threshold 0.70.
- Top churn drivers: **recency**, **spend trend (last 3m vs prev 3m)**, **active months**, recent spend, tenure.
- Champions (20% customers) = **~43% of spend**.
- 772 high-risk customers → ~₹20 crore historical spend at risk → retention / win-back offers.

## Files
| Path | Kya hai |
|---|---|
| `src/generate_data.py` | Data generation |
| `src/analysis.py` | Poora analysis + model |
| `sql/analysis_queries.sql` | SQL queries |
| `outputs/charts/` | 7 charts |
| `outputs/customer_scores_for_powerbi.csv` | Power BI dashboard input |
| `POWER_BI_GUIDE.md` | Dashboard banane ke steps |
