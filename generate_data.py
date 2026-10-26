"""
Step 1: American Express style credit-card transaction data generate karna.
(Real Amex data public nahi hota, isliye realistic simulated data banaya hai.
 Same code Kaggle ke 'Credit Card Transactions' / 'Amex Default Prediction' dataset pe bhi chalega.)

Output:
  data/customers.csv     -> 5,000 card members (demographics + card type)
  data/transactions.csv  -> ~3 lakh transactions (18 months)
"""
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
DATA = Path(__file__).resolve().parents[1] / "data"
N_CUSTOMERS = 5000
START, END = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-06-30")

CATEGORIES = ["Travel", "Dining", "Groceries", "Shopping", "Fuel", "Entertainment", "Utilities", "Electronics"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]
CARDS = {"Green": 0.45, "Gold": 0.35, "Platinum": 0.15, "Centurion": 0.05}
AVG_TICKET = {"Travel": 9000, "Dining": 1800, "Groceries": 2200, "Shopping": 3500,
              "Fuel": 2000, "Entertainment": 1200, "Utilities": 2500, "Electronics": 7000}


def make_customers() -> pd.DataFrame:
    card = RNG.choice(list(CARDS), N_CUSTOMERS, p=list(CARDS.values()))
    income_base = pd.Series(card).map({"Green": 8, "Gold": 14, "Platinum": 25, "Centurion": 60}).values
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
        "age": RNG.integers(22, 66, N_CUSTOMERS),
        "gender": RNG.choice(["M", "F"], N_CUSTOMERS, p=[0.58, 0.42]),
        "city": RNG.choice(CITIES, N_CUSTOMERS),
        "card_type": card,
        "annual_income_lakh": np.round(income_base * RNG.lognormal(0, 0.35, N_CUSTOMERS), 1),
        "tenure_months": RNG.integers(3, 120, N_CUSTOMERS),
        "credit_limit": None,
        "num_support_calls": RNG.poisson(1.2, N_CUSTOMERS),
        "has_autopay": RNG.choice([1, 0], N_CUSTOMERS, p=[0.6, 0.4]),
    })
    df["credit_limit"] = (df["annual_income_lakh"] * 100000 * RNG.uniform(0.15, 0.35, N_CUSTOMERS)).round(-3)
    # Hidden churn propensity (model ko ye directly nahi milega, sirf behaviour se seekhna hoga)
    logit = (-1.6 + 0.35 * df["num_support_calls"] - 0.012 * df["tenure_months"]
             - 0.8 * df["has_autopay"] + 0.5 * (df["card_type"] == "Green") - 0.4 * (df["card_type"] == "Platinum"))
    df["churn_prob_hidden"] = 1 / (1 + np.exp(-logit))
    df["will_churn_hidden"] = RNG.random(N_CUSTOMERS) < df["churn_prob_hidden"]
    return df


def make_transactions(cust: pd.DataFrame) -> pd.DataFrame:
    total_days = (END - START).days
    frames = []
    for c in cust.itertuples():
        # har customer ka apna category preference
        pref = RNG.dirichlet(np.ones(len(CATEGORIES)) * 0.6)
        # kuch customers naturally kam card use karte hain (1-2 txn/month) -> real life noise
        monthly_txns = max(0.7, RNG.lognormal(np.log(6 if c.card_type != "Green" else 4), 0.7))
        n = max(1, RNG.poisson(monthly_txns * 18))
        days = np.sort(RNG.integers(0, total_days, n))
        if RNG.random() < 0.25:
            # non-churners me bhi "slow phase" hota hai (vacation, job change) -> model ke liye confusion
            s = RNG.integers(0, total_days - 90)
            days = days[~((days > s) & (days < s + 90) & (RNG.random(len(days)) < 0.7))]
        if c.will_churn_hidden:
            # churner: stop se ~4 mahine pehle se activity dheere girti hai, phir band.
            # Stop day mostly observation window ke END ke aas paas / baad -> model ko early signal se pakadna hai
            stop_day = RNG.integers(int(total_days * 0.74), int(total_days * 0.86))
            fade_start = stop_day - 120
            fade = np.clip((days - fade_start) / 120, 0, 1)
            keep = (days < stop_day) & (RNG.random(len(days)) > fade * RNG.uniform(0.2, 0.7))
            days = days[keep]
        n = len(days)
        if n == 0:
            continue
        cats = RNG.choice(CATEGORIES, n, p=pref)
        tier = {"Green": 0.8, "Gold": 1.0, "Platinum": 1.6, "Centurion": 3.0}[c.card_type]
        amounts = np.array([AVG_TICKET[k] for k in cats]) * tier * RNG.lognormal(-0.2, 0.6, n)
        frames.append(pd.DataFrame({
            "customer_id": c.customer_id,
            "txn_date": START + pd.to_timedelta(days, unit="D"),
            "category": cats,
            "amount": amounts.round(2),
            "channel": RNG.choice(["Online", "POS", "Contactless"], n, p=[0.45, 0.35, 0.20]),
            "is_international": (RNG.random(n) < np.where(cats == "Travel", 0.35, 0.03)).astype(int),
        }))
    tx = pd.concat(frames, ignore_index=True).sort_values("txn_date")
    tx.insert(0, "txn_id", [f"T{i:07d}" for i in range(1, len(tx) + 1)])
    return tx


def main():
    DATA.mkdir(exist_ok=True)
    cust = make_customers()
    tx = make_transactions(cust)
    cust.drop(columns=["churn_prob_hidden", "will_churn_hidden"]).to_csv(DATA / "customers.csv", index=False)
    tx.to_csv(DATA / "transactions.csv", index=False)
    print(f"customers: {len(cust):,}  transactions: {len(tx):,}")


if __name__ == "__main__":
    main()
