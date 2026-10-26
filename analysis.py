"""
Step 2-5: SQL analysis -> EDA charts -> RFM segmentation -> Churn prediction model -> Power BI exports

Run:  python src/generate_data.py   (pehle data)
      python src/analysis.py
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, CHARTS = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")

OBS_END = pd.Timestamp("2026-03-31")   # features is date tak ke data se
LABEL_END = pd.Timestamp("2026-06-30") # churn label: Apr-Jun 2026 me koi txn nahi = churn


def log(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ------------------------------------------------------------------ 1. Load + SQL
def load_and_sql():
    cust = pd.read_csv(DATA / "customers.csv")
    tx = pd.read_csv(DATA / "transactions.csv", parse_dates=["txn_date"])
    log("DATA QUALITY CHECK")
    print(f"customers {cust.shape}, transactions {tx.shape}")
    print("nulls:", int(tx.isna().sum().sum() + cust.isna().sum().sum()),
          "| duplicate txn_id:", int(tx.txn_id.duplicated().sum()),
          "| negative amounts:", int((tx.amount <= 0).sum()))

    conn = sqlite3.connect(":memory:")
    cust.to_sql("customers", conn, index=False)
    tx.assign(txn_date=tx.txn_date.dt.strftime("%Y-%m-%d")).to_sql("transactions", conn, index=False)
    queries = [q.strip() for q in (ROOT / "sql" / "analysis_queries.sql").read_text().split(";") if "SELECT" in q]
    names = ["Q1_monthly", "Q2_category", "Q3_card_tier", "Q4_top10", "Q5_top_category", "Q6_inactive", "Q7_city_mom"]
    results = {}
    for name, q in zip(names, queries):
        results[name] = pd.read_sql(q, conn)
        results[name].to_csv(OUT / f"sql_{name}.csv", index=False)
    log("SQL: CARD TIER SUMMARY"); print(results["Q3_card_tier"].to_string(index=False))
    log("SQL: CATEGORY SUMMARY"); print(results["Q2_category"].to_string(index=False))
    return cust, tx, results


# ------------------------------------------------------------------ 2. EDA
def eda(cust, tx, res):
    m = res["Q1_monthly"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(m.month, m.spend_crore, marker="o")
    ax.set_title("Monthly card spend (₹ crore)"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(CHARTS / "01_monthly_spend.png", dpi=120); plt.close(fig)

    c = res["Q2_category"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=c, y="category", x="spend_crore", ax=ax, color="#2f6fdb")
    ax.set_title("Spend by category (₹ crore)")
    fig.tight_layout(); fig.savefig(CHARTS / "02_category_spend.png", dpi=120); plt.close(fig)

    t = res["Q3_card_tier"]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=t, x="card_type", y="spend_per_customer", ax=ax, color="#1f8a5b")
    ax.set_title("Spend per customer by card tier (₹)")
    fig.tight_layout(); fig.savefig(CHARTS / "03_card_tier.png", dpi=120); plt.close(fig)

    piv = tx.assign(month=tx.txn_date.dt.month).pivot_table(index="category", columns="month",
                                                             values="amount", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(piv.div(piv.sum(axis=1), axis=0), cmap="Blues", ax=ax)
    ax.set_title("Seasonality: share of each category's spend by calendar month")
    fig.tight_layout(); fig.savefig(CHARTS / "04_seasonality_heatmap.png", dpi=120); plt.close(fig)


# ------------------------------------------------------------------ 3. RFM segmentation
def rfm_segments(cust, tx):
    obs = tx[tx.txn_date <= OBS_END]
    rfm = obs.groupby("customer_id").agg(
        recency=("txn_date", lambda d: (OBS_END - d.max()).days),
        frequency=("txn_id", "count"),
        monetary=("amount", "sum")).reset_index()
    # Quintile scores 1-5 (recency me kam = better, isliye labels ulte)
    rfm["R"] = pd.qcut(rfm.recency.rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def label(r):
        if r.R >= 4 and r.F >= 4 and r.M >= 4: return "Champions"
        if r.R >= 3 and r.F >= 3:             return "Loyal"
        if r.R >= 4 and r.F <= 2:             return "New / Promising"
        if r.R <= 2 and r.F >= 3:             return "At Risk"
        if r.R <= 2 and r.F <= 2:             return "Hibernating"
        return "Need Attention"
    rfm["segment"] = rfm.apply(label, axis=1)
    summary = rfm.groupby("segment").agg(customers=("customer_id", "count"),
                                         avg_recency=("recency", "mean"),
                                         avg_frequency=("frequency", "mean"),
                                         avg_spend=("monetary", "mean")).round(0)
    summary["spend_share_pct"] = (100 * rfm.groupby("segment").monetary.sum() / rfm.monetary.sum()).round(1)
    summary = summary.sort_values("avg_spend", ascending=False)
    log("RFM SEGMENTS"); print(summary.to_string())
    summary.to_csv(OUT / "rfm_segment_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=summary.index, y=summary.customers, ax=ax, color="#7b5ea7")
    ax.set_title("Customers per RFM segment"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(CHARTS / "05_rfm_segments.png", dpi=120); plt.close(fig)
    return rfm


# ------------------------------------------------------------------ 4. Churn model
def build_features(cust, tx):
    obs = tx[tx.txn_date <= OBS_END]
    label_win = tx[(tx.txn_date > OBS_END) & (tx.txn_date <= LABEL_END)]
    last3 = obs[obs.txn_date > OBS_END - pd.DateOffset(months=3)]
    prev3 = obs[(obs.txn_date <= OBS_END - pd.DateOffset(months=3)) &
                (obs.txn_date > OBS_END - pd.DateOffset(months=6))]

    f = obs.groupby("customer_id").agg(
        recency_days=("txn_date", lambda d: (OBS_END - d.max()).days),
        txn_count=("txn_id", "count"),
        total_spend=("amount", "sum"),
        avg_ticket=("amount", "mean"),
        active_months=("txn_date", lambda d: d.dt.to_period("M").nunique()),
        intl_share=("is_international", "mean"),
        online_share=("channel", lambda s: (s == "Online").mean()),
        n_categories=("category", "nunique"))
    f["spend_last3m"] = last3.groupby("customer_id").amount.sum()
    f["spend_prev3m"] = prev3.groupby("customer_id").amount.sum()
    f = f.fillna({"spend_last3m": 0, "spend_prev3m": 0})
    # Trend feature: last 3 months vs pichle 3 months (girta hua spend = churn signal)
    f["spend_trend"] = (f.spend_last3m + 1) / (f.spend_prev3m + 1)
    cat_share = obs.pivot_table(index="customer_id", columns="category", values="amount",
                                aggfunc="sum", fill_value=0)
    cat_share = cat_share.div(cat_share.sum(axis=1), axis=0).add_prefix("share_")
    f = f.join(cat_share).reset_index().merge(cust, on="customer_id")
    f["utilisation"] = (f.total_spend / 15) / f.credit_limit   # ~15 months observation
    f["churned"] = (~f.customer_id.isin(label_win.customer_id)).astype(int)
    return f


def churn_model(cust, tx):
    df = build_features(cust, tx)
    log("CHURN MODEL")
    print(f"customers in model: {len(df):,} | churn rate: {df.churned.mean():.1%}")

    cat_cols = ["card_type", "city", "gender"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["customer_id", "churned"]]
    X, y = df[cat_cols + num_cols], df.churned
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                        random_state=42),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {}
    for name, mdl in models.items():
        pipe = Pipeline([("pre", pre), ("clf", mdl)])
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_tr, y_tr)
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        scores[name] = (cv_auc, test_auc, pipe)
        print(f"{name:22s} CV ROC-AUC {cv_auc:.3f} | Test ROC-AUC {test_auc:.3f}")
    pd.DataFrame({k: v[:2] for k, v in scores.items()}, index=["cv_auc", "test_auc"]).T.round(3) \
        .to_csv(OUT / "model_comparison.csv")

    best_name = max(scores, key=lambda k: scores[k][0])
    best = scores[best_name][2]
    proba = best.predict_proba(X_te)[:, 1]
    # Threshold: business ko recall chahiye (zyada churners pakadne hain), isliye 0.5 ki jagah tune karte hain
    prec, rec, thr = precision_recall_curve(y_te, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[np.argmax(f1[:-1])])
    pred = (proba >= threshold).astype(int)
    print(f"\nBest model: {best_name} | threshold (max F1): {threshold:.2f}")
    print(classification_report(y_te, pred, digits=3))
    cm = confusion_matrix(y_te, pred)
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)

    # Charts: ROC + confusion matrix + feature importance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, (_, _, p) in scores.items():
        fpr, tpr, _ = roc_curve(y_te, p.predict_proba(X_te)[:, 1])
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_te, p.predict_proba(X_te)[:, 1]):.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].legend(); axes[0].set_title("ROC curve (test)")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual"); axes[1].set_title(f"Confusion matrix – {best_name}")
    fig.tight_layout(); fig.savefig(CHARTS / "06_model_performance.png", dpi=120); plt.close(fig)

    rf = scores["Random Forest"][2]
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(12)
    imp.index = [i.split("__")[1] for i in imp.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#d1495b")
    ax.set_title("Top churn drivers (Random Forest feature importance)")
    fig.tight_layout(); fig.savefig(CHARTS / "07_feature_importance.png", dpi=120); plt.close(fig)
    log("TOP CHURN DRIVERS"); print(imp.round(3).to_string())

    # Saare customers ko score karo -> risk band -> marketing action
    df["churn_probability"] = best.predict_proba(X)[:, 1].round(3)
    df["risk_band"] = pd.cut(df.churn_probability, [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"],
                             include_lowest=True)
    return df, best_name, threshold


# ------------------------------------------------------------------ 5. Recommendations + exports
def recommendations(scored, rfm):
    out = scored.merge(rfm[["customer_id", "R", "F", "M", "segment"]], on="customer_id", how="left")
    top_cat = out.filter(like="share_").idxmax(axis=1).str.replace("share_", "")
    out["top_category"] = top_cat

    def action(r):
        if r.risk_band == "High" and r.segment in ("Champions", "Loyal", "At Risk"):
            return "Retention call + fee waiver / bonus Membership Rewards"
        if r.risk_band == "High":
            return f"Win-back offer: 10% cashback on {r.top_category}"
        if r.segment == "Champions":
            return "Upgrade offer (Platinum) + travel partner perks"
        if r.risk_band == "Medium":
            return f"Engagement nudge: bonus points on {r.top_category}"
        return f"Cross-sell partner offer on {r.top_category}"
    out["recommended_action"] = out.apply(action, axis=1)

    cols = ["customer_id", "card_type", "city", "age", "tenure_months", "total_spend", "txn_count",
            "recency_days", "segment", "top_category", "churn_probability", "risk_band",
            "churned", "recommended_action"]
    out[cols].to_csv(OUT / "customer_scores_for_powerbi.csv", index=False)

    at_risk_value = out.loc[out.risk_band == "High", "total_spend"].sum()
    log("BUSINESS SUMMARY")
    print(out.risk_band.value_counts().to_string())
    print(f"High-risk customers ka 15-month spend: ₹{at_risk_value / 1e7:.1f} crore (revenue at risk)")
    print(out.groupby("recommended_action").customer_id.count().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    cust, tx, res = load_and_sql()
    eda(cust, tx, res)
    rfm = rfm_segments(cust, tx)
    scored, best, thr = churn_model(cust, tx)
    recommendations(scored, rfm)
    print("\nAll outputs saved in outputs/")
