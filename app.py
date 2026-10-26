"""
American Express Data Analytics — Flask Dashboard (port 5001)
Run:
    python app.py
Then open: http://127.0.0.1:5001
"""
import csv
import json
from pathlib import Path
from flask import Flask, jsonify, render_template, send_from_directory

app = Flask(__name__)

BASE   = Path(__file__).parent
OUT    = BASE / "outputs"
CHARTS = OUT / "charts"


def read_csv(filename):
    path = OUT / filename
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


# ─── Static chart serving ───────────────────────────────────────────────────

@app.route("/charts/<path:filename>")
def serve_chart(filename):
    return send_from_directory(CHARTS, filename)


# ─── API endpoints ───────────────────────────────────────────────────────────

@app.get("/api/monthly")
def api_monthly():
    return jsonify(read_csv("sql_Q1_monthly.csv"))


@app.get("/api/categories")
def api_categories():
    return jsonify(read_csv("sql_Q2_category.csv"))


@app.get("/api/card-tiers")
def api_card_tiers():
    return jsonify(read_csv("sql_Q3_card_tier.csv"))


@app.get("/api/rfm-segments")
def api_rfm():
    return jsonify(read_csv("rfm_segment_summary.csv"))


@app.get("/api/churn-models")
def api_churn_models():
    return jsonify(read_csv("model_comparison.csv"))


@app.get("/api/top-customers")
def api_top_customers():
    return jsonify(read_csv("sql_Q4_top10.csv"))


@app.get("/api/city-mom")
def api_city_mom():
    return jsonify(read_csv("sql_Q7_city_mom.csv"))


@app.get("/api/summary")
def api_summary():
    monthly = read_csv("sql_Q1_monthly.csv")
    total_spend = sum(float(r["spend_crore"]) for r in monthly)
    total_txns  = sum(int(r["txn_count"]) for r in monthly)
    rfm = read_csv("rfm_segment_summary.csv")
    champions = next((r for r in rfm if r["segment"] == "Champions"), {})
    at_risk   = next((r for r in rfm if r["segment"] == "At Risk"), {})
    return jsonify({
        "total_spend_crore": round(total_spend, 1),
        "total_txns":        total_txns,
        "total_customers":   5000,
        "champions_count":   champions.get("customers", 0),
        "at_risk_count":     at_risk.get("customers", 0),
        "churn_rate":        10.8,
        "revenue_at_risk_cr": 20.0,
    })


# ─── Main page ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
