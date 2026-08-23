"""Streamlit demo for loan default prediction.

Run with ``dsj serve loan_default``. The threshold slider is the point of the
app: the model outputs a probability, and where you cut it is a business
decision the metrics table makes visible rather than hides.
"""

from __future__ import annotations

import streamlit as st

from dsjourney.artifacts import load_bundle
from projects.loan_default import pipeline

PURPOSES = [
    "Debt Consolidation",
    "Home Improvements",
    "Business Loan",
    "Buy a Car",
    "Buy House",
    "Medical Bills",
    "Take a Trip",
    "Educational Expenses",
    "Other",
]
OWNERSHIP = ["Rent", "Home Mortgage", "Own Home"]

st.set_page_config(page_title="Loan Default Prediction", page_icon="=", layout="centered")


@st.cache_resource
def _load():
    """Load the trained bundle once per session."""
    return load_bundle("loan_default")


def main() -> None:
    st.title("Loan Default Prediction")
    st.caption("Random forest on 240k consumer loan applications. Positive class = charged off.")

    try:
        bundle = _load()
    except FileNotFoundError:
        st.error("No trained model found. Run `uv run dsj train loan_default` first.")
        return

    with st.sidebar:
        st.subheader("Model")
        st.metric("Recall", f"{bundle.metrics.get('recall', 0):.3f}")
        st.metric("ROC AUC", f"{bundle.metrics.get('roc_auc', 0):.3f}")
        st.metric("Precision", f"{bundle.metrics.get('precision', 0):.3f}")
        st.caption(
            "Accuracy is deliberately not the headline: predicting 'never defaults' "
            "for everyone scores 69% and is worth nothing."
        )
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)

    left, right = st.columns(2)
    with left:
        loan_amount = st.number_input("Loan amount", 1_000, 100_000, 15_000, step=1_000)
        annual_income = st.number_input("Annual income", 10_000, 500_000, 60_000, step=5_000)
        monthly_debt = st.number_input("Monthly debt", 0, 20_000, 800, step=100)
        credit_score = st.slider("Credit score", 300, 850, 700)
        long_term = st.checkbox("Long term (60 months)")
    with right:
        years_in_job = st.slider("Years in current job", 0, 10, 5)
        credit_history_years = st.slider("Years of credit history", 0, 60, 15)
        open_accounts = st.slider("Open accounts", 0, 50, 10)
        credit_balance = st.number_input("Current credit balance", 0, 500_000, 10_000, step=1_000)
        max_open_credit = st.number_input("Maximum open credit", 0, 1_000_000, 25_000, step=1_000)

    third_left, third_right = st.columns(2)
    home_ownership = third_left.selectbox("Home ownership", OWNERSHIP)
    purpose = third_right.selectbox("Purpose", PURPOSES)

    delinquent = st.checkbox("Has a past delinquency")
    months_since_delinquent = (
        st.slider("Months since last delinquency", 0, 120, 24) if delinquent else -1
    )

    if st.button("Assess application", type="primary", use_container_width=True):
        payload = {
            "loan_amount": loan_amount,
            "long_term": long_term,
            "credit_score": credit_score,
            "years_in_job": years_in_job,
            "annual_income": annual_income,
            "monthly_debt": monthly_debt,
            "credit_history_years": credit_history_years,
            "months_since_delinquent": months_since_delinquent,
            "open_accounts": open_accounts,
            "credit_problems": 0,
            "credit_balance": credit_balance,
            "max_open_credit": max_open_credit,
            "bankruptcies": 0,
            "tax_liens": 0,
            "home_ownership": home_ownership,
            "purpose": purpose,
        }
        row = bundle.prepare(pipeline.prepare_input(payload))
        probability = float(bundle.model.predict_proba(row)[0][1])

        st.metric("Probability of default", f"{probability:.1%}")
        if probability >= threshold:
            st.error(f"Flagged for review at a {threshold:.0%} threshold.")
        else:
            st.success(f"Below the {threshold:.0%} threshold.")
        st.progress(min(probability, 1.0))


main()
