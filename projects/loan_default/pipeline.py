"""Feature engineering for consumer loan default prediction.

Two decisions here differ deliberately from the source notebook:

1. **The target is inverted.** The notebook predicted ``Fully Paid`` = 1, which
   makes the majority class the positive one and flatters every metric. Here the
   positive class is ``charged_off``, so recall answers the question a lender
   actually asks: of the loans that went bad, how many did we catch?

2. **Missingness is a feature.** 55% of rows have no delinquency history and 24%
   have no credit score. Those gaps are structural, not random - a thin credit
   file is informative - so each imputed column keeps a ``*_missing`` flag.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsjourney import preprocess
from dsjourney.config import load_project_config
from dsjourney.datasets import load_dataset

CONFIG = load_project_config("loan_default")

COLUMN_RENAMES = {
    "Loan ID": "loan_id",
    "Customer ID": "customer_id",
    "Loan Status": "loan_status",
    "Current Loan Amount": "loan_amount",
    "Term": "long_term",
    "Credit Score": "credit_score",
    "Years in current job": "years_in_job",
    "Home Ownership": "home_ownership",
    "Annual Income": "annual_income",
    "Purpose": "purpose",
    "Monthly Debt": "monthly_debt",
    "Years of Credit History": "credit_history_years",
    "Months since last delinquent": "months_since_delinquent",
    "Number of Open Accounts": "open_accounts",
    "Number of Credit Problems": "credit_problems",
    "Current Credit Balance": "credit_balance",
    "Maximum Open Credit": "max_open_credit",
    "Bankruptcies": "bankruptcies",
    "Tax Liens": "tax_liens",
}

IDENTIFIER_COLUMNS = ["loan_id", "customer_id"]
CATEGORICAL_COLUMNS = ["home_ownership", "purpose"]

# Documented request body for the API and the CLI.
EXAMPLE_INPUT = {
    "loan_amount": 15000,
    "long_term": False,
    "credit_score": 690,
    "years_in_job": 5,
    "annual_income": 60000,
    "monthly_debt": 800,
    "credit_history_years": 15,
    "months_since_delinquent": -1,
    "open_accounts": 10,
    "credit_problems": 0,
    "credit_balance": 10000,
    "max_open_credit": 25000,
    "bankruptcies": 0,
    "tax_liens": 0,
    "home_ownership": "Rent",
    "purpose": "Debt Consolidation",
}
CURRENCY_COLUMNS = ["monthly_debt", "max_open_credit"]

# to_numeric leaves NaN wherever a currency string could not be parsed, so the
# cleaned columns need imputing as well as the natively missing ones.
IMPUTED_COLUMNS = [
    "credit_score",
    "annual_income",
    "years_in_job",
    "bankruptcies",
    "tax_liens",
    "monthly_debt",
    "max_open_credit",
    "credit_utilisation",
    "debt_to_income",
]

VALUE_MAPS = {
    "loan_status": {"Charged Off": 1, "Fully Paid": 0},
    "long_term": {"Long Term": 1, "Short Term": 0},
    "years_in_job": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
    },
    # The source data spells the same category two ways.
    "home_ownership": {"HaveMortgage": "Home Mortgage"},
    "purpose": {"other": "Other"},
}

# Credit scores are recorded on two scales: a 300-850 FICO scale and, for ~16k
# rows, the same number multiplied by ten. Anything above 850 is rescaled rather
# than discarded.
MAX_VALID_CREDIT_SCORE = 850


def load_raw() -> pd.DataFrame:
    """Read the loan dataset and normalise its column names."""
    return load_dataset(CONFIG).rename(columns=COLUMN_RENAMES)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn raw applications into a numeric frame with a ``charged_off`` target."""
    prepared = (
        preprocess.drop_columns(frame, IDENTIFIER_COLUMNS)
        .pipe(preprocess.drop_duplicate_rows)
        .pipe(preprocess.to_numeric, CURRENCY_COLUMNS)
        .pipe(preprocess.map_values, VALUE_MAPS)
        .pipe(_fix_credit_score_scale)
        .pipe(preprocess.safe_ratio, "credit_balance", "max_open_credit", "credit_utilisation")
        .pipe(preprocess.safe_ratio, "monthly_debt", "annual_income", "debt_to_income")
        .pipe(preprocess.flag_and_fill_missing, "months_since_delinquent", fill_value=-1)
        .pipe(preprocess.impute_numeric, IMPUTED_COLUMNS, strategy="median")
    )
    encoded = preprocess.one_hot(prepared, CATEGORICAL_COLUMNS, drop_first=True)
    return encoded.rename(columns={"loan_status": "charged_off"}).dropna(subset=["charged_off"])


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a single model-ready row from an application record."""
    annual_income = float(payload.get("annual_income", 60_000))
    monthly_debt = float(payload.get("monthly_debt", 800))
    credit_balance = float(payload.get("credit_balance", 10_000))
    max_open_credit = float(payload.get("max_open_credit", 25_000)) or float("nan")

    row = pd.DataFrame(
        [
            {
                "loan_amount": float(payload.get("loan_amount", 15_000)),
                "long_term": int(bool(payload.get("long_term", False))),
                "credit_score": float(payload.get("credit_score", 700)),
                "years_in_job": float(payload.get("years_in_job", 5)),
                "annual_income": annual_income,
                "monthly_debt": monthly_debt,
                "credit_history_years": float(payload.get("credit_history_years", 15)),
                "months_since_delinquent": float(payload.get("months_since_delinquent", -1)),
                "months_since_delinquent_missing": int(
                    payload.get("months_since_delinquent", -1) < 0
                ),
                "open_accounts": float(payload.get("open_accounts", 10)),
                "credit_problems": float(payload.get("credit_problems", 0)),
                "credit_balance": credit_balance,
                "max_open_credit": max_open_credit,
                "bankruptcies": float(payload.get("bankruptcies", 0)),
                "tax_liens": float(payload.get("tax_liens", 0)),
                "credit_utilisation": credit_balance / max_open_credit,
                "debt_to_income": monthly_debt / annual_income if annual_income else 0.0,
                "home_ownership": payload.get("home_ownership", "Rent"),
                "purpose": payload.get("purpose", "Debt Consolidation"),
            }
        ]
    )
    return preprocess.one_hot(row, CATEGORICAL_COLUMNS, drop_first=False)


def _fix_credit_score_scale(frame: pd.DataFrame) -> pd.DataFrame:
    """Divide out-of-range credit scores by ten to put them back on the FICO scale."""
    scores = frame["credit_score"]
    corrected = scores.where(scores <= MAX_VALID_CREDIT_SCORE, scores / 10)
    return frame.assign(credit_score=corrected)
