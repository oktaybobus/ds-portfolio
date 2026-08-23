"""Tests for the projects whose datasets are fetched rather than committed.

Marked ``needs_data`` so ``make test-fast`` and CI skip them; run them locally
after ``python scripts/fetch_assets.py --all``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney.datasets import DatasetNotFoundError
from projects.customer_segments import pipeline as segments
from projects.loan_default import pipeline as loans
from projects.review_sentiment import pipeline as sentiment

pytestmark = pytest.mark.needs_data


def _load_or_skip(module: object) -> pd.DataFrame:
    """Load a project's raw data, skipping the test when it has not been fetched."""
    try:
        return module.load_raw()  # type: ignore[attr-defined]
    except DatasetNotFoundError as error:
        pytest.skip(str(error))


class TestLoanDefault:
    @pytest.fixture(scope="class")
    def features(self) -> pd.DataFrame:
        return loans.build_features(_load_or_skip(loans))

    def test_target_is_binary_and_named_for_default(self, features: pd.DataFrame) -> None:
        assert sorted(features["charged_off"].unique()) == [0, 1]

    def test_target_dtype_is_numeric(self, features: pd.DataFrame) -> None:
        """Object-dtype labels make scikit-learn report 'unknown label type'."""
        assert pd.api.types.is_numeric_dtype(features["charged_off"])

    def test_default_rate_is_measured_after_deduplication(self, features: pd.DataFrame) -> None:
        """26.7%, not the 31.4% the raw file shows - see the duplicate test below."""
        assert features["charged_off"].mean() == pytest.approx(0.267, abs=0.01)

    def test_every_duplicate_row_is_a_default(self) -> None:
        """A data-quality fact worth locking down.

        All 16,611 duplicated rows are charged-off loans, so the raw file
        over-states the default rate by 4.7 points. Any change to the
        de-duplication step should have to confront this deliberately.
        """
        raw = _load_or_skip(loans)
        deduplicated = raw.drop_duplicates()
        removed_rows = len(raw) - len(deduplicated)
        is_default = raw["loan_status"] == "Charged Off"
        removed_defaults = int(
            is_default.sum() - (deduplicated["loan_status"] == "Charged Off").sum()
        )

        assert removed_rows == 16_611
        assert removed_defaults == removed_rows

    def test_credit_scores_are_on_one_scale(self, features: pd.DataFrame) -> None:
        assert features["credit_score"].max() <= 850

    def test_missingness_is_kept_as_features(self, features: pd.DataFrame) -> None:
        assert "months_since_delinquent_missing" in features.columns
        assert "credit_score_missing" in features.columns
        assert features["credit_score_missing"].sum() > 0

    def test_no_missing_values_survive(self, features: pd.DataFrame) -> None:
        assert features.isna().sum().sum() == 0

    def test_prepare_input_produces_one_row(self) -> None:
        row = loans.prepare_input({"credit_score": 640, "annual_income": 45_000})
        assert len(row) == 1
        assert row["debt_to_income"].iloc[0] > 0


class TestCustomerSegments:
    @pytest.fixture(scope="class")
    def rfm(self) -> pd.DataFrame:
        return segments.build_rfm(_load_or_skip(segments))

    def test_order_dates_are_parsed_as_seconds(self, rfm: pd.DataFrame) -> None:
        """The notebook's missing unit="s" collapsed every date onto 1970-01-01.

        With correct parsing the customer base spans years, so recency has real
        spread instead of being ~0 for everyone.
        """
        assert rfm["recency_days"].max() > 365
        assert rfm["recency_days"].std() > 100

    def test_one_row_per_customer(self, rfm: pd.DataFrame) -> None:
        assert rfm.index.is_unique
        assert rfm.index.name == "customer_id"

    def test_monetary_values_are_positive(self, rfm: pd.DataFrame) -> None:
        assert (rfm["monetary"] > 0).all()

    def test_log_scaling_compresses_the_tail(self, rfm: pd.DataFrame) -> None:
        scaled = segments.log_scale_rfm(rfm)
        assert scaled["monetary"].skew() < rfm["monetary"].skew()

    def test_describe_segments_summarises_each_cluster(self, rfm: pd.DataFrame) -> None:
        labels = np.resize([0, 1, 2, 3], len(rfm))
        summary = segments.describe_segments(rfm, labels)
        assert len(summary) == 4
        assert summary["customers"].sum() == len(rfm)


class TestReviewSentiment:
    @pytest.fixture(scope="class")
    def features(self) -> pd.DataFrame:
        return sentiment.build_features(_load_or_skip(sentiment))

    def test_neutral_reviews_are_dropped(self, features: pd.DataFrame) -> None:
        raw = _load_or_skip(sentiment)
        assert len(features) < len(raw)
        assert (raw["stars"] == 3).sum() > 0

    def test_labels_are_binary(self, features: pd.DataFrame) -> None:
        assert sorted(features["sentiment"].unique()) == [0, 1]

    def test_text_is_normalised(self, features: pd.DataFrame) -> None:
        sample = features["text"].head(200)
        assert sample.str.islower().all()
        assert not sample.str.contains(r"[!?.,]").any()

    def test_no_empty_documents_survive(self, features: pd.DataFrame) -> None:
        assert (features["text"].str.len() > 0).all()
