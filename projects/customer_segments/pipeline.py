"""RFM feature construction for customer segmentation.

The source file is one wide denormalised export: 181 columns prefixed
``Customers.``, ``Orders.``, ``Order_Items.`` and ``Products.``, one row per
order line. This module splits it back into logical tables, aggregates each
customer's Recency, Frequency and Monetary value, and log-transforms them - all
three are heavily right-skewed, and KMeans on raw values would simply isolate
the handful of biggest spenders.

Fixed from the source notebook
------------------------------
``Orders.placed_date`` holds Unix timestamps in seconds. The notebook parsed it
with ``pd.to_datetime(...)`` and no ``unit`` argument, so pandas read the
integers as nanoseconds and every order landed within two seconds of
1970-01-01. Recency was therefore ~0 for every customer and contributed nothing
to the clustering, which was in effect an FM segmentation wearing an RFM label.
Parsing with ``unit="s"`` restores dates spanning 2013-2016.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsjourney.config import load_project_config
from dsjourney.datasets import load_dataset

CONFIG = load_project_config("customer_segments")

CUSTOMER_ID = "Customers.id"
ORDER_ID = "Orders.id"
ORDER_DATE = "Orders.placed_date"
LINE_PRICE = "Order_Items.price"
LINE_ID = "Order_Items.id"

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]

EXAMPLE_INPUT = {"recency_days": 90, "frequency": 2, "monetary": 250.0}

SEGMENT_LABELS = {
    0: "At risk",
    1: "Loyal",
    2: "Champions",
    3: "New / low value",
}


def load_raw() -> pd.DataFrame:
    """Read the denormalised e-commerce export."""
    return load_dataset(CONFIG)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order lines into one log-scaled RFM row per customer."""
    return log_scale_rfm(build_rfm(frame))


def build_rfm(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute raw Recency (days), Frequency (orders) and Monetary (spend) per customer.

    Returns a frame indexed by customer id, in original units - useful for
    profiling segments in the demo app, where log values would be unreadable.
    """
    lines = frame.drop_duplicates(subset=[LINE_ID]) if LINE_ID in frame.columns else frame
    # Project down to the four columns RFM needs: assigning onto the full
    # 181-column export fragments the frame and is an order of magnitude slower.
    needed = lines[[CUSTOMER_ID, ORDER_ID, ORDER_DATE, LINE_PRICE]].copy()
    dated = needed.assign(order_date=pd.to_datetime(needed[ORDER_DATE], unit="s", errors="coerce"))
    dated = dated.dropna(subset=["order_date", CUSTOMER_ID])

    snapshot = dated["order_date"].max() + pd.Timedelta(days=1)
    rfm = dated.groupby(CUSTOMER_ID).agg(
        recency_days=("order_date", lambda dates: (snapshot - dates.max()).days),
        frequency=(ORDER_ID, "nunique"),
        monetary=(LINE_PRICE, "sum"),
    )
    rfm.index.name = "customer_id"
    return rfm[rfm["monetary"] > 0]


def log_scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Apply ``log1p`` to the three RFM columns to tame their right skew."""
    return pd.DataFrame(np.log1p(rfm[RFM_COLUMNS].to_numpy()), columns=RFM_COLUMNS, index=rfm.index)


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a single log-scaled RFM row for one customer profile."""
    row = pd.DataFrame(
        [
            {
                "recency_days": float(payload.get("recency_days", 90)),
                "frequency": float(payload.get("frequency", 2)),
                "monetary": float(payload.get("monetary", 250)),
            }
        ]
    )
    return log_scale_rfm(row)


def describe_segments(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Summarise each cluster in original units so the segments can be named."""
    profiled = rfm.assign(segment=labels)
    summary = profiled.groupby("segment").agg(
        customers=("monetary", "size"),
        median_recency_days=("recency_days", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
        total_monetary=("monetary", "sum"),
    )
    return summary.sort_values("median_monetary", ascending=False)
