"""Streamlit demo for RFM customer segmentation.

Run with ``dsj serve customer_segments``. Enter a customer's Recency, Frequency
and Monetary values and see which of the four segments they fall into.
"""

from __future__ import annotations

import streamlit as st

from dsjourney.artifacts import load_bundle
from projects.customer_segments import pipeline

st.set_page_config(page_title="Customer Segmentation", page_icon="=", layout="centered")


@st.cache_resource
def _load():
    """Load the trained clustering bundle once per session."""
    return load_bundle("customer_segments")


@st.cache_data
def _segment_profiles():
    """Recompute the segment profile table from the source data."""
    bundle = _load()
    rfm = pipeline.build_rfm(pipeline.load_raw())
    scaled = pipeline.log_scale_rfm(rfm)
    labels = bundle.model.predict(bundle.scaler.transform(scaled))
    return pipeline.describe_segments(rfm, labels)


def main() -> None:
    st.title("RFM Customer Segmentation")
    st.caption("KMeans over log-scaled Recency, Frequency and Monetary value.")

    try:
        bundle = _load()
    except FileNotFoundError:
        st.error("No trained model found. Run `uv run python projects/customer_segments/train.py`.")
        return

    with st.sidebar:
        st.subheader("Clustering quality")
        for name, value in bundle.metrics.items():
            st.metric(name.replace("_", " ").title(), f"{value:.3f}")
        st.caption(
            f"k = {bundle.extra.get('n_clusters', '?')} on {bundle.extra.get('rows', '?')} customers"
        )

    st.subheader("Segment profiles")
    try:
        st.dataframe(_segment_profiles(), use_container_width=True)
    except FileNotFoundError:
        st.info("Fetch the dataset to see the segment profile table.")

    st.subheader("Classify a customer")
    left, middle, right = st.columns(3)
    recency_days = left.number_input("Days since last order", 0, 2_000, 90)
    frequency = middle.number_input("Number of orders", 1, 200, 2)
    monetary = right.number_input("Total spend", 0.0, 100_000.0, 250.0, step=50.0)

    if st.button("Find segment", type="primary", use_container_width=True):
        payload = {"recency_days": recency_days, "frequency": frequency, "monetary": monetary}
        scaled = bundle.scaler.transform(pipeline.prepare_input(payload))
        segment = int(bundle.model.predict(scaled)[0])
        st.success(f"Segment {segment}")
        st.caption(
            "Compare the row for this segment in the table above to see how this "
            "customer sits against the rest of the base."
        )


main()
