"""Streamlit demo for univariate forecasting.

Run with ``dsj serve series_forecast``. Every forecast is drawn against the
held-out actuals, never past the end of a fully-fitted series - which is the
whole point of the project.
"""

from __future__ import annotations

import streamlit as st

from dsjourney import forecasting, viz
from projects.series_forecast import pipeline

st.set_page_config(page_title="Time-Series Forecasting", page_icon="=", layout="centered")


@st.cache_data
def _series(key: str):
    """Load and index one series."""
    spec = pipeline.spec_for(key)
    return pipeline.build_series(pipeline.load_raw(key), spec), spec


def main() -> None:
    st.title("Univariate Time-Series Forecasting")
    st.caption("Chronological holdout, every method scored against a naive baseline.")

    key = st.selectbox(
        "Series",
        sorted(pipeline.SERIES),
        format_func=lambda k: pipeline.SERIES[k].title,
    )

    try:
        values, spec = _series(key)
    except FileNotFoundError as error:
        st.error(str(error))
        return

    horizon = st.slider("Holdout length (periods)", 4, min(len(values) // 4, 180), spec.horizon)
    split = forecasting.chronological_split(values, horizon=horizon)

    strength = forecasting.seasonal_strength(split.train, period=spec.period)
    left, right = st.columns(2)
    left.metric("Trend strength", f"{strength['trend_strength']:.3f}")
    right.metric("Seasonal strength", f"{strength['seasonal_strength']:.3f}")

    table = forecasting.compare_forecasters(
        split,
        params={
            "seasonal_naive": {"period": spec.period},
            "holt_winters": {"period": spec.period},
            "sarima": {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, min(spec.period, 12))},
        },
    )
    st.dataframe(table, use_container_width=True)

    winner = str(table.iloc[0]["method"])
    skill = float(table.iloc[0].get("skill_vs_naive", 0))
    if skill > 0:
        st.success(f"{winner} beats the naive baseline by {skill:.1%}.")
    else:
        st.warning(
            f"Nothing beats the naive baseline on this series - {winner} only matches it. "
            "That is a real result, not a bug."
        )

    forecast = forecasting.build_forecast(
        winner,
        split.train,
        split.horizon,
        **({"period": spec.period} if winner in {"seasonal_naive", "holt_winters"} else {}),
    )
    st.pyplot(
        viz.forecast_plot(split.train, split.test, forecast, title=f"{spec.title} - {winner}")
    )


main()
