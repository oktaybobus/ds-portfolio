"""Streamlit demo for Istanbul apartment price prediction.

Run with ``dsj serve istanbul_housing``.
"""

from __future__ import annotations

import streamlit as st

from dsjourney.artifacts import load_bundle
from projects.istanbul_housing import pipeline

DISTRICTS = [
    "Kadıköy",
    "Beşiktaş",
    "Şişli",
    "Beyoğlu",
    "Üsküdar",
    "Ataşehir",
    "Maltepe",
    "Kartal",
    "Pendik",
    "Bakırköy",
    "Bahçelievler",
    "Esenyurt",
    "Beylikdüzü",
    "Başakşehir",
    "Sarıyer",
    "Ümraniye",
    "Zeytinburnu",
    "Fatih",
]

st.set_page_config(page_title="Istanbul Apartment Prices", page_icon="=", layout="centered")


@st.cache_resource
def _load():
    """Load the trained bundle once per session."""
    return load_bundle("istanbul_housing")


def main() -> None:
    st.title("Istanbul Apartment Price Prediction")
    st.caption("CatBoost over 10,599 Emlakjet listings. Prices in millions of Turkish lira.")

    try:
        bundle = _load()
    except FileNotFoundError:
        st.error("No trained model found. Run `uv run dsj train istanbul_housing` first.")
        return

    with st.sidebar:
        st.subheader("Model")
        st.metric("R² (price scale)", f"{bundle.metrics.get('r2_original', 0):.3f}")
        st.metric("MAE", f"{bundle.metrics.get('mae_original', 0):.2f} M TL")
        st.caption(
            "The model is trained on log-transformed prices; these figures are "
            "measured after converting predictions back to lira."
        )

    left, right = st.columns(2)
    with left:
        ilce = st.selectbox("District", DISTRICTS)
        brut_m2 = st.slider("Gross area (m²)", 40, 500, 130)
        net_m2 = st.slider("Net area (m²)", 30, 450, 110)
    with right:
        toplam_oda = st.select_slider(
            "Total rooms (bedrooms + living)",
            [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0],
            value=3.5,
        )
        bina_yasi = st.slider("Building age (years)", 0, 40, 8)

    if net_m2 > brut_m2:
        st.warning("Net area is larger than gross area - check the inputs.")

    if st.button("Estimate price", type="primary", use_container_width=True):
        payload = {
            "ilce": ilce,
            "mahalle": "Others",
            "brut_m2": float(brut_m2),
            "net_m2": float(net_m2),
            "toplam_oda": float(toplam_oda),
            "bina_yasi": float(bina_yasi),
        }
        row = bundle.prepare(pipeline.prepare_input(payload))
        price = float(pipeline.postprocess(bundle.model.predict(row))[0])

        st.success(f"Estimated asking price: {price:,.2f} million TL")
        margin = bundle.metrics.get("mae_original", 0)
        if margin:
            st.caption(
                f"Typical error on held-out listings is ±{margin:.2f} million TL, "
                f"so read this as roughly {max(price - margin, 0):,.1f}-{price + margin:,.1f} M TL."
            )
        st.caption(
            "Neighbourhood is left unspecified; a named neighbourhood would narrow the estimate."
        )


main()
