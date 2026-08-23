"""Streamlit demo for laptop price prediction.

Run with ``dsj serve laptop_price``. The app loads the saved bundle rather than
retraining, so it starts instantly and always shows the same model the metrics
in RESULTS.md describe.
"""

from __future__ import annotations

import streamlit as st

from dsjourney.artifacts import load_bundle
from projects.laptop_price import pipeline

BRANDS = ["Acer", "Asus", "Dell", "HP", "Lenovo", "MSI", "Toshiba", "Apple", "Others"]
TYPES = ["Notebook", "Ultrabook", "Gaming", "2 in 1 Convertible", "Workstation", "Netbook"]
CPU_TIERS = [
    "Intel Core i3",
    "Intel Core i5",
    "Intel Core i7",
    "Intel Celeron",
    "Intel Pentium",
    "AMD Ryzen",
    "AMD A/E Series",
    "Other",
]
GPU_TIERS = [
    "Intel Graphics",
    "Nvidia GTX",
    "Nvidia RTX",
    "Nvidia Quadro",
    "Nvidia Other",
    "AMD Radeon",
    "Other",
]

st.set_page_config(page_title="Laptop Price Prediction", page_icon="=", layout="centered")


@st.cache_resource
def _load():
    """Load the trained bundle once per session."""
    return load_bundle("laptop_price")


def main() -> None:
    st.title("Laptop Price Prediction")
    st.caption("Random forest over engineered hardware features. Source: 1,303 retail listings.")

    try:
        bundle = _load()
    except FileNotFoundError:
        st.error("No trained model found. Run `uv run dsj train laptop_price` first.")
        return

    with st.sidebar:
        st.subheader("Model")
        for name, value in bundle.metrics.items():
            st.metric(name.upper(), f"{value:.4f}")
        st.caption(
            f"{bundle.extra.get('estimator_key', '?')} on {bundle.extra.get('train_rows', '?')} rows"
        )

    left, right = st.columns(2)
    with left:
        company = st.selectbox("Brand", BRANDS, index=BRANDS.index("Dell"))
        type_name = st.selectbox("Form factor", TYPES)
        ram_gb = st.select_slider("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
        weight_kg = st.slider("Weight (kg)", 0.7, 4.5, 1.9, 0.1)
        inches = st.slider("Screen size (inches)", 10.0, 18.4, 15.6, 0.1)
    with right:
        resolution = st.selectbox(
            "Resolution", ["1366x768", "1920x1080", "2560x1440", "2560x1600", "3840x2160"], index=1
        )
        cpu_brand = st.selectbox("CPU tier", CPU_TIERS, index=1)
        cpu_ghz = st.slider("CPU clock (GHz)", 0.9, 3.6, 2.5, 0.1)
        cpu_generation = st.slider("CPU generation", 0, 10, 8)
        gpu_brand = st.selectbox("GPU tier", GPU_TIERS)

    storage_left, storage_right = st.columns(2)
    ssd_gb = storage_left.select_slider("SSD (GB)", [0, 128, 256, 512, 1000], value=256)
    hdd_gb = storage_right.select_slider("HDD (GB)", [0, 500, 1000, 2000], value=0)

    flags_left, flags_right = st.columns(2)
    touchscreen = flags_left.checkbox("Touchscreen")
    ips = flags_right.checkbox("IPS panel")

    if st.button("Predict price", type="primary", use_container_width=True):
        width, height = (int(part) for part in resolution.split("x"))
        payload = {
            "company": company,
            "type_name": type_name,
            "ram_gb": ram_gb,
            "weight_kg": weight_kg,
            "inches": inches,
            "screen_width": width,
            "screen_height": height,
            "touchscreen": touchscreen,
            "ips": ips,
            "ssd_gb": ssd_gb,
            "hdd_gb": hdd_gb,
            "cpu_brand": cpu_brand,
            "cpu_ghz": cpu_ghz,
            "cpu_generation": cpu_generation,
            "gpu_brand": gpu_brand,
        }
        row = bundle.prepare(pipeline.prepare_input(payload))
        price = float(pipeline.postprocess(bundle.model.predict(row))[0])
        st.success(f"Estimated price: {price:,.0f}")
        st.caption(
            "The model was trained on log-transformed prices; this figure is the "
            "prediction converted back to the original currency scale."
        )


main()
