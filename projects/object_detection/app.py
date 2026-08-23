"""Streamlit demo for face and object detection.

Run with ``dsj serve object_detection``. The parameter sliders are the point:
the source notebook left `scaleFactor` at whatever the tutorial used, and on
the reference image that choice is the difference between six faces and none.
"""

from __future__ import annotations

import streamlit as st

from dsjourney import detection
from projects.object_detection import pipeline

st.set_page_config(page_title="Detection", page_icon="=", layout="wide")


@st.cache_data
def _image(name: str):
    """Read one sample image."""
    return detection.load_image(pipeline.image_path(name))


def main() -> None:
    st.title("Face and Object Detection")
    st.caption("Haar cascades, tuned against a counted ground truth rather than left at defaults.")

    try:
        listing = pipeline.load_raw()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    names = listing["image"].tolist()
    name = st.selectbox("Image", names, index=names.index("g8.jpg") if "g8.jpg" in names else 0)

    with st.sidebar:
        st.subheader("Cascade parameters")
        scale_factor = st.slider("scale_factor", 1.02, 1.40, 1.05, 0.01)
        min_neighbours = st.slider("min_neighbours", 1, 12, 5)
        min_size = st.slider("min_size (px)", 8, 80, 24)
        st.caption(
            "scale_factor closer to 1.0 searches more scales and finds more faces. "
            "At 1.30 the reference image yields zero detections."
        )
        expected = pipeline.FACE_COUNTS.get(name)
        if expected is not None:
            st.metric("Faces actually present", expected)

    image = _image(name)
    boxes = pipeline.detect_in(
        name, scale_factor=scale_factor, min_neighbours=min_neighbours, min_size=min_size
    )

    left, right = st.columns([3, 1])
    with left:
        st.pyplot(detection.draw_boxes(image, boxes, title=f"{name}: {len(boxes)} detection(s)"))
    with right:
        st.metric("Detected", len(boxes))
        if expected is not None:
            st.metric("Error", abs(len(boxes) - expected))
            if expected == 0 and boxes:
                st.warning(
                    "There are no frontal faces in this photograph - everything found here "
                    "is a false positive, which is what a low min_neighbours buys you."
                )

    if expected is not None:
        st.subheader("Parameter sweep")
        st.dataframe(
            detection.sweep_cascade_parameters(image, pipeline.cascade_path(), expected=expected),
            use_container_width=True,
        )


main()
