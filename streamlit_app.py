import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import streamlit as st
import numpy as np
from PIL import Image

import cv2
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAI Studio",
    page_icon="🧬",
    layout="wide"
)


# ─────────────────────────────────────────────────────────────
# SAFE DEEPFACE LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception as e:
        st.error(f"DeepFace load failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────
def analyze_image(img_array):
    DeepFace = load_deepface()

    if DeepFace is None:
        raise RuntimeError("DeepFace not loaded")

    if img_array is None or img_array.size == 0:
        raise ValueError("Empty image")

    # Convert RGB → BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    result = DeepFace.analyze(
        img_bgr,
        actions=["age", "gender", "emotion", "race"],
        enforce_detection=False,
        silent=True
    )

    if isinstance(result, dict):
        result = [result]

    return result


# ─────────────────────────────────────────────────────────────
# RESULT DISPLAY
# ─────────────────────────────────────────────────────────────
def show_results(res_list):
    if not res_list:
        st.warning("No face detected")
        return

    res = res_list[0]

    st.subheader("Results")

    st.write("Age:", res.get("age"))
    st.write("Gender:", res.get("dominant_gender"))
    st.write("Emotion:", res.get("dominant_emotion"))
    st.write("Race:", res.get("dominant_race"))


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🧬 FaceAI Studio")

mode = st.radio("Choose Mode", ["Upload Image", "Camera"])

# ─────────────────────────────────────────────────────────────
# UPLOAD MODE
# ─────────────────────────────────────────────────────────────
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption="Input Image", use_container_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                try:
                    result = analyze_image(img_array)
                    show_results(result)
                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────
# CAMERA MODE
# ─────────────────────────────────────────────────────────────
else:
    camera = st.camera_input("Capture Image")

    if camera:
        image = Image.open(camera).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption="Captured Image", use_container_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                try:
                    result = analyze_image(img_array)
                    show_results(result)
                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("FaceAI Studio • DeepFace + Streamlit Cloud Safe Build")