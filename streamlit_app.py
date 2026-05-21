import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAI Studio",
    page_icon="🧬",
    layout="centered"
)


# ─────────────────────────────────────────────
# SAFE DEEPFACE LOADER (lazy import)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception as e:
        st.error(f"DeepFace load failed: {e}")
        return None


# ─────────────────────────────────────────────
# ANALYSIS FUNCTION (NO CV2)
# ─────────────────────────────────────────────
def analyze_image(img_array):
    DeepFace = load_deepface()

    if DeepFace is None:
        raise RuntimeError("DeepFace not available")

    if img_array is None or img_array.size == 0:
        raise ValueError("Empty image")

    # DeepFace accepts RGB numpy array directly
    result = DeepFace.analyze(
        img_array,
        actions=["age", "gender", "emotion", "race"],
        enforce_detection=False,
        silent=True
    )

    if isinstance(result, dict):
        result = [result]

    return result


# ─────────────────────────────────────────────
# RESULT DISPLAY
# ─────────────────────────────────────────────
def show_results(result):
    if not result:
        st.warning("No face detected")
        return

    r = result[0]

    st.subheader("🧠 Analysis Result")

    st.write(f"**Age:** {r.get('age', '--')}")
    st.write(f"**Gender:** {r.get('dominant_gender', '--')}")
    st.write(f"**Emotion:** {r.get('dominant_emotion', '--')}")
    st.write(f"**Race:** {r.get('dominant_race', '--')}")


# ─────────────────────────────────────────────
# UI HEADER
# ─────────────────────────────────────────────
st.title("🧬 FaceAI Studio")
st.caption("DeepFace powered facial analysis (Streamlit Cloud Safe)")


mode = st.radio("Select Mode", ["Upload Image", "Camera"])


# ─────────────────────────────────────────────
# UPLOAD MODE
# ─────────────────────────────────────────────
if mode == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption="Input Image", use_container_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing face..."):
                try:
                    result = analyze_image(img_array)
                    show_results(result)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# CAMERA MODE
# ─────────────────────────────────────────────
else:
    camera = st.camera_input("Take a photo")

    if camera:
        image = Image.open(camera).convert("RGB")
        img_array = np.array(image)

        st.image(image, caption="Captured Image", use_container_width=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing face..."):
                try:
                    result = analyze_image(img_array)
                    show_results(result)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("FaceAI Studio • Stable Streamlit Cloud Build")