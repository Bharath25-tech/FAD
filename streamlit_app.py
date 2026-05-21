import os
# Suppress TensorFlow logging before any TF import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAI Studio | Neural Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@300;500&display=swap');

:root {
    --primary: #7b2ff7;
    --secondary: #00d2ff;
    --accent: #ff6ec7;
    --bg-dark: #0a0a0f;
    --card-bg: rgba(255, 255, 255, 0.03);
    --border: rgba(255, 255, 255, 0.08);
}

/* ── App Background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #1a1a2e 0%, #0a0a0f 70%) !important;
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background:
        linear-gradient(rgba(123, 47, 247, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(123, 47, 247, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    z-index: -1;
    pointer-events: none;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer {visibility: hidden;}

/* ── Hero ── */
.hero-box {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-box h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6ec7, #00d2ff);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 8s linear infinite;
    margin-bottom: 0.3rem;
    letter-spacing: 2px;
}
@keyframes shine { to { background-position: 300% center; } }
.hero-box p {
    color: #8892b0;
    font-size: 0.8rem;
    letter-spacing: 5px;
    text-transform: uppercase;
}
.hero-line {
    width: 200px; height: 2px;
    margin: 1.2rem auto 0;
    background: linear-gradient(90deg, transparent, var(--primary), var(--secondary), transparent);
    box-shadow: 0 0 15px var(--primary);
}

/* ── Glass Card ── */
.g-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
    transition: border 0.3s ease;
}
.g-card:hover { border-color: rgba(123, 47, 247, 0.4); }
.g-card::after {
    content: "";
    position: absolute;
    top: 0; left: -100%; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
    animation: scanline 4s linear infinite;
}
@keyframes scanline { 0%{left:-100%} 100%{left:100%} }

/* ── Result Grid ── */
.res-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin: 1.2rem 0;
}
.res-card {
    background: rgba(123, 47, 247, 0.05);
    border: 1px solid rgba(123, 47, 247, 0.15);
    border-radius: 18px;
    padding: 1.3rem;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.res-card:hover {
    transform: translateY(-6px);
    background: rgba(123, 47, 247, 0.1);
    box-shadow: 0 12px 30px rgba(0,0,0,0.4), 0 0 12px rgba(123,47,247,0.2);
    border-color: var(--primary);
}
.res-icon { font-size: 2rem; display: block; margin-bottom: 0.4rem; filter: drop-shadow(0 0 8px rgba(123,47,247,0.5)); }
.res-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #6b7280; margin-bottom: 0.4rem; }
.res-value { font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 700; color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.15); }

/* ── Emotion Bars ── */
.emo-section {
    margin-top: 1.5rem; padding: 1.3rem;
    background: rgba(0,0,0,0.2);
    border-radius: 20px;
    border: 1px solid var(--border);
}
.emo-bar-wrap { margin-bottom: 0.8rem; }
.emo-bar-head {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: #a0aec0; margin-bottom: 5px;
    font-family: 'Inter', sans-serif;
}
.emo-bar-track {
    width: 100%; height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px; overflow: hidden;
}
.emo-bar-fill {
    height: 100%; border-radius: 10px;
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #06060a !important;
    border-right: 1px solid var(--border);
}
.sb-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: var(--primary); letter-spacing: 2px; margin-bottom: 0.5rem;
}
.sb-status-box {
    padding: 1rem; background: rgba(255,255,255,0.02);
    border-radius: 12px; border: 1px solid var(--border); margin-bottom: 1.5rem;
}
.sb-status {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 8px; font-size: 0.8rem; color: #8892b0;
    font-family: 'JetBrains Mono', monospace;
}
.sb-dot {
    width: 8px; height: 8px; border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ── Image Container ── */
.img-container {
    position: relative; border-radius: 20px; overflow: hidden;
    border: 1px solid var(--border);
}

/* ── Button Override ── */
.stButton>button {
    width: 100%; border-radius: 10px;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    color: white; border: none; padding: 0.5rem 1rem;
    font-weight: 700; transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(123, 47, 247, 0.4);
}
</style>
""", unsafe_allow_html=True)

# ─── Lazy-load DeepFace (heavy import) ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_deepface():
    """Import DeepFace once and cache the module."""
    from deepface import DeepFace
    return DeepFace

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1>🧬 FaceAI Studio</h1>
    <p>Neural Facial Analysis Engine</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">⚡ NEURAL CORE</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#4a5568;font-size:0.6rem;letter-spacing:3px;margin-bottom:1.5rem;">'
        'V3.0 // ANALYTICS ENGINE</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-status-box">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-status">'
        '<span class="sb-dot" style="background:#22c55e;box-shadow:0 0 8px #22c55e;"></span>'
        'DEEPFACE: ONLINE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-status">'
        '<span class="sb-dot" style="background:#22c55e;box-shadow:0 0 8px #22c55e;"></span>'
        'OPENCV: ACTIVE</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-status">'
        '<span class="sb-dot" style="background:#00d2ff;box-shadow:0 0 8px #00d2ff;"></span>'
        'MODELS: OPTIMIZED</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🛠️ CONFIGURATION")
    mode = st.radio("Operation Mode", ["🖼️ Image Upload", "📸 Camera Capture"])

    st.markdown("---")
    st.markdown("### 📊 ANALYTICS")
    st.info(
        "The engine performs real-time neural mapping to estimate "
        "age, race, and emotional state."
    )

    st.markdown("---")
    if st.button("🔄 Reset System"):
        st.rerun()

# ─── Emotion Helpers ──────────────────────────────────────────────────────────
EMOTION_DATA = {
    'happy':    ('😊', '#facc15'),
    'sad':      ('😢', '#60a5fa'),
    'angry':    ('😡', '#f87171'),
    'fear':     ('😨', '#a78bfa'),
    'surprise': ('😲', '#fb923c'),
    'neutral':  ('😐', '#94a3b8'),
    'disgust':  ('🤢', '#34d399'),
}


def show_results(analysis_list):
    """Render analysis results with premium custom HTML cards."""
    if not analysis_list:
        st.warning("No face detected. Please ensure the face is clearly visible.")
        return

    res = analysis_list[0]
    age = res.get('age', '--')
    emotion = res.get('dominant_emotion', 'neutral')
    gender = res.get('dominant_gender', 'Unknown')
    race = res.get('dominant_race', 'Unknown')
    emotions = res.get('emotion', {})

    emo_icon, _ = EMOTION_DATA.get(emotion, ('🤔', '#818cf8'))

    # Result cards grid
    st.markdown(f"""
    <div class="res-grid" style="grid-template-columns: repeat(3, 1fr);">
        <div class="res-card">
            <span class="res-icon">📅</span>
            <div class="res-label">Estimated Age</div>
            <div class="res-value">{age}</div>
        </div>
        <div class="res-card">
            <span class="res-icon">{emo_icon}</span>
            <div class="res-label">Mood</div>
            <div class="res-value">{str(emotion).capitalize()}</div>
        </div>
        <div class="res-card">
            <span class="res-icon">🌍</span>
            <div class="res-label">Ethnicity</div>
            <div class="res-value">{str(race).capitalize()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Emotion bars
    if emotions:
        st.markdown('<div class="emo-section">', unsafe_allow_html=True)
        st.markdown(
            '<div style="margin-bottom:1rem;font-size:0.7rem;font-weight:700;'
            'letter-spacing:3px;color:#8892b0;text-transform:uppercase;text-align:center;">'
            'Neural Emotion Mapping</div>',
            unsafe_allow_html=True,
        )

        sorted_emos = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

        for emo_name, score in sorted_emos:
            e_icon, e_color = EMOTION_DATA.get(emo_name, ('🤔', '#818cf8'))
            pct = max(0.0, min(float(score), 100.0))
            if pct < 0.5:
                continue

            st.markdown(f"""
            <div class="emo-bar-wrap">
                <div class="emo-bar-head">
                    <span>{e_icon} {emo_name.upper()}</span>
                    <span style="color:{e_color};font-family:'JetBrains Mono';">{pct:.1f}%</span>
                </div>
                <div class="emo-bar-track">
                    <div class="emo-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{e_color}88,{e_color});box-shadow:0 0 10px {e_color}55;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def analyze_image(img_array):
    """Run DeepFace analysis on an RGB numpy array."""
    DeepFace = load_deepface()

    if img_array is None or img_array.size == 0:
        raise ValueError("Empty image — nothing to analyze.")

    # Convert RGBA → RGB if needed
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    result = DeepFace.analyze(
        img_bgr,
        actions=["age", "emotion", "gender", "race"],
        enforce_detection=False,
        silent=True,
    )

    if isinstance(result, dict):
        result = [result]
    return result


# ─── Image Upload Mode ────────────────────────────────────────────────────────
if "🖼️" in mode:
    st.markdown("""
    <div class="g-card">
        <div style="font-size:0.65rem;letter-spacing:3px;color:var(--secondary);text-transform:uppercase;margin-bottom:0.3rem;">Module 01</div>
        <div style="font-size:1.4rem;font-weight:700;color:#fff;">Neural Portrait Decoder</div>
        <div style="color:#8892b0;font-size:0.9rem;margin-top:0.2rem;">Upload a high-resolution portrait for deep attribute extraction.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        img_array = np.array(image)

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            with st.spinner("🧠 DECODING NEURAL FEATURES..."):
                try:
                    results = analyze_image(img_array)
                    show_results(results)
                    st.success("✅ DECODING COMPLETE")
                except Exception as e:
                    st.error(f"ENGINE ERROR: {e}")

# ─── Camera Capture Mode ──────────────────────────────────────────────────────
else:
    st.markdown("""
    <div class="g-card">
        <div style="font-size:0.65rem;letter-spacing:3px;color:var(--accent);text-transform:uppercase;margin-bottom:0.3rem;">Module 02</div>
        <div style="font-size:1.4rem;font-weight:700;color:#fff;">Live Biometric Capture</div>
        <div style="color:#8892b0;font-size:0.9rem;margin-top:0.2rem;">Initiate real-time optic scan via system camera.</div>
    </div>
    """, unsafe_allow_html=True)

    camera_photo = st.camera_input("OPTIC SCAN")

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        img_array = np.array(image)

        col_1, col_2 = st.columns([1, 1.2])

        with col_1:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_2:
            with st.spinner("📡 ESTABLISHING NEURAL LINK..."):
                try:
                    results = analyze_image(img_array)
                    show_results(results)
                    st.success("✅ BIOMETRIC DATA EXTRACTED")
                except Exception as e:
                    st.error(f"CAPTURE ERROR: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.7rem;letter-spacing:2px;padding:1rem;">
    CORE SYSTEM: ACTIVE | NEURAL ENGINE: V3.0 | ENCRYPTION: AES-256
</div>
""", unsafe_allow_html=True)
