import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Contextual Emotion Detection",
    page_icon="🧠",
    layout="centered"
)

# =========================
# CUSTOM CSS (COMPACT + ANIMATED)
# =========================
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; max-width: 820px; }
h1, h2, h3 { margin-bottom: 0.4rem; }

/* Card */
.card {
    background: #0f172a;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    border-left: 5px solid var(--accent);
    animation: fadeSlide 0.6s ease forwards;
}

/* Animation */
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Badge */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-weight: 600;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.9rem;
}

/* Legend */
.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.legend-item {
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Suggestions */
.suggest {
    display: inline-block;
    padding: 0.3rem 0.6rem;
    margin-right: 0.3rem;
    margin-top: 0.3rem;
    border-radius: 999px;
    background: #1e293b;
    font-size: 0.8rem;
}

/* Button */
button:hover {
    transform: scale(1.02);
    transition: 0.2s ease;
}

/* Progress */
.progress > div > div {
    transition: width 0.8s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# =========================
# EMOTION MAPS
# =========================
EMOTION_EMOJI = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲"
}

EMOTION_COLOR = {
    "anger": "#ef4444",
    "fear": "#a855f7",
    "joy": "#22c55e",
    "love": "#ec4899",
    "sadness": "#60a5fa",
    "surprise": "#facc15"
}

# =========================
# LOAD MODEL & OBJECTS
# =========================
model = tf.keras.models.load_model("emotion_lstm_model.keras", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = 100
EMOTIONS = list(label_encoder.classes_)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# PREDICTION FUNCTION
# =========================
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    probs = model.predict(padded, verbose=0)[0]
    idx = np.argmax(probs)
    return label_encoder.inverse_transform([idx])[0], probs[idx], probs

# =========================
# SIDEBAR (COMPACT)
# =========================
with st.sidebar:
    st.markdown("## 🧠 Contextual Emotion Detection")
    st.markdown("**Pure Deep Learning NLP App**")
    st.divider()
    st.markdown("### 🎯 What this app does")
    st.markdown("""
    - Detects **human emotion** from text  
    - Uses **Bidirectional LSTM**  
    - Handles **ambiguous sentences**  
    - Shows **confidence & uncertainty**
    """)
    st.divider()
    st.markdown("### 🧪 Tips")
    st.markdown("""
    - Try **mixed emotion** sentences  
    - Sarcasm may confuse the model  
    - Low confidence = ambiguity
    """)

# =========================
# HEADER
# =========================
st.title("🧠 Contextual Emotion Detection")
st.caption("Compact • Animated • Deep Learning NLP")

st.markdown("### 🎭 Supported Emotion Classes")

st.markdown("""
<div class='legend'>
  <div class='legend-item' style='background:#ef444420;color:#ef4444'>😠 Anger</div>
  <div class='legend-item' style='background:#a855f720;color:#a855f7'>😨 Fear</div>
  <div class='legend-item' style='background:#22c55e20;color:#22c55e'>😊 Joy</div>
  <div class='legend-item' style='background:#ec489920;color:#ec4899'>❤️ Love</div>
  <div class='legend-item' style='background:#60a5fa20;color:#60a5fa'>😢 Sadness</div>
  <div class='legend-item' style='background:#facc1520;color:#facc15'>😲 Surprise</div>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT
# =========================
text = st.text_area(
    "Text input",
    placeholder="It felt strange how light everything suddenly seemed.",
    height=90
)

# examples = [
#     "I laughed, then realized I wanted to cry.",
#     "That came out of nowhere.",
#     "I stayed, even when I didn’t have to."
# ]

# example = st.selectbox("Try example", ["—"] + examples)
# if example != "—":
#     text = example

# =========================
# PREDICT
# =========================
if st.button("✨ Analyze Emotion", use_container_width=True):
    if text.strip():
        emotion, confidence, probs = predict_emotion(text)

        st.session_state.history.append({
            "Text": text,
            "Emotion": emotion,
            "Confidence": round(confidence, 2)
        })

        emoji = EMOTION_EMOJI[emotion]
        color = EMOTION_COLOR[emotion]

        st.markdown(
            f"""
            <div class="card" style="--accent:{color}">
                <span class="badge" style="background:{color}20;color:{color}">
                    {emoji} {emotion.upper()}
                </span>
                <p style="margin-top:0.4rem"><b>Confidence:</b> {confidence:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(confidence))

        if confidence < 0.5:
            st.warning("⚠️ Ambiguous sentence detected")

        # Compact distribution
        df = pd.DataFrame({"Emotion": EMOTIONS, "Score": probs})
        st.bar_chart(df.set_index("Emotion"), height=180)

# =========================
# HISTORY (COLLAPSIBLE)
# =========================
if st.session_state.history:
    with st.expander("🕘 Prediction History"):
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# =========================
# ABOUT
# =========================
# with st.expander("ℹ️ Model Info"):
#     st.markdown("""
#     **Architecture:** Bidirectional LSTM  
#     **Classes:** 😠 😨 😊 ❤️ 😢 😲  
#     **Dataset:** Emotion Dataset for NLP  
#     **Key Idea:** Emotion is contextual, not keyword-based.
#     """)
