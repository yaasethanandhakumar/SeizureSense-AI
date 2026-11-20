import os
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st

# ============================================================
# PROJECT CONFIGURATION
# ============================================================
BASE_DIR = Path(r"C:\Projects\Epilepsy Seizure Prediction")
MODEL_PATH = BASE_DIR / "saved_models" / "cnn_bilstm_attention_corrected_best.keras"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploaded_eeg"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL / EEG CONFIGURATION
# ============================================================
MODEL_PARAMETERS = "164,673"
THRESHOLD = 0.88
MIN_PREICTAL_PERCENTAGE = 5.0

CHANNELS = 23
SAMPLING_RATE = 256
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = SAMPLING_RATE * SEGMENT_SECONDS
FILTER_LOW = 0.5
FILTER_HIGH = 40.0

TRAIN_MEAN = 2.8031432730557526e-09
TRAIN_STD = 4.079479549545795e-05

# Final test performance supplied in the original dashboard
ACCURACY = 87.57
PRECISION = 87.63
RECALL = 77.25
F1_SCORE = 82.12
ROC_AUC = 94.10
SPECIFICITY = 93.61

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SeizureSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================
for key, default in {
    "page": "Dashboard",
    "prediction_done": False,
    "prediction_result": None,
    "prediction_output": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# LAZY MODEL LOADING
# TensorFlow is imported only when prediction is requested.
# ============================================================
@st.cache_resource
def get_model():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.layers import Layer
    from keras.saving import register_keras_serializable

    @register_keras_serializable(package="EEG", name="AttentionLayer")
    class AttentionLayer(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(
                name="attention_weight",
                shape=(input_shape[-1], input_shape[-1]),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b = self.add_weight(
                name="attention_bias",
                shape=(input_shape[-1],),
                initializer="zeros",
                trainable=True,
            )
            self.V = self.add_weight(
                name="attention_vector",
                shape=(input_shape[-1], 1),
                initializer="glorot_uniform",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):
            score = tf.tanh(tf.matmul(inputs, self.W) + self.b)
            score = tf.matmul(score, self.V)
            weights = tf.nn.softmax(score, axis=1)
            return tf.reduce_sum(inputs * weights, axis=1)

        def get_config(self):
            return super().get_config()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\\n{MODEL_PATH}")

    return load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )

# ============================================================
# EEG PREPROCESSING
# MNE is imported only when prediction is requested.
# ============================================================
def preprocess_eeg(file_path):
    import mne

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
    if len(eeg_channels) >= CHANNELS:
        eeg_channels = eeg_channels[:CHANNELS]
    elif len(raw.ch_names) >= CHANNELS:
        eeg_channels = raw.ch_names[:CHANNELS]
    else:
        raise ValueError(
            f"Expected at least {CHANNELS} channels, "
            f"but only {len(raw.ch_names)} were found."
        )

    raw.pick(eeg_channels)

    if int(round(raw.info["sfreq"])) != SAMPLING_RATE:
        raw.resample(SAMPLING_RATE, npad="auto", verbose=False)

    raw.filter(FILTER_LOW, FILTER_HIGH, verbose=False)
    data = raw.get_data()[:CHANNELS, :]

    number_of_segments = data.shape[1] // SEGMENT_SAMPLES
    if number_of_segments == 0:
        raise ValueError("The uploaded EEG recording is shorter than 10 seconds.")

    usable_samples = number_of_segments * SEGMENT_SAMPLES
    data = data[:, :usable_samples]

    segments = data.reshape(CHANNELS, number_of_segments, SEGMENT_SAMPLES)
    segments = np.transpose(segments, (1, 2, 0))
    segments = ((segments - TRAIN_MEAN) / TRAIN_STD).astype(np.float32)

    return segments, number_of_segments, len(raw.ch_names)

# ============================================================
# FILE / PREDICTION HELPERS
# ============================================================
def get_file_signature(file_path):
    path = Path(file_path)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns

@st.cache_data(show_spinner=False, max_entries=3)
def predict_eeg(file_path, file_signature):
    segments, number_of_segments, channel_count = preprocess_eeg(file_path)
    model = get_model()

    probabilities = np.asarray(
        model.predict(segments, batch_size=16, verbose=0)
    ).reshape(-1)

    predictions = (probabilities >= THRESHOLD).astype(int)
    normal_count = int(np.sum(predictions == 0))
    preictal_count = int(np.sum(predictions == 1))
    preictal_percentage = (preictal_count / number_of_segments) * 100
    overall_result = (
        "PREICTAL"
        if preictal_percentage >= MIN_PREICTAL_PERCENTAGE
        else "NORMAL"
    )

    return {
        "result": overall_result,
        "segments": number_of_segments,
        "normal_count": normal_count,
        "preictal_count": preictal_count,
        "preictal_percentage": preictal_percentage,
        "probabilities": probabilities,
        "predictions": predictions,
        "channels": channel_count,
    }

def save_uploaded_file(uploaded_file):
    file_path = UPLOAD_DIR / uploaded_file.name
    if file_path.exists() and file_path.stat().st_size == uploaded_file.size:
        return file_path

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# ============================================================
# HTML / CSS RENDERING
# IMPORTANT: Streamlit 1.60 supports st.html().
# All standalone HTML is rendered with st.html instead of
# st.markdown(..., unsafe_allow_html=True).
# ============================================================
st.html("""
<style>
.stApp {
    background:#020B1B;
    color:#fff;
}
.block-container { max-width:100%!important; padding:1rem 1.1rem!important; }
#MainMenu,footer { visibility:hidden; }

header[data-testid="stHeader"] {
    background: transparent !important;
}

section[data-testid="stSidebar"] {
    background:#020817;
    border-right:1px solid #1C314B;
}
section[data-testid="stSidebar"] > div { padding-top:.8rem; }
section[data-testid="stSidebar"] .stButton > button {
    width:100%; min-height:40px; background:#0B2542!important;
    border:1px solid #315A8A!important; border-radius:8px;
    color:#fff!important; text-align:left; box-shadow:none!important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background:#0B2542!important; border-color:#315A8A!important;
}
.sidebar-brand {display:flex;align-items:center;gap:10px;padding:10px;background:#071B31!important;opacity:1!important;border:1px solid #1D3855;border-radius:10px;margin-bottom:14px}
.sidebar-logo{font-size:30px}.sidebar-title{font-size:20px;font-weight:700;color:#FFFFFF!important;opacity:1!important}
.sidebar-section{color:#B9C9DB;font-size:11px;font-weight:600;letter-spacing:.7px;text-transform:uppercase;margin:15px 8px 8px}
.sidebar-status,.sidebar-row{border:1px solid #1D3855;border-radius:8px;background:#07182B;padding:10px 12px;margin:6px 0}
.status-ready{color:#47E88D;font-weight:700;font-size:12px}.status-description{color:#A8B9CC;font-size:11px;margin-top:6px;line-height:1.45}
.sidebar-row{display:flex;justify-content:space-between;gap:10px}.sidebar-label{color:#9FB2C8;font-size:11px}.sidebar-value{color:#fff;font-size:11px;font-weight:600;text-align:right}

.hero{height:153px;position:relative;overflow:hidden;border:1px solid #1D3855;border-radius:10px;background:#061329;}
.hero-grid{position:absolute;inset:0;background:#071B31}
.hero-left{position:absolute;left:38px;top:25px;z-index:5}.hero-title{font-size:28px;font-weight:700;margin-bottom:10px}.hero-title-ai{color:#FF72C7}.hero-subtitle{color:#C6D6E8;font-size:15px;line-height:1.55}.hero-brain{position:absolute;left:56%;top:5px;font-size:95px;}.hero-status{position:absolute;right:28px;top:25px;padding:9px 14px;border:1px solid #15475B;border-radius:7px;background:#051927;color:#47E88D;font-size:12px;font-weight:700}.hero-date{position:absolute;right:34px;bottom:23px;color:#C9D5E4;font-size:12px}

.section-title{color:#fff;font-size:14px;font-weight:700;margin:22px 0 11px}.section-icon{color:#3C9DFF;font-size:17px;margin-right:7px}
.metric-card{min-height:119px;padding:16px 18px;border:1px solid #20334B;border-radius:11px;background:#0D1D31;box-shadow:0 8px 20px #000000}
.metric-name{color:#B9C9DC;font-size:11px;margin-bottom:12px}.metric-number{font-size:27px;font-weight:700;line-height:1;margin-bottom:9px}.metric-description{color:#A8B9CC;font-size:10px}.metric-blue{color:#4B8DFF}.metric-green{color:#40E98C}.metric-yellow{color:#FFC72C}

.prediction-card{border:1px solid #1E324B;border-radius:12px;background:#09182B;padding:18px}
.step-number{display:inline-flex;width:27px;height:27px;border-radius:50%;align-items:center;justify-content:center;background:#12345A;color:#7DB5FF;font-weight:700;margin-right:8px}.step-title{color:#fff;font-size:14px;font-weight:700}.step-description{color:#9FB2C8;font-size:11px;line-height:1.5;margin:10px 0 14px}.arrow{font-size:34px;color:#4B8DFF;text-align:center;padding-top:65px}.info-strip{margin-top:12px;padding:8px 10px;border:1px solid #263F5B;border-radius:7px;background:#07192C;color:#AFC3D9;font-size:10px}.result-box{min-height:105px;padding:18px;text-align:center;border:2px solid #31577E;border-radius:9px;background:#0B213A!important;opacity:1!important}.result-dash{font-size:28px;color:#DCEBFA}.result-normal{font-size:22px;font-weight:800;color:#FFFFFF}.result-preictal{font-size:22px;font-weight:800;color:#FFFFFF}.result-description{color:#FFFFFF;font-size:11px;font-weight:600;margin-top:9px}
.bright-normal{min-height:105px;padding:20px;text-align:center;border:2px solid #19A86B;border-radius:9px;background:#0B8F5A!important;color:#FFFFFF!important}.bright-normal-title{font-size:24px;font-weight:800;color:#FFFFFF}.bright-preictal{min-height:105px;padding:20px;text-align:center;border:2px solid #D9363E;border-radius:9px;background:#E53935!important;color:#FFFFFF!important}.bright-preictal-title{font-size:24px;font-weight:800;color:#FFFFFF}.bright-info{padding:14px 18px;border:2px solid #2787D9;border-radius:9px;background:#0B5FA5!important;color:#FFFFFF!important;font-size:15px;font-weight:700}.bright-ready{padding:12px 16px;border:2px solid #19A86B;border-radius:8px;background:#0B8F5A!important;color:#FFFFFF!important;font-weight:700}.bright-fail{padding:12px 16px;border:2px solid #D9363E;border-radius:8px;background:#C62828!important;color:#FFFFFF!important;font-weight:700}.detail-card{min-height:100px;padding:15px 18px;border:2px solid #31577E;border-radius:9px;background:#102D4A!important}.detail-label{font-size:13px;color:#CFE6FF;font-weight:700}.detail-value{font-size:30px;color:#FFFFFF;font-weight:800;margin-top:8px}
.system-card{min-height:95px;padding:14px;border:1px solid #20334B;border-radius:10px;background:#0B2038}.system-title{color:#9FB2C8;font-size:10px}.system-value{color:#fff;font-size:13px;font-weight:600;margin-top:12px;line-height:1.35}
.footer{margin-top:25px;padding:14px;text-align:center;border-top:1px solid #1D3148;color:#8196AD;font-size:10px}.footer-pink{color:#FF72C7}
/* ------------------------------------------------------------
   SOLID STREAMLIT CONTROLS
   ------------------------------------------------------------ */
[data-testid="stFileUploader"] {
    background:#020B1B !important;
    border:0 !important;
    box-shadow:none !important;
    padding:0 !important;
    opacity:1 !important;
    color:#FFFFFF !important;
}
[data-testid="stFileUploader"] * {
    opacity:1 !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color:#FFFFFF !important;
}
[data-testid="stFileUploaderDropzone"] {
    background:#0B2038 !important;
    border:1px solid #315A8A !important;
    border-radius:8px !important;
    box-shadow:none !important;
    opacity:1 !important;
    color:#FFFFFF !important;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:focus-within {
    background:#0B2038 !important;
    border-color:#315A8A !important;
    box-shadow:none !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] button:focus,
[data-testid="stFileUploaderDropzone"] button:active {
    background:#173A61 !important;
    color:#FFFFFF !important;
    border:1px solid #477FB5 !important;
    box-shadow:none !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    fill:#FFFFFF !important;
    color:#FFFFFF !important;
}

/* Native Streamlit metrics and alerts: fully solid */
[data-testid="stMetric"],
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    opacity:1 !important;
}
[data-testid="stMetricLabel"] p {color:#B9C9DC !important;}
[data-testid="stMetricValue"] {color:#FFFFFF !important;}
div[data-testid="stAlert"] {
    background:#07192C !important;
    border:1px solid #263F5B !important;
    opacity:1 !important;
}
div[data-testid="stAlert"] * {opacity:1 !important;}

.stButton > button,
.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
    border-radius:8px !important;
    background:#173A61 !important;
    color:#FFFFFF !important;
    border:1px solid #477FB5 !important;
    box-shadow:none !important;
    opacity:1 !important;
}
.stFileUploader {margin-bottom:5px;}
</style>
""")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.html("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🧠</div>
        <div class="sidebar-title">SeizureSense AI</div>
    </div>
    <div class="sidebar-section">Navigation</div>
    """)

    navigation = [
        ("🏠", "Dashboard"),
        ("📊", "Model Performance"),
        ("〽️", "EEG Prediction"),
        ("ℹ️", "About"),
    ]

    for icon, label in navigation:
        if st.button(f"{icon}   {label}", key=f"sidebar_{label}", use_container_width=True):
            st.session_state.page = label
            st.session_state.prediction_done = False
            st.session_state.prediction_result = None
            st.session_state.prediction_output = None
            st.rerun()

    st.html('<div class="sidebar-section">System Status</div>')
    if MODEL_PATH.exists():
        st.html("""
        <div class="sidebar-status">
            <div class="status-ready">● &nbsp; MODEL READY</div>
            <div class="status-description">Final CNN-BiLSTM-Attention model is loaded and ready for predictions.</div>
        </div>
        """)
    else:
        st.error("Final model was not found.")

    st.html('<div class="sidebar-section">Model Information</div>')
    for label, value in [
        ("Architecture", "CNN + BiLSTM + Attention"),
        ("Parameters", MODEL_PARAMETERS),
        ("Threshold", f"{THRESHOLD:.2f}"),
        ("Classes", "Normal / Preictal"),
    ]:
        st.html(f'<div class="sidebar-row"><span class="sidebar-label">{label}</span><span class="sidebar-value">{value}</span></div>')

    st.html('<div class="sidebar-section">EEG Configuration</div>')
    for label, value in [
        ("Channels", str(CHANNELS)),
        ("Sampling Rate", f"{SAMPLING_RATE} Hz"),
        ("Segment Length", f"{SEGMENT_SECONDS} sec"),
        ("Filter Range", f"{FILTER_LOW} – {FILTER_HIGH} Hz"),
    ]:
        st.html(f'<div class="sidebar-row"><span class="sidebar-label">{label}</span><span class="sidebar-value">{value}</span></div>')

page = st.session_state.page

# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    current_time = datetime.now().strftime("%d %B %Y  |  %I:%M:%S %p")
    st.html(f"""
    <div class="hero">
        <div class="hero-grid"></div>
        <div class="hero-left">
            <div class="hero-title">🧠 SeizureSense <span class="hero-title-ai">AI</span></div>
            <div class="hero-subtitle">Attention-Based Deep Learning for<br>Early Epileptic Seizure Prediction</div>
        </div>
        <div class="hero-brain">🧠</div>
        <div class="hero-status">🟢 &nbsp; SYSTEM OPERATIONAL</div>
        <div class="hero-date">📅 &nbsp; {current_time}</div>
    </div>
    """)

    st.html('<div class="section-title"><span class="section-icon">▥</span>MODEL PERFORMANCE</div>')
    metric_columns = st.columns(5, gap="medium")
    metrics = [
        ("ACCURACY", f"{ACCURACY:.2f}%", "Overall accuracy", "metric-blue"),
        ("PRECISION", f"{PRECISION:.2f}%", "Preictal precision", "metric-green"),
        ("RECALL", f"{RECALL:.2f}%", "Preictal recall", "metric-yellow"),
        ("F1 SCORE", f"{F1_SCORE:.2f}%", "Harmonic mean", "metric-blue"),
        ("ROC-AUC", f"{ROC_AUC:.2f}%", "Discrimination ability", "metric-green"),
    ]
    for column, (name, value, description, color) in zip(metric_columns, metrics):
        with column:
            st.html(f'<div class="metric-card"><div class="metric-name">{name}</div><div class="metric-number {color}">{value}</div><div class="metric-description">{description}</div></div>')

    st.html('<div class="section-title"><span class="section-icon">〽</span>EEG SEIZURE PREDICTION</div>')
    c1, arrow_col, c2, c3 = st.columns([1.1, 0.18, 1.05, 1.05], gap="medium")

    with c1:
        st.html('<div><span class="step-number">1</span><span class="step-title">Upload EEG EDF File</span></div><div class="step-description">Upload a multi-channel EEG recording in EDF format.</div>')
        uploaded_file = st.file_uploader("Upload EEG EDF file", type=["edf"], key="dashboard_edf", label_visibility="collapsed")
    
    with arrow_col:
        st.html('<div class="arrow">➜</div>')
    
    with c2:
        st.html('<div><span class="step-number">2</span><span class="step-title">Run Prediction</span></div><div class="step-description">Click below to analyze the uploaded EEG.</div>')
        run_prediction = st.button("▶  Run Prediction", use_container_width=True, key="dashboard_run_prediction")
        st.html(f'<div class="info-strip">ⓘ &nbsp; Final model uses threshold <b>{THRESHOLD:.2f}</b>.</div>')
    
    with c3:
        st.html('<div><span class="step-number">3</span><span class="step-title">View Results</span></div><div class="step-description">Prediction results appear here.</div>')
        if not st.session_state.prediction_done:
            st.html('<div class="result-box"><div class="result-dash">-</div><div class="result-description">No prediction yet<br><br>Upload an EEG file and run prediction.</div></div>')
        elif st.session_state.prediction_result == "NORMAL":
            st.html('<div class="bright-normal"><div class="bright-normal-title">✓ NORMAL</div><div class="result-description">No preictal activity detected</div></div>')
        else:
            st.html('<div class="bright-preictal"><div class="bright-preictal-title">⚠ PREICTAL</div><div class="result-description">Preictal activity detected</div></div>')
    

    if run_prediction:
        if uploaded_file is None:
            st.warning("Please upload an EEG EDF file before running prediction.")
        else:
            with st.spinner("Processing EEG signal and running prediction..."):
                try:
                    file_path = save_uploaded_file(uploaded_file)
                    result = predict_eeg(str(file_path), get_file_signature(file_path))
                    st.session_state.prediction_done = True
                    st.session_state.prediction_result = result["result"]
                    st.session_state.prediction_output = result
                    st.rerun()
                except Exception as error:
                    st.session_state.prediction_done = False
                    st.error("Prediction failed.")
                    st.exception(error)

    if st.session_state.prediction_done and st.session_state.prediction_output:
        result = st.session_state.prediction_output
        st.html('<div class="section-title"><span class="section-icon">◉</span>PREDICTION ANALYSIS</div>')
        detail_columns = st.columns(4)
        details = [
            ("Total Segments", str(result["segments"])),
            ("Normal Segments", str(result["normal_count"])),
            ("Preictal Segments", str(result["preictal_count"])),
            ("Preictal Percentage", f'{result["preictal_percentage"]:.2f}%'),
        ]
        for column, (label, value) in zip(detail_columns, details):
            with column:
                st.metric(label, value)
        st.info(f"Classification threshold: {THRESHOLD:.2f} | File-level threshold: {MIN_PREICTAL_PERCENTAGE:.1f}%")

    st.html('<div class="section-title"><span class="section-icon">▣</span>SYSTEM INFORMATION</div>')
    system_columns = st.columns(6, gap="medium")
    system_information = [
        ("🧠", "Architecture", "CNN + BiLSTM +<br>Attention"),
        ("〽", "EEG Channels", "23 Channels"),
        ("◉", "Sampling Rate", "256 Hz"),
        ("✣", "Segment Length", "10 Seconds"),
        ("⚙", "Filter Range", "0.5 – 40 Hz"),
        ("♧", "Output Classes", "Normal / Preictal"),
    ]
    for column, (icon, title, value) in zip(system_columns, system_information):
        with column:
            st.html(f'<div class="system-card"><div class="system-title">{icon}&nbsp; {title}</div><div class="system-value">{value}</div></div>')

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.html('<div class="section-title"><span class="section-icon">▥</span>MODEL PERFORMANCE</div>')
    st.title("Final Model Performance")
    st.write("Final evaluation on the untouched test set.")

    performance_columns = st.columns(5, gap="medium")
    performance_data = [
        ("ACCURACY", f"{ACCURACY:.2f}%", "Overall accuracy", "metric-blue"),
        ("PRECISION", f"{PRECISION:.2f}%", "Preictal precision", "metric-green"),
        ("RECALL", f"{RECALL:.2f}%", "Preictal recall", "metric-yellow"),
        ("F1 SCORE", f"{F1_SCORE:.2f}%", "Harmonic mean", "metric-blue"),
        ("ROC-AUC", f"{ROC_AUC:.2f}%", "Discrimination ability", "metric-green"),
    ]
    for column, (name, value, description, color) in zip(performance_columns, performance_data):
        with column:
            st.html(f'<div class="metric-card"><div class="metric-name">{name}</div><div class="metric-number {color}">{value}</div><div class="metric-description">{description}</div></div>')

    st.markdown(f"""
### Final Configuration

- **Model:** CNN + BiLSTM + Attention
- **Parameters:** {MODEL_PARAMETERS}
- **Classification threshold:** {THRESHOLD:.2f}
- **Input:** {CHANNELS}-channel EEG
- **Sampling rate:** {SAMPLING_RATE} Hz
- **Segment duration:** {SEGMENT_SECONDS} seconds
- **Band-pass filter:** {FILTER_LOW}–{FILTER_HIGH} Hz
""")

    st.markdown("### Confusion Matrix")
    matrix_columns = st.columns(4)
    for column, (label, value) in zip(matrix_columns, [("TN", "337"), ("FP", "23"), ("FN", "48"), ("TP", "163")]):
        with column:
            st.metric(label, value)
    st.info(f"The threshold of {THRESHOLD:.2f} was selected using the validation set and then frozen before final test evaluation.")

# ============================================================
# EEG PREDICTION PAGE
# ============================================================
elif page == "EEG Prediction":
    st.html('<div class="section-title"><span class="section-icon">〽</span>EEG SEIZURE PREDICTION</div>')
    st.title("EEG Prediction")
    st.write("Upload a multi-channel EEG recording in EDF format.")

    prediction_file = st.file_uploader("Choose EEG EDF file", type=["edf"], key="prediction_page_file")
    if prediction_file is not None:
        st.html(f'<div class="bright-ready">✓ Ready: {prediction_file.name}</div>')
        if st.button("▶ Run EEG Prediction", use_container_width=True, key="prediction_page_button"):
            with st.spinner("Analyzing EEG signal..."):
                try:
                    prediction_path = save_uploaded_file(prediction_file)
                    result = predict_eeg(str(prediction_path), get_file_signature(prediction_path))
                    if result["result"] == "PREICTAL":
                        st.html('<div class="bright-preictal"><div class="bright-preictal-title">⚠ PREICTAL ACTIVITY DETECTED</div><div class="result-description">Seizure-related preictal activity detected in the uploaded EEG.</div></div>')
                    else:
                        st.html('<div class="bright-normal"><div class="bright-normal-title">✓ NORMAL</div><div class="result-description">No preictal activity detected in the uploaded EEG.</div></div>')

                    st.markdown("### Prediction Details")
                    detail_columns = st.columns(4)
                    details = [
                        ("Segments", result["segments"]),
                        ("Normal", result["normal_count"]),
                        ("Preictal", result["preictal_count"]),
                        ("Preictal %", f'{result["preictal_percentage"]:.2f}%'),
                    ]
                    for column, (label, value) in zip(detail_columns, details):
                        with column:
                            st.html(f'<div class="detail-card"><div class="detail-label">{label}</div><div class="detail-value">{value}</div></div>')
                    st.html(f'<div class="bright-info">ⓘ Threshold used: {THRESHOLD:.2f}</div>')
                except Exception as error:
                    st.error("Prediction failed.")
                    st.exception(error)

# ============================================================
# ABOUT PAGE
# ============================================================
elif page == "About":
    st.html('<div class="section-title"><span class="section-icon">ⓘ</span>ABOUT</div>')
    st.title("About SeizureSense AI")
    st.write("""
SeizureSense AI is an EEG-based deep learning system designed for early epileptic seizure prediction using multi-channel EEG signals.
""")
    st.markdown(f"""
### Deep Learning Architecture

**CNN → BiLSTM → Attention → Classification**

**CNN**

Extracts important spatial and signal features from multi-channel EEG recordings.

**BiLSTM**

Learns temporal dependencies and sequential patterns in EEG signals.

**Attention Mechanism**

Focuses on informative portions of the temporal EEG representation.

**Classification**

Produces the final classification: **Normal / Preictal**

### Final EEG Configuration

- **Channels:** {CHANNELS}
- **Sampling Rate:** {SAMPLING_RATE} Hz
- **Segment Duration:** {SEGMENT_SECONDS} seconds
- **Band-Pass Filter:** {FILTER_LOW}–{FILTER_HIGH} Hz
- **Classification Threshold:** {THRESHOLD:.2f}
- **Output Classes:** Normal / Preictal

### Final Test Performance

- **Accuracy:** {ACCURACY:.2f}%
- **Precision:** {PRECISION:.2f}%
- **Recall:** {RECALL:.2f}%
- **F1 Score:** {F1_SCORE:.2f}%
- **ROC-AUC:** {ROC_AUC:.2f}%
- **Specificity:** {SPECIFICITY:.2f}%
""")

# ============================================================
# FOOTER
# ============================================================
st.html("""
<div class="footer">
    <span class="footer-pink">🧠</span>
    SeizureSense AI &nbsp;•&nbsp; EEG-Based Early Seizure Prediction System
    &nbsp;•&nbsp; CNN-BiLSTM with Attention Mechanism
</div>
""")
