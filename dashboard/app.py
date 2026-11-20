import streamlit as st
import streamlit.components.v1 as components
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Projects\Epilepsy Seizure Prediction")

PREDICT_SCRIPT = BASE_DIR / "predict_eeg.py"

MODEL_PATH = (
    BASE_DIR
    / "saved_models"
    / "cnn_bilstm_attention_corrected.keras"
)

OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR = OUTPUT_DIR / "uploaded_eeg"

RESULT_FILE = OUTPUT_DIR / "eeg_prediction_results.txt"


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SeizureSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_output" not in st.session_state:
    st.session_state.prediction_output = ""


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 75% 0%,
            rgba(40, 100, 180, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #020817 0%,
            #03152E 55%,
            #020B1B 100%
        );

    color: #FFFFFF;
}


.block-container {
    max-width: 100% !important;

    padding-top: 0.8rem !important;

    padding-left: 1.1rem !important;

    padding-right: 1.1rem !important;

    padding-bottom: 1rem !important;
}


header {
    visibility: hidden;
}


#MainMenu {
    visibility: hidden;
}


footer {
    visibility: hidden;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #020817 0%,
            #031426 55%,
            #020817 100%
        );

    border-right:
        1px solid #1C314B;
}


section[data-testid="stSidebar"] > div {
    padding-top: 0.8rem;
}


section[data-testid="stSidebar"] .stButton > button {

    width: 100%;

    min-height: 42px;

    background:
        transparent !important;

    border:
        1px solid transparent !important;

    border-radius: 8px;

    color:
        #FFFFFF !important;

    font-size: 14px;

    font-weight: 600;

    text-align: left;

    box-shadow: none !important;
}


section[data-testid="stSidebar"] .stButton > button:hover {

    background:
        rgba(
            47,
            93,
            180,
            0.35
        ) !important;

    border:
        1px solid #315A8A !important;

    color:
        #FFFFFF !important;
}


section[data-testid="stSidebar"] p {

    color:
        #D7E6F4 !important;
}


.sidebar-heading {

    color:
        #8EB7D7;

    font-size:
        10px;

    font-weight:
        800;

    letter-spacing:
        1px;

    text-transform:
        uppercase;

    margin-top:
        20px;

    margin-bottom:
        8px;
}


.sidebar-value {

    color:
        #54D6FF !important;

    font-size:
        12px;

    font-weight:
        700;
}


.sidebar-ready {

    color:
        #43F08E;

    font-size:
        12px;

    font-weight:
        800;
}


.sidebar-not-ready {

    color:
        #FF3344;

    font-size:
        12px;

    font-weight:
        800;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            #082640,
            #041426
        );

    border:
        1px solid #20405D;

    border-radius:
        12px;

    padding:
        15px 17px;

    min-height:
        110px;
}


[data-testid="stMetricLabel"] {

    color:
        #BBD0E3 !important;

    font-size:
        12px !important;
}


[data-testid="stMetricValue"] {

    color:
        #FFFFFF !important;

    font-size:
        27px !important;

    font-weight:
        800 !important;
}


[data-testid="stMetricDelta"] {

    color:
        #70D5FF !important;

    font-size:
        10px !important;
}


/* ============================================================
   CONTAINERS
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {

    background:
        linear-gradient(
            145deg,
            rgba(
                8,
                31,
                53,
                0.96
            ),
            rgba(
                3,
                17,
                32,
                0.96
            )
        );

    border:
        1px solid #1E3D59 !important;

    border-radius:
        12px !important;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploaderDropzone"] {

    background:
        #06192C !important;

    border:
        1px dashed #42617D !important;

    border-radius:
        9px !important;
}


[data-testid="stFileUploaderDropzoneInstructions"] {

    color:
        #C5D7E8 !important;
}


[data-testid="stFileUploader"] button {

    background:
        #3267D5 !important;

    color:
        #FFFFFF !important;

    border:
        none !important;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {

    background:
        linear-gradient(
            90deg,
            #3737C8,
            #4C51DE
        ) !important;

    color:
        #FFFFFF !important;

    border:
        1px solid #515BE6 !important;

    border-radius:
        8px;

    min-height:
        43px;

    font-weight:
        700;

    box-shadow:
        0 5px 18px
        rgba(
            45,
            65,
            220,
            0.22
        ) !important;
}


.stButton > button:hover {

    background:
        linear-gradient(
            90deg,
            #4747D8,
            #5C61EE
        ) !important;

    color:
        #FFFFFF !important;
}


/* ============================================================
   NORMAL RESULT
   ============================================================ */

.normal-result {

    background:
        #00B84A;

    color:
        #FFFFFF;

    border:
        2px solid #32FF7E;

    border-radius:
        12px;

    padding:
        20px;

    text-align:
        center;

    font-size:
        22px;

    font-weight:
        900;

    box-shadow:
        0 0 20px
        rgba(
            0,
            255,
            100,
            0.45
        );

    margin-top:
        10px;
}


.normal-subtext {

    color:
        #FFFFFF;

    font-size:
        13px;

    font-weight:
        600;

    margin-top:
        7px;
}


/* ============================================================
   BRIGHT RED PREICTAL RESULT
   ============================================================ */

.preictal-result {

    background:
        #FF0000;

    color:
        #FFFFFF;

    border:
        3px solid #FF4D4D;

    border-radius:
        12px;

    padding:
        20px;

    text-align:
        center;

    font-size:
        22px;

    font-weight:
        900;

    letter-spacing:
        0.4px;

    box-shadow:
        0 0 18px
        rgba(
            255,
            0,
            0,
            0.95
        ),
        0 0 38px
        rgba(
            255,
            0,
            0,
            0.60
        );

    margin-top:
        10px;
}


.preictal-icon {

    font-size:
        30px;

    margin-right:
        8px;
}


.preictal-subtext {

    color:
        #FFFFFF;

    font-size:
        13px;

    font-weight:
        700;

    margin-top:
        7px;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {

    color:
        #FFFFFF;

    font-size:
        15px;

    font-weight:
        800;

    margin-top:
        22px;

    margin-bottom:
        12px;
}


.section-icon {

    color:
        #54D6FF;

    margin-right:
        7px;
}


/* ============================================================
   ALERTS
   ============================================================ */

[data-testid="stAlert"] {

    border-radius:
        9px !important;
}


[data-testid="stAlert"] p {

    color:
        #FFFFFF !important;
}


/* ============================================================
   EXPANDER
   ============================================================ */

[data-testid="stExpander"] {

    background:
        #06182A !important;

    border:
        1px solid #203B55 !important;

    border-radius:
        9px !important;
}


[data-testid="stExpander"] summary {

    color:
        #FFFFFF !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer-text {

    text-align:
        center;

    color:
        #8198AE;

    font-size:
        10px;

    margin-top:
        20px;

    padding-top:
        12px;

    border-top:
        1px solid #172B42;
}


/* ============================================================
   FINAL TEXT VISIBILITY FIX
   ============================================================ */

/* Sidebar brand: SeizureSense AI */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h1 *,
section[data-testid="stSidebar"] h2 *,
section[data-testid="stSidebar"] h3 *,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 * {
    color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Sidebar navigation text */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] .stButton > button *,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* EEG uploader label: Choose EEG EDF file */
[data-testid="stFileUploader"] > label,
[data-testid="stFileUploader"] > label *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label * {
    color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Uploader instruction text */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"],
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* File type / size text */
[data-testid="stFileUploader"] small {
    color: #DCEBFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Upload button text */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
    opacity: 1 !important;
    visibility: visible !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOGO BANNER
# ============================================================

def show_logo_banner():

    now = datetime.now()

    date_text = now.strftime(
        "%d %B %Y"
    )

    time_text = now.strftime(
        "%I:%M:%S %p"
    )


    logo_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>

    body {{
        margin: 0;
        background: #03152E;
        font-family: Arial, Helvetica, sans-serif;
    }}

    .banner {{
        height: 150px;
        width: 100%;
        border: 1px solid #1D3B59;
        border-radius: 12px;
        background:
            radial-gradient(
                circle at 70% 50%,
                rgba(42, 100, 190, 0.15),
                transparent 35%
            ),
            linear-gradient(
                110deg,
                #041329,
                #031A36,
                #020B1B
            );
        position: relative;
        overflow: hidden;
    }}

    .banner::after {{
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        right: 22%;
        top: -110px;
        border-radius: 50%;
        border: 1px solid rgba(84,214,255,0.10);
        box-shadow:
            0 0 0 30px rgba(84,214,255,0.025),
            0 0 0 60px rgba(84,214,255,0.018);
    }}

    .logo {{
        position: absolute;
        left: 36px;
        top: 31px;
        display: flex;
        align-items: center;
        z-index: 5;
    }}

    .logo-box {{
        width: 67px;
        height: 67px;
        border-radius: 18px;
        background:
            linear-gradient(
                145deg,
                #0877D5,
                #27389A
            );
        border: 1px solid #65DFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow:
            0 0 25px
            rgba(84,214,255,0.25);
    }}

    .logo-symbol {{
        position: relative;
        width: 39px;
        height: 39px;
    }}

    .pulse {{
        position: absolute;
        width: 39px;
        height: 20px;
        left: 0;
        top: 10px;
    }}

    .pulse::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 10px;
        width: 7px;
        height: 3px;
        background: white;
        box-shadow:
            8px 0 0 white,
            16px -12px 0 white,
            19px 12px 0 white,
            28px 0 0 white,
            36px 0 0 white;
        transform: skewY(-18deg);
    }}

    .cross {{
        position: absolute;
        left: 16px;
        top: 2px;
        width: 8px;
        height: 35px;
        border-left: 3px solid #FF72C7;
        transform: rotate(25deg);
    }}

    .brand {{
        margin-left: 18px;
    }}

    .brand-name {{
        color: #FFFFFF;
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }}

    .brand-name span {{
        color: #FF72C7;
    }}

    .brand-subtitle {{
        color: #B8CCE0;
        font-size: 13px;
        margin-top: 7px;
    }}

    .status {{
        position: absolute;
        right: 26px;
        top: 21px;
        padding: 9px 13px;
        border-radius: 8px;
        border: 1px solid #176A43;
        background: rgba(2,33,24,0.90);
        color: #49EA91;
        font-size: 11px;
        font-weight: 800;
    }}

    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #49EA91;
        box-shadow: 0 0 8px #49EA91;
        margin-right: 6px;
    }}

    .date {{
        position: absolute;
        right: 28px;
        bottom: 21px;
        color: #AFC5D9;
        font-size: 10px;
        text-align: right;
        line-height: 1.6;
    }}

    </style>
    </head>

    <body>

    <div class="banner">

        <div class="logo">

            <div class="logo-box">

                <div class="logo-symbol">

                    <div class="pulse"></div>

                    <div class="cross"></div>

                </div>

            </div>

            <div class="brand">

                <div class="brand-name">
                    SeizureSense <span>AI</span>
                </div>

                <div class="brand-subtitle">
                    Attention-Based Deep Learning for Early Epileptic Seizure Prediction
                </div>

            </div>

        </div>

        <div class="status">
            <span class="status-dot"></span>
            SYSTEM OPERATIONAL
        </div>

        <div class="date">
            📅 {date_text}<br>
            {time_text}
        </div>

    </div>

    </body>
    </html>
    """


    components.html(
        logo_html,
        height=158,
        scrolling=False
    )


# ============================================================
# PREDICTION RESULT DETECTION
# ============================================================

def detect_prediction(output):

    """
    Detect ONLY the final file-level RESULT line.

    Individual segment predictions such as:

        Prediction: PREICTAL

    must NOT change the overall file classification.

    Expected final RESULT lines:

        RESULT: PREICTAL ACTIVITY DETECTED

    OR:

        RESULT: NO PREICTAL ACTIVITY DETECTED
    """

    text = (
        output
        .replace("\r", "")
        .strip()
    )


    # --------------------------------------------------------
    # Find ONLY the line beginning with RESULT:
    # --------------------------------------------------------

    result_line = None


    for line in text.splitlines():

        line = line.strip()

        if line.upper().startswith("RESULT:"):

            result_line = line.upper()

            break


    # --------------------------------------------------------
    # No RESULT line found
    # --------------------------------------------------------

    if result_line is None:

        return "UNKNOWN"


    # --------------------------------------------------------
    # Exact file-level classification
    # --------------------------------------------------------

    if result_line == "RESULT: PREICTAL ACTIVITY DETECTED":

        return "PREICTAL"


    if result_line == "RESULT: NO PREICTAL ACTIVITY DETECTED":

        return "NORMAL"


    # --------------------------------------------------------
    # Unknown result
    # --------------------------------------------------------

    return "UNKNOWN"


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction(file_path):

    try:

        process = subprocess.run(
            [
                sys.executable,
                str(PREDICT_SCRIPT)
            ],
            input=str(file_path) + "\n",
            text=True,
            capture_output=True,
            cwd=str(BASE_DIR),
            timeout=600
        )


        stdout = process.stdout or ""

        stderr = process.stderr or ""


        complete_output = (
            stdout
            + "\n"
            + stderr
        )


        # ----------------------------------------------------
        # Read prediction result file
        # ----------------------------------------------------

        result_text = ""


        if RESULT_FILE.exists():

            try:

                result_text = RESULT_FILE.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )


                complete_output += (
                    "\n"
                    + result_text
                )


            except Exception:

                pass


        # ----------------------------------------------------
        # Check process status
        # ----------------------------------------------------

        if process.returncode != 0:

            return {
                "success": False,
                "result": "ERROR",
                "output": complete_output
            }


        # ----------------------------------------------------
        # IMPORTANT:
        # Detect classification ONLY from RESULT:
        # ----------------------------------------------------

        result = detect_prediction(
            result_text
        )


        return {
            "success": True,
            "result": result,
            "output": complete_output
        }


    except subprocess.TimeoutExpired:

        return {
            "success": False,
            "result": "ERROR",
            "output": "Prediction timed out after 10 minutes."
        }


    except Exception as error:

        return {
            "success": False,
            "result": "ERROR",
            "output": str(error)
        }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 SeizureSense **AI**"
    )


    st.markdown("---")


    st.markdown(
        '<div class="sidebar-heading">Navigation</div>',
        unsafe_allow_html=True
    )


    navigation = [
        ("🏠", "Dashboard"),
        ("📊", "Model Performance"),
        ("〽️", "EEG Prediction"),
        ("ℹ️", "About")
    ]


    for icon, label in navigation:

        clicked = st.button(
            f"{icon}   {label}",
            key=f"nav_{label}",
            use_container_width=True
        )


        if clicked:

            st.session_state.page = label

            st.rerun()


    st.markdown(
        '<div class="sidebar-heading">System Status</div>',
        unsafe_allow_html=True
    )


    if MODEL_PATH.exists():

        st.markdown(
            '<p class="sidebar-ready">● MODEL READY</p>',
            unsafe_allow_html=True
        )

        st.caption(
            "AI model is loaded and ready."
        )

    else:

        st.markdown(
            '<p class="sidebar-not-ready">● MODEL NOT FOUND</p>',
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="sidebar-heading">Model Information</div>',
        unsafe_allow_html=True
    )


    st.write(
        "**Architecture:**"
    )


    st.markdown(
        '<span class="sidebar-value">CNN + BiLSTM + Attention</span>',
        unsafe_allow_html=True
    )


    st.write(
        "**Parameters:**"
    )


    st.markdown(
        '<span class="sidebar-value">148,610</span>',
        unsafe_allow_html=True
    )


    st.write(
        "**Classes:**"
    )


    st.markdown(
        '<span class="sidebar-value">Normal / Preictal</span>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="sidebar-heading">EEG Configuration</div>',
        unsafe_allow_html=True
    )


    st.write(
        "**Channels:** 23"
    )


    st.write(
        "**Sampling Rate:** 256 Hz"
    )


    st.write(
        "**Segment Length:** 10 sec"
    )


    st.write(
        "**Filter Range:** 0.5 – 40 Hz"
    )


# ============================================================
# CURRENT PAGE
# ============================================================

page = st.session_state.page


# ============================================================
# DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    show_logo_banner()


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown(
        '<div class="section-title"><span class="section-icon">▥</span> MODEL PERFORMANCE</div>',
        unsafe_allow_html=True
    )


    metric_columns = st.columns(
        5,
        gap="medium"
    )


    with metric_columns[0]:

        st.metric(
            "Accuracy",
            "72.16%",
            "Overall accuracy"
        )


    with metric_columns[1]:

        st.metric(
            "Precision",
            "53.31%",
            "Preictal precision"
        )


    with metric_columns[2]:

        st.metric(
            "Recall",
            "97.97%",
            "Preictal recall"
        )


    with metric_columns[3]:

        st.metric(
            "F1 Score",
            "69.05%",
            "Harmonic mean"
        )


    with metric_columns[4]:

        st.metric(
            "ROC-AUC",
            "89.14%",
            "Discrimination ability"
        )


    # ========================================================
    # EEG PREDICTION
    # ========================================================

    st.markdown(
        '<div class="section-title"><span class="section-icon">〽</span> EEG SEIZURE PREDICTION</div>',
        unsafe_allow_html=True
    )


    c1, arrow_col, c2, c3 = st.columns(
        [
            1.05,
            0.15,
            1.05,
            1.05
        ],
        gap="medium"
    )


    # ========================================================
    # UPLOAD
    # ========================================================

    with c1:

        with st.container(border=True):

            st.markdown(
                "### 1️⃣ Upload EEG EDF File"
            )


            st.caption(
                "Upload a multi-channel EEG recording."
            )


            uploaded_file = st.file_uploader(
                "Select EEG EDF file",
                type=["edf"],
                key="dashboard_edf"
            )


            if uploaded_file is not None:

                st.success(
                    f"✓ {uploaded_file.name}"
                )


    # ========================================================
    # ARROW
    # ========================================================

    with arrow_col:

        st.write("")

        st.write("")

        st.markdown(
            "## ➜"
        )


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    with c2:

        with st.container(border=True):

            st.markdown(
                "### 2️⃣ Run Prediction"
            )


            st.caption(
                "Analyze the EEG using the trained model."
            )


            run_button = st.button(
                "▶  Run Prediction",
                use_container_width=True,
                key="dashboard_run"
            )


            st.info(
                "CNN + BiLSTM + Attention"
            )


    # ========================================================
    # RESULTS
    # ========================================================

    with c3:

        with st.container(border=True):

            st.markdown(
                "### 3️⃣ View Results"
            )


            st.caption(
                "Final EEG classification."
            )


            if not st.session_state.prediction_done:

                st.info(
                    "◉ WAITING FOR PREDICTION\n\n"
                    "Upload an EEG file and run prediction."
                )


            elif st.session_state.prediction_result == "NORMAL":

                st.markdown(
                    """
                    <div class="normal-result">
                        🟢 NORMAL — STABLE
                        <div class="normal-subtext">
                            No preictal activity detected.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif st.session_state.prediction_result == "PREICTAL":

                st.markdown(
                    """
                    <div class="preictal-result">
                        <span class="preictal-icon">⚠️</span>
                        PREICTAL DETECTED
                        <div class="preictal-subtext">
                            Preictal EEG activity detected.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.warning(
                    "Prediction completed, "
                    "but the result could not be identified."
                )


    # ========================================================
    # RUN PREDICTION LOGIC
    # ========================================================

    if run_button:

        if uploaded_file is None:

            st.warning(
                "Please upload an EEG EDF file first."
            )


        elif not PREDICT_SCRIPT.exists():

            st.error(
                "predict_eeg.py was not found."
            )


        elif not MODEL_PATH.exists():

            st.error(
                "The trained model was not found."
            )


        else:

            upload_path = (
                UPLOAD_DIR
                / uploaded_file.name
            )


            with open(
                upload_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            with st.spinner(
                "Processing EEG signal..."
            ):

                prediction = run_prediction(
                    upload_path
                )


            if not prediction["success"]:

                st.session_state.prediction_done = False

                st.session_state.prediction_result = None

                st.error(
                    "Prediction process failed."
                )


                with st.expander(
                    "Technical Error"
                ):

                    st.code(
                        prediction["output"],
                        language="text"
                    )


            else:

                st.session_state.prediction_done = True

                st.session_state.prediction_result = (
                    prediction["result"]
                )

                st.session_state.prediction_output = (
                    prediction["output"]
                )

                st.rerun()


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown(
        '<div class="section-title"><span class="section-icon">ⓘ</span> HOW IT WORKS</div>',
        unsafe_allow_html=True
    )


    st.info(
        "Upload EEG → Run Prediction → "
        "CNN Feature Extraction → "
        "BiLSTM Temporal Learning → "
        "Attention Mechanism → "
        "Normal / Preictal Classification"
    )


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title"><span class="section-icon">▣</span> SYSTEM INFORMATION</div>',
        unsafe_allow_html=True
    )


    info_columns = st.columns(
        6,
        gap="medium"
    )


    system_info = [
        ("🧠", "Model Architecture", "CNN + BiLSTM + Attention"),
        ("〽️", "EEG Channels", "23 Channels"),
        ("◉", "Sampling Rate", "256 Hz"),
        ("⏱️", "Segment Length", "10 Seconds"),
        ("⚙️", "Filter Range", "0.5 – 40 Hz"),
        ("◈", "Output Classes", "2 Classes")
    ]


    for column, info in zip(
        info_columns,
        system_info
    ):

        icon, title, value = info


        with column:

            with st.container(border=True):

                st.markdown(
                    f"**{icon} {title}**"
                )


                st.caption(
                    value
                )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title"><span class="section-icon">▥</span> MODEL PERFORMANCE</div>',
        unsafe_allow_html=True
    )


    st.title(
        "Model Performance"
    )


    st.write(
        "CNN-BiLSTM Attention model evaluation metrics."
    )


    columns = st.columns(
        5,
        gap="medium"
    )


    with columns[0]:

        st.metric(
            "Accuracy",
            "72.16%"
        )


    with columns[1]:

        st.metric(
            "Precision",
            "53.31%"
        )


    with columns[2]:

        st.metric(
            "Recall",
            "97.97%"
        )


    with columns[3]:

        st.metric(
            "F1 Score",
            "69.05%"
        )


    with columns[4]:

        st.metric(
            "ROC-AUC",
            "89.14%"
        )


    st.divider()


    st.subheader(
        "Model Architecture"
    )


    st.write(
        "CNN → BiLSTM → Attention → Classification"
    )


    st.subheader(
        "Input Configuration"
    )


    st.write(
        "23-channel EEG"
    )


    st.write(
        "Sampling Rate: 256 Hz"
    )


    st.write(
        "Window Length: 10 seconds"
    )


    st.write(
        "Filter Range: 0.5–40 Hz"
    )


# ============================================================
# EEG PREDICTION PAGE
# ============================================================

elif page == "EEG Prediction":

    st.markdown(
        '<div class="section-title"><span class="section-icon">〽</span> EEG SEIZURE PREDICTION</div>',
        unsafe_allow_html=True
    )


    st.title(
        "EEG Prediction"
    )


    st.write(
        "Upload an EEG EDF recording for prediction."
    )


    prediction_file = st.file_uploader(
        "Choose EEG EDF file",
        type=["edf"],
        key="prediction_page_uploader"
    )


    if prediction_file is not None:

        prediction_path = (
            UPLOAD_DIR
            / prediction_file.name
        )


        with open(
            prediction_path,
            "wb"
        ) as file:

            file.write(
                prediction_file.getbuffer()
            )


        st.success(
            f"✓ {prediction_file.name}"
        )


        prediction_button = st.button(
            "▶ Run EEG Prediction",
            use_container_width=True,
            key="prediction_page_run"
        )


        if prediction_button:

            with st.spinner(
                "Analyzing EEG signal..."
            ):

                prediction = run_prediction(
                    prediction_path
                )


            if not prediction["success"]:

                st.error(
                    "Prediction process failed."
                )


                with st.expander(
                    "Technical Error"
                ):

                    st.code(
                        prediction["output"],
                        language="text"
                    )


            elif prediction["result"] == "NORMAL":

                st.markdown(
                    """
                    <div class="normal-result">
                        🟢 NORMAL — STABLE
                        <div class="normal-subtext">
                            No preictal activity detected.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            elif prediction["result"] == "PREICTAL":

                st.markdown(
                    """
                    <div class="preictal-result">
                        <span class="preictal-icon">⚠️</span>
                        PREICTAL DETECTED
                        <div class="preictal-subtext">
                            Preictal EEG activity detected.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.warning(
                    "Prediction completed, "
                    "but the classification could not be identified."
                )


            with st.expander(
                "Prediction Details"
            ):

                st.code(
                    prediction["output"],
                    language="text"
                )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="section-title"><span class="section-icon">ⓘ</span> ABOUT</div>',
        unsafe_allow_html=True
    )


    st.title(
        "About SeizureSense AI"
    )


    st.write(
        """
        SeizureSense AI is an EEG-based deep learning system
        for early epileptic seizure prediction using
        multi-channel EEG signals.
        """
    )


    st.subheader(
        "Architecture"
    )


    st.write(
        "CNN → BiLSTM → Attention → Classification"
    )


    st.subheader(
        "CNN"
    )


    st.write(
        "Extracts important features from EEG signals."
    )


    st.subheader(
        "BiLSTM"
    )


    st.write(
        "Learns temporal dependencies in EEG sequences."
    )


    st.subheader(
        "Attention"
    )


    st.write(
        "Focuses on informative portions of the EEG sequence."
    )


    st.subheader(
        "Classification"
    )


    st.write(
        "Produces Normal or Preictal classification."
    )


    st.subheader(
        "EEG Configuration"
    )


    st.write(
        "Channels: 23"
    )


    st.write(
        "Sampling Rate: 256 Hz"
    )


    st.write(
        "Segment Length: 10 seconds"
    )


    st.write(
        "Filter Range: 0.5–40 Hz"
    )


    st.write(
        "Output Classes: Normal / Preictal"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        🧠 SeizureSense AI &nbsp;•&nbsp;
        EEG-Based Early Seizure Prediction System &nbsp;•&nbsp;
        CNN-BiLSTM with Attention Mechanism
    </div>
    """,
    unsafe_allow_html=True
)

