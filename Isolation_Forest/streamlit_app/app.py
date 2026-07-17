"""
Anomaly Detection Dashboard — Machine Temperature / IoT Sensor Data
Model: Isolation Forest (unsupervised)

Upload any timestamped sensor CSV (timestamp, value) and this app replicates
the full feature engineering + scoring pipeline live, using the pre-trained
model artifacts (scaler.pkl, iso_forest_model.pkl, model_config.pkl).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Anomaly Detection — IoT Sensor Data",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
GOLD = "#F5B700"
GOLD_GLOW = "#FFD84D"
BG = "#0D0D0D"
CARD_BG = "#171717"
TEXT = "#FFFFFF"
TEXT_MUTED = "#D4D4D4"
RED = "#FF5C5C"
GREEN = "#3DDC97"

# ---------------------------------------------------------------------------
# Custom CSS — this is what makes it not look like default Streamlit
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 20% 0%, #1a1a1a 0%, {BG} 45%);
    }}

    /* Hide default streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Style the header bar instead of hiding it — it contains the sidebar
       collapse/expand toggle. Hiding it entirely traps the sidebar closed
       with no way to reopen it. */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        display: block !important;
        color: {GOLD} !important;
    }}
    [data-testid="collapsedControl"] svg {{
        fill: {GOLD} !important;
    }}

    /* Hero header */
    .hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: {TEXT};
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }}
    .hero-title span {{
        color: {GOLD};
        text-shadow: 0 0 20px rgba(245, 183, 0, 0.35);
    }}
    .hero-subtitle {{
        color: {TEXT_MUTED};
        font-size: 1.02rem;
        margin-bottom: 1.8rem;
        max-width: 720px;
    }}

    /* Metric cards */
    .metric-card {{
        background: linear-gradient(145deg, {CARD_BG}, #131313);
        border: 1px solid rgba(245, 183, 0, 0.15);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    }}
    .metric-label {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}
    .metric-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: {TEXT};
    }}
    .metric-accent {{ color: {GOLD}; }}
    .metric-red {{ color: {RED}; }}
    .metric-green {{ color: {GREEN}; }}

    /* Section labels */
    .section-label {{
        font-family: 'Space Grotesk', sans-serif;
        color: {GOLD};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin: 2rem 0 0.6rem 0;
        border-left: 3px solid {GOLD};
        padding-left: 10px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: #0A0A0A;
        border-right: 1px solid rgba(245, 183, 0, 0.1);
    }}

    /* Widget labels (file uploader, slider, etc.) — Streamlit defaults to a dim
       gray that disappears on dark backgrounds, so force it readable + bigger */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] label,
    .stSlider label,
    .stFileUploader label {{
        color: {TEXT} !important;
        font-weight: 600;
        font-size: 1.02rem !important;
        opacity: 1 !important;
    }}

    /* File uploader helper text ("Drag and drop..." / "Limit 200MB...") — smaller, still readable */
    [data-testid="stFileUploaderDropzoneInstructions"] div,
    [data-testid="stFileUploaderDropzoneInstructions"] span {{
        color: {TEXT_MUTED} !important;
        font-size: 0.78rem !important;
    }}

    /* Slider current-value readout and min/max ticks */
    [data-testid="stSliderThumbValue"],
    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* Uploaded filename row */
    [data-testid="stFileUploaderFile"] span {{
        color: {TEXT} !important;
    }}

    /* "Browse files" button + any secondary button — Streamlit's default hover
       turns text/background grayish-on-grayish, so define explicit states */
    section[data-testid="stSidebar"] button,
    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {{
        background-color: {CARD_BG} !important;
        color: {TEXT} !important;
        border: 1px solid rgba(245, 183, 0, 0.4) !important;
    }}
    section[data-testid="stSidebar"] button:hover,
    button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {{
        background-color: {GOLD} !important;
        color: #0D0D0D !important;
        border: 1px solid {GOLD} !important;
    }}
    section[data-testid="stSidebar"] button:hover p,
    button[kind="secondary"]:hover p {{
        color: #0D0D0D !important;
    }}

    /* st.caption text ("No file handy?...") defaults to very low-contrast gray */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    .stCaption, .stCaption p, small {{
        color: {TEXT_MUTED} !important;
        opacity: 1 !important;
    }}

    /* Dataframe */
    .stDataFrame {{
        border: 1px solid rgba(245, 183, 0, 0.15);
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Buttons */
    .stButton>button, .stDownloadButton>button {{
        background: {GOLD};
        color: #0D0D0D;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        background: {GOLD_GLOW};
        color: #0D0D0D;
    }}

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {{
        background: {CARD_BG};
        border: 1.5px dashed rgba(245, 183, 0, 0.35);
        border-radius: 12px;
    }}

    hr {{ border-color: rgba(245, 183, 0, 0.1); }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    model = joblib.load(ARTIFACT_DIR / "iso_forest_model.pkl")
    config = joblib.load(ARTIFACT_DIR / "model_config.pkl")
    return scaler, model, config

try:
    scaler, model, config = load_artifacts()
    FEATURE_COLS = config["feature_cols"]
    ROLLING_WINDOW = config["rolling_window"]
    DEFAULT_THRESHOLD = config["best_threshold"]
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False


# ---------------------------------------------------------------------------
# Feature engineering — mirrors the training notebook exactly
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    df["rolling_mean"] = df["value"].rolling(window=window).mean()
    df["rolling_std"] = df["value"].rolling(window=window).std()
    df["rolling_min"] = df["value"].rolling(window=window).min()
    df["rolling_max"] = df["value"].rolling(window=window).max()

    df["diff_1"] = df["value"].diff()
    df["pct_change"] = df["value"].pct_change()

    df["lag_1"] = df["value"].shift(1)
    df["lag_3"] = df["value"].shift(3)
    df["lag_6"] = df["value"].shift(6)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df.dropna().reset_index(drop=True)


def score_data(df_features: pd.DataFrame, threshold: float):
    X = scaler.transform(df_features[FEATURE_COLS])
    raw_score = model.decision_function(X)      # higher = more normal
    anomaly_score = -raw_score                  # flip: higher = more anomalous
    is_anomaly = (anomaly_score >= threshold).astype(int)
    df_features = df_features.copy()
    df_features["anomaly_score"] = anomaly_score
    df_features["is_anomaly"] = is_anomaly
    return df_features


# ---------------------------------------------------------------------------
# Plot helpers (Plotly, dark themed)
# ---------------------------------------------------------------------------
def plot_series(df_scored: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_scored["timestamp"], y=df_scored["value"],
        mode="lines", name="Sensor value",
        line=dict(color=TEXT_MUTED, width=1.4),
    ))
    anomalies = df_scored[df_scored["is_anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=anomalies["timestamp"], y=anomalies["value"],
        mode="markers", name="Flagged anomaly",
        marker=dict(color=GOLD, size=7, line=dict(color=GOLD_GLOW, width=1),
                    symbol="circle"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT_MUTED, family="Inter"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=TEXT, size=13),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=440,
        xaxis=dict(gridcolor="#222"),
        yaxis=dict(gridcolor="#222", title="Sensor Value"),
    )
    return fig


def plot_score_distribution(df_scored: pd.DataFrame, threshold: float):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_scored["anomaly_score"], nbinsx=60,
        marker=dict(color=GOLD, opacity=0.75),
        name="Anomaly score",
    ))
    fig.add_vline(x=threshold, line=dict(color=RED, dash="dash", width=2),
                  annotation_text="threshold", annotation_font_color=RED)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT_MUTED, family="Inter"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=320,
        xaxis=dict(title="Anomaly score (higher = more anomalous)", gridcolor="#222"),
        yaxis=dict(title="Count", gridcolor="#222"),
        showlegend=False,
    )
    return fig


def metric_card(label, value, css_class="metric-accent"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<div class='hero-title' style='font-size:1.5rem;'>⚡ <span>Control Panel</span></div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#D4D4D4; font-size:0.9rem;'>Upload sensor data to score it live against the trained Isolation Forest model.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV (timestamp, value)", type=["csv"])

    use_sample = st.button("▶ Try Sample Data", use_container_width=True)
    st.caption("No file handy? Load a bundled demo dataset with a few injected anomalies.")

    st.markdown("---")
    st.markdown("<div class='section-label'>Threshold</div>", unsafe_allow_html=True)
    if artifacts_loaded:
        threshold = st.slider(
            "Anomaly score threshold", min_value=-0.5, max_value=0.5,
            value=float(DEFAULT_THRESHOLD), step=0.005,
            help="Lower threshold = more points flagged as anomalies. Default is the F1-optimal value from evaluation."
        )
    else:
        threshold = 0.08

    st.markdown("---")
    st.markdown("""
    <div style='color:#D4D4D4; font-size:0.82rem; line-height:1.5;'>
    <b style="color:#F5B700;">Model:</b> Isolation Forest<br>
    <b style="color:#F5B700;">Features:</b> rolling stats, rate of change, lags, time features<br>
    <b style="color:#F5B700;">Trained on:</b> NAB machine temperature dataset
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("<div class='hero-title'>Anomaly Detection <span>Dashboard</span></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Unsupervised anomaly detection for time-series sensor data, "
    "powered by an Isolation Forest trained on rolling-window behavioral features. "
    "Upload any timestamped sensor CSV to score it live.</div>",
    unsafe_allow_html=True
)

if not artifacts_loaded:
    st.error(
        "Model artifacts not found. Place `scaler.pkl`, `iso_forest_model.pkl`, and "
        "`model_config.pkl` in the same folder as `app.py` (generated at the end of the training notebook)."
    )
    st.stop()

# Track whether sample mode is active across reruns (Streamlit reruns the
# whole script on every interaction, so a plain local variable isn't enough)
if use_sample:
    st.session_state["data_source"] = "sample"
elif uploaded_file is not None:
    st.session_state["data_source"] = "upload"

data_source = st.session_state.get("data_source")

if data_source is None:
    st.markdown("<div class='section-label'>Get started</div>", unsafe_allow_html=True)
    st.info("Upload a CSV with `timestamp` and `value` columns, or click **Try Sample Data** in the sidebar, to run detection.")
    st.stop()

# ---- Load + validate data ----
try:
    if data_source == "sample":
        raw_df = pd.read_csv(ARTIFACT_DIR / "sample_data.csv")
        st.markdown(
            "<p style='color:#F5B700; font-size:0.85rem; margin-bottom:1rem;'>"
            "📁 Showing results for the bundled sample dataset (synthetic, with 3 injected anomalies).</p>",
            unsafe_allow_html=True
        )
    else:
        raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if not {"timestamp", "value"}.issubset(raw_df.columns):
    st.error("CSV must contain `timestamp` and `value` columns.")
    st.stop()

raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

with st.spinner("Engineering features and scoring..."):
    df_features = engineer_features(raw_df, ROLLING_WINDOW)
    if df_features.empty:
        st.error(f"Not enough rows after feature engineering (need > {ROLLING_WINDOW} rows).")
        st.stop()
    df_scored = score_data(df_features, threshold)

# ---- KPI row ----
total_points = len(df_scored)
n_anomalies = int(df_scored["is_anomaly"].sum())
anomaly_rate = n_anomalies / total_points * 100
avg_score = df_scored["anomaly_score"].mean()

st.markdown("<div class='section-label'>Overview</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("Total Readings", f"{total_points:,}")
with c2: metric_card("Anomalies Flagged", f"{n_anomalies:,}", "metric-red" if n_anomalies else "metric-green")
with c3: metric_card("Anomaly Rate", f"{anomaly_rate:.2f}%")
with c4: metric_card("Avg. Anomaly Score", f"{avg_score:.3f}")

# ---- Time series plot ----
st.markdown("<div class='section-label'>Sensor Timeline</div>", unsafe_allow_html=True)
st.plotly_chart(plot_series(df_scored), use_container_width=True)

# ---- Score distribution ----
st.markdown("<div class='section-label'>Anomaly Score Distribution</div>", unsafe_allow_html=True)
st.plotly_chart(plot_score_distribution(df_scored, threshold), use_container_width=True)

# ---- Flagged anomalies table ----
st.markdown("<div class='section-label'>Flagged Anomalies</div>", unsafe_allow_html=True)
flagged = df_scored[df_scored["is_anomaly"] == 1][["timestamp", "value", "anomaly_score"]].sort_values(
    "anomaly_score", ascending=False
).reset_index(drop=True)

if flagged.empty:
    st.success("No anomalies flagged at the current threshold.")
else:
    st.dataframe(flagged, use_container_width=True, height=280)
    csv = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("Download flagged anomalies (CSV)", csv, "flagged_anomalies.csv", "text/csv")