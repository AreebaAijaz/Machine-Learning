"""
Streamlit frontend for the Crop Yield Prediction project.

Loads the trained model directly (instead of calling the FastAPI backend) so
predictions are instant, with no cold-start delay from a sleeping free-tier API.
The FastAPI backend still exists separately as the project's API layer.
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ---------- Config ----------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_yield_rf_model.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "crop_yield_model_columns.pkl")

YEAR_MIN = 1997  # matches df['year'].min() used during training

CROPS = ["Rice", "Maize", "Moong(Green Gram)", "Urad", "Groundnut"]
SEASONS = ["Autumn", "Kharif", "Rabi", "Summer", "Whole Year", "Winter"]
STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Delhi", "Goa", "Gujarat", "Haryana", "Himachal Pradesh",
    "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
    "Puducherry", "Punjab", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
]

# typical yield range per crop, used to place a prediction in context (from training data)
CROP_REFERENCE = {
    "Rice":              {"mean": 2.22, "min": 0.02, "max": 8.78},
    "Maize":             {"mean": 2.41, "min": 0.00, "max": 22.38},
    "Moong(Green Gram)": {"mean": 0.53, "min": 0.00, "max": 1.77},
    "Urad":              {"mean": 0.58, "min": 0.00, "max": 4.19},
    "Groundnut":         {"mean": 1.36, "min": 0.20, "max": 3.67},
}

CROP_ICON = {
    "Rice": "🌾", "Maize": "🌽", "Moong(Green Gram)": "🫛",
    "Urad": "🫘", "Groundnut": "🥜",
}

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")


# ---------- Load model once, cached across reruns ----------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, model_columns


def build_feature_row(crop, season, state, area, fertilizer, pesticide, year, model_columns):
    """
    Turns raw form input into the exact same feature format the model was trained on:
    per-hectare fertilizer/pesticide, years since 1997, and one-hot encoded
    crop/season/state columns (matching model_columns exactly, same order).
    """
    row = {col: 0 for col in model_columns}

    row["area"] = area
    row["fertilizer_per_area"] = fertilizer / area
    row["pesticide_per_area"] = pesticide / area
    row["years_since_start"] = year - YEAR_MIN

    for col in (f"crop_{crop}", f"season_{season}", f"state_{state}"):
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[model_columns]


try:
    model, model_columns = load_model()
    model_load_error = None
except Exception as e:
    model, model_columns = None, None
    model_load_error = str(e)


# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.cyp-hero {
    padding: 28px 32px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1F4D36 0%, #2E6B4C 100%);
    margin-bottom: 28px;
}
.cyp-hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: #FAF6EC;
    margin: 0 0 6px 0;
}
.cyp-hero p {
    color: #D9E8DE;
    font-size: 0.95rem;
    margin: 0;
}

.cyp-card {
    background: #FFFFFF;
    border: 1px solid #E4DCC8;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 20px;
}
.cyp-card h3 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: #1F4D36;
    margin-top: 0;
}

.cyp-result {
    background: #FFFFFF;
    border: 1px solid #E4DCC8;
    border-left: 5px solid #C98A2C;
    border-radius: 14px;
    padding: 24px 26px;
    margin-top: 8px;
}
.cyp-result .eyebrow {
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8A8368;
    font-weight: 600;
}
.cyp-result .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    color: #1F4D36;
    margin: 4px 0 2px 0;
}
.cyp-result .unit {
    font-size: 1rem;
    color: #6B6552;
    font-weight: 500;
}

.cyp-bar-track {
    position: relative;
    height: 34px;
    background: #F1ECDD;
    border-radius: 8px;
    margin-top: 18px;
    overflow: hidden;
}
.cyp-bar-fill {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    background: linear-gradient(90deg, #C98A2C, #E3AC52);
    border-radius: 8px 0 0 8px;
}
.cyp-bar-avg {
    position: absolute;
    top: -4px; bottom: -4px;
    width: 2px;
    background: #1F4D36;
}
.cyp-bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #8A8368;
    margin-top: 6px;
    font-family: 'IBM Plex Mono', monospace;
}
.cyp-legend {
    display: flex;
    gap: 18px;
    font-size: 0.8rem;
    color: #6B6552;
    margin-top: 10px;
}
.cyp-legend span.dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: 5px;
}

div.stButton > button {
    background: #1F4D36;
    color: #FAF6EC;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #2E6B4C;
    color: #FFFFFF;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: #6B6552;
}
.stTabs [aria-selected="true"] {
    color: #1F4D36 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #1F4D36 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: #E4DCC8 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Hero header ----------
st.markdown("""
<div class="cyp-hero">
    <h1>🌾 Crop Yield Predictor</h1>
    <p>Random Forest model trained on Indian government crop production data (1997–2020)</p>
</div>
""", unsafe_allow_html=True)

if model_load_error:
    st.error(
        "Could not load the model file. Make sure 'crop_yield_rf_model.pkl' and "
        "'crop_yield_model_columns.pkl' are inside the 'model/' folder next to this app.\n\n"
        f"Details: {model_load_error}"
    )
    st.stop()

tab_predict, tab_batch, tab_about = st.tabs(["Predict", "Batch upload", "About this model"])

# ---------- Tab 1: single prediction ----------
with tab_predict:
    st.markdown('<div class="cyp-card"><h3>Farm details</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", CROPS, format_func=lambda c: f"{CROP_ICON[c]}  {c}")
        season = st.selectbox("Season", SEASONS)
        state = st.selectbox("State", STATES)
        year = st.number_input("Year", min_value=1997, max_value=2035, value=2024)

    with col2:
        area = st.number_input("Area (hectares)", min_value=0.1, value=10.0, step=0.5)
        fertilizer = st.number_input("Total fertilizer used (kg)", min_value=0.0, value=1500.0, step=50.0)
        pesticide = st.number_input("Total pesticide used (kg)", min_value=0.0, value=5.0, step=0.5)

    predict_clicked = st.button("Predict yield", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        try:
            features = build_feature_row(crop, season, state, area, fertilizer, pesticide, int(year), model_columns)
            result = round(float(model.predict(features)[0]), 3)

            ref = CROP_REFERENCE[crop]
            fill_pct = max(0, min(100, (result / ref["max"]) * 100))
            avg_pct = max(0, min(100, (ref["mean"] / ref["max"]) * 100))

            st.markdown(f"""
            <div class="cyp-result">
                <div class="eyebrow">Predicted yield &middot; {crop}</div>
                <div class="value">{result} <span class="unit">tons / hectare</span></div>
                <div class="cyp-bar-track">
                    <div class="cyp-bar-fill" style="width:{fill_pct}%;"></div>
                    <div class="cyp-bar-avg" style="left:{avg_pct}%;"></div>
                </div>
                <div class="cyp-bar-labels">
                    <span>0</span><span>{ref['max']} (max observed)</span>
                </div>
                <div class="cyp-legend">
                    <div><span class="dot" style="background:#C98A2C;"></span>Your prediction</div>
                    <div><span class="dot" style="background:#1F4D36;"></span>Typical average ({ref['mean']})</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Couldn't generate a prediction: {e}")

# ---------- Tab 2: batch prediction via CSV ----------
with tab_batch:
    st.markdown('<div class="cyp-card"><h3>Batch prediction</h3>', unsafe_allow_html=True)
    st.caption("Upload a CSV with columns: crop, season, state, area, fertilizer, pesticide, year")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Run batch prediction"):
            predictions = []
            progress = st.progress(0)
            for i, row in batch_df.iterrows():
                try:
                    features = build_feature_row(
                        row["crop"], row["season"], row["state"],
                        float(row["area"]), float(row["fertilizer"]), float(row["pesticide"]),
                        int(row["year"]), model_columns,
                    )
                    predictions.append(round(float(model.predict(features)[0]), 3))
                except Exception:
                    predictions.append(None)
                progress.progress((i + 1) / len(batch_df))

            batch_df["predicted_yield"] = predictions
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                batch_df.to_csv(index=False),
                file_name="crop_yield_predictions.csv",
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 3: about ----------
with tab_about:
    st.markdown('<div class="cyp-card"><h3>About this model</h3>', unsafe_allow_html=True)
    st.markdown("""
    This model predicts crop yield (tons/hectare) using a **Random Forest Regressor**,
    trained on Indian government agricultural data narrowed down to 5 commonly grown crops:
    Rice, Maize, Moong (Green Gram), Urad, and Groundnut.

    This app loads the trained model directly for instant predictions. A separate FastAPI
    backend also exposes the same model through a REST API.

    **Model performance on held-out test data:**
    - MAE: 0.23 tons/hectare
    - RMSE: 0.56 tons/hectare
    - R²: 0.83

    **Features used:** crop, season, state, land area, fertilizer and pesticide use per hectare,
    and year.
    """)
    st.markdown('</div>', unsafe_allow_html=True)