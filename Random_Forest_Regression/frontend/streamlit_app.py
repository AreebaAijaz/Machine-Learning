"""
Streamlit frontend for the Crop Yield Prediction project.

Sends farm details to the FastAPI backend and shows the predicted yield,
along with how it compares to the typical range for that crop.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Config ----------
API_URL = "https://machine-learning-2-15wf.onrender.com" # replace with deployed backend URL

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

# typical yield range per crop, used only to show where a prediction sits (from training data)
CROP_REFERENCE = {
    "Rice":              {"mean": 2.22, "min": 0.02, "max": 8.78},
    "Maize":             {"mean": 2.41, "min": 0.00, "max": 22.38},
    "Moong(Green Gram)": {"mean": 0.53, "min": 0.00, "max": 1.77},
    "Urad":              {"mean": 0.58, "min": 0.00, "max": 4.19},
    "Groundnut":         {"mean": 1.36, "min": 0.20, "max": 3.67},
}

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")

st.title("Crop Yield Predictor")
st.caption("Random Forest model trained on Indian government crop production data (1997-2020)")

tab_predict, tab_batch, tab_about = st.tabs(["Predict", "Batch upload", "About this model"])

# ---------- Tab 1: single prediction ----------
with tab_predict:
    st.subheader("Enter farm details")

    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", CROPS)
        season = st.selectbox("Season", SEASONS)
        state = st.selectbox("State", STATES)
        year = st.number_input("Year", min_value=1997, max_value=2035, value=2024)

    with col2:
        area = st.number_input("Area (hectares)", min_value=0.1, value=10.0, step=0.5)
        fertilizer = st.number_input("Total fertilizer used (kg)", min_value=0.0, value=1500.0, step=50.0)
        pesticide = st.number_input("Total pesticide used (kg)", min_value=0.0, value=5.0, step=0.5)

    if st.button("Predict yield", type="primary"):
        payload = {
            "crop": crop, "season": season, "state": state,
            "area": area, "fertilizer": fertilizer, "pesticide": pesticide, "year": int(year),
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()["predicted_yield"]

            st.metric("Predicted yield", f"{result} tons/hectare")

            # comparison chart: where does this prediction sit vs the typical range for this crop
            ref = CROP_REFERENCE[crop]
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh([0], [ref["max"]], color="#e0e0e0")
            ax.axvline(ref["mean"], color="gray", linestyle="--", label="Typical average")
            ax.axvline(result, color="green", linewidth=3, label="Your prediction")
            ax.set_yticks([])
            ax.set_xlabel("Yield (tons/hectare)")
            ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig)

        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the prediction API: {e}")

# ---------- Tab 2: batch prediction via CSV ----------
with tab_batch:
    st.subheader("Upload a CSV for multiple predictions")
    st.caption("Columns required: crop, season, state, area, fertilizer, pesticide, year")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head())

        if st.button("Run batch prediction"):
            predictions = []
            progress = st.progress(0)
            for i, row in batch_df.iterrows():
                payload = row[["crop", "season", "state", "area", "fertilizer", "pesticide", "year"]].to_dict()
                payload["year"] = int(payload["year"])
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    response.raise_for_status()
                    predictions.append(response.json()["predicted_yield"])
                except requests.exceptions.RequestException:
                    predictions.append(None)
                progress.progress((i + 1) / len(batch_df))

            batch_df["predicted_yield"] = predictions
            st.dataframe(batch_df)
            st.download_button(
                "Download results as CSV",
                batch_df.to_csv(index=False),
                file_name="crop_yield_predictions.csv",
            )

# ---------- Tab 3: about ----------
with tab_about:
    st.subheader("About this model")
    st.markdown("""
    This model predicts crop yield (tons/hectare) using a **Random Forest Regressor**,
    trained on Indian government agricultural data narrowed down to 5 commonly grown crops:
    Rice, Maize, Moong (Green Gram), Urad, and Groundnut.

    **Model performance on held-out test data:**
    - MAE: 0.23 tons/hectare
    - RMSE: 0.56 tons/hectare
    - R²: 0.83

    **Features used:** crop, season, state, land area, fertilizer and pesticide use per hectare,
    and year.
    """)