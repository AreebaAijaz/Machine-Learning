"""
FastAPI backend for the Crop Yield Prediction model.

Loads the trained Random Forest model + the exact column list it was trained on,
then exposes a /predict endpoint that takes raw farm inputs and returns predicted yield.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Literal

app = FastAPI(title="Crop Yield Prediction API")

# allow the Streamlit app (running on a different domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load the trained model and its expected column order ----------
model = joblib.load("crop_yield_rf_model.pkl")
model_columns = joblib.load("crop_yield_model_columns.pkl")

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

YEAR_MIN = 1997  # matches df['year'].min() used during training


class PredictionInput(BaseModel):
    crop: Literal[tuple(CROPS)]           # type: ignore
    season: Literal[tuple(SEASONS)]       # type: ignore
    state: Literal[tuple(STATES)]         # type: ignore
    area: float = Field(..., gt=0, description="Land area in hectares")
    fertilizer: float = Field(..., ge=0, description="Total fertilizer used (kg)")
    pesticide: float = Field(..., ge=0, description="Total pesticide used (kg)")
    year: int = Field(..., ge=1997, le=2035, description="Crop year")


class PredictionOutput(BaseModel):
    predicted_yield: float


def build_feature_row(data: PredictionInput) -> pd.DataFrame:
    """
    Turns raw user input into the exact same feature format the model was trained on:
    per-hectare fertilizer/pesticide, years since 1997, and one-hot encoded
    crop/season/state columns (matching model_columns exactly, same order).
    """
    row = {col: 0 for col in model_columns}

    row["area"] = data.area
    row["fertilizer_per_area"] = data.fertilizer / data.area
    row["pesticide_per_area"] = data.pesticide / data.area
    row["years_since_start"] = data.year - YEAR_MIN

    # one-hot columns follow the pattern "crop_<name>", "season_<name>", "state_<name>"
    # (the first category of each was dropped during training via drop_first=True,
    # so if none of these match, it just means the row belongs to that dropped baseline category)
    crop_col = f"crop_{data.crop}"
    season_col = f"season_{data.season}"
    state_col = f"state_{data.state}"

    for col in (crop_col, season_col, state_col):
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[model_columns]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Crop Yield Prediction API is running"}


@app.get("/options")
def get_options():
    """Lets the frontend fetch valid dropdown values instead of hardcoding them twice."""
    return {"crops": CROPS, "seasons": SEASONS, "states": STATES}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        features = build_feature_row(data)
        prediction = model.predict(features)[0]
        return PredictionOutput(predicted_yield=round(float(prediction), 3))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))