# Crop Yield Prediction

A machine learning project that predicts crop yield (tons/hectare) based on farm and location details, using a Random Forest Regression model trained on real Indian government agricultural data.

**Live app:** https://crop-yield-prooduction.streamlit.app/

**API:** https://machine-learning-2-15wf.onrender.com/docs

## About the project

This model predicts yield for 5 commonly grown crops — Rice, Maize, Moong (Green Gram), Urad, and Groundnut — based on:

- Crop type
- Season (Kharif, Rabi, Summer, etc.)
- State
- Land area
- Fertilizer and pesticide usage
- Year

The dataset covers crop production records from 1997-2020.

## Model performance

| Metric | Score |
|--------|-------|
| MAE    | 0.23  |
| RMSE   | 0.56  |
| R²     | 0.83  |

## How it works

1. **Model training** — Covers data cleaning, feature engineering (fertilizer/pesticide per hectare), one-hot encoding, and Random Forest training.
2. **Backend** — a FastAPI app (`backend/`) that loads the trained model and serves predictions through a `/predict` endpoint. Deployed on Render.
3. **Frontend** — a Streamlit app (`frontend/`) with a simple form for single predictions and a CSV upload option for batch predictions. Deployed on Streamlit Cloud.

## Tech stack

- Python, pandas, scikit-learn
- FastAPI
- Streamlit
- Render (backend hosting)
- Streamlit Cloud (frontend hosting)

## Project structure

```
├── notebook/
│   └── crop_yield_production.ipynb
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
└── README.md
```
