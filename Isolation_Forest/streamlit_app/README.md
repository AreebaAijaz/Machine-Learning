# Anomaly Detection in IoT Sensor Data 🌡️⚡

A machine learning project that detects unusual behavior in machine temperature sensor data — the kind of tool that could warn engineers *before* a machine breaks down, not after.

**Live demo:** [add your Streamlit Cloud link here]
---

## What this project does

Imagine a sensor on a machine that records its temperature every 5 minutes, 24/7. Most of the time, the temperature stays in a normal range. But sometimes, something goes wrong — a slow overheating, a sudden crash, a warning sign before a full breakdown.

This project builds a system that automatically spots those unusual moments — **without ever being told what "unusual" looks like**. That's the core challenge here: there are no labels to learn from during training. The model has to figure out on its own what "normal" looks like, and flag anything that doesn't fit.

This is called **unsupervised anomaly detection**, and it's exactly how real-world industrial monitoring systems work, since failures are rare and rarely labeled in advance.

---

## The dataset

I used the **NAB (Numenta Anomaly Benchmark)** dataset — specifically `machine_temperature_system_failure.csv`, which contains real temperature readings from an industrial machine over about 2.5 months.

This dataset is special because it includes 4 real, documented events:
1. A planned shutdown
2. A sharp, sudden failure
3. A subtle early-warning drift (the hardest one to catch — it looks almost like normal noise)
4. A catastrophic failure, which happened because the early warning (#3) went unnoticed

That story — a missed warning leading to a bigger failure — is exactly the kind of problem anomaly detection is built to solve.

---

## How it works

### 1. Feature engineering
A single temperature reading doesn't say much on its own. So instead of just looking at the raw value, I built extra features that describe the *behavior* around each point:
- Rolling averages and volatility (last 1 hour)
- How fast the value is changing
- Recent past values (lag features)
- Time-of-day and day-of-week patterns

This gives the model a much richer picture — not just "what is the temperature," but "is this behaving strangely compared to its recent pattern."

### 2. Two models, compared
- **Isolation Forest** — flags points that are easy to "isolate" from the rest of the data using random splits. Anomalies tend to be easy to isolate; normal points don't.
- **K-Means** — groups the data into clusters of "normal" behavior, then flags points that sit far away from every cluster.

Both were trained the same way, then compared fairly using precision, recall, and F1 score against the 4 known real anomaly events.

### 3. Result
**Isolation Forest performed better** and was chosen as the final model. After tuning the decision threshold (instead of guessing), F1 score improved by about 80% over the initial baseline.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Isolation Forest (tuned) | 0.41 | 0.39 | **0.40** |
| Isolation Forest (baseline) | 0.14 | 0.50 | 0.22 |
| K-Means (baseline) | 0.14 | 0.48 | 0.21 |

**Why Isolation Forest won:** K-Means assumes normal behavior forms neat, round clusters — but real machine behavior isn't always that tidy, especially during a slow, drifting failure. Isolation Forest doesn't make that assumption, so it handles irregular patterns better.

---

## The dashboard

Instead of leaving results buried in a notebook, I built a **Streamlit web app** where anyone can upload their own sensor data (timestamp + value) and see it scored live — no coding required.

**Features:**
- Drag-and-drop CSV upload, or try the built-in sample dataset with one click
- Live anomaly scoring using the trained model
- Interactive timeline chart with anomalies highlighted
- Adjustable sensitivity threshold
- Downloadable list of flagged anomalies

---

## Tech stack

- **Python** — pandas, numpy, scikit-learn
- **Isolation Forest & K-Means** — scikit-learn
- **Streamlit** — web dashboard
- **Plotly** — interactive charts

---

## Project structure

```
├── Anomaly_Detection_in_IoT_Sensor_Data.ipynb   # full analysis: EDA → features → models → evaluation
├── streamlit_app/
│   ├── app.py                 # dashboard app
│   ├── requirements.txt
│   ├── sample_data.csv        # demo dataset for testing the app
│   ├── scaler.pkl             # trained feature scaler
│   ├── iso_forest_model.pkl   # trained Isolation Forest model
│   ├── model_config.pkl       # feature list, rolling window size, threshold
│   └── .streamlit/config.toml # dashboard theme
└── README.md
```

---

## Running it yourself

**Notebook:**
Open `Anomaly_Detection_in_IoT_Sensor_Data.ipynb` in Google Colab or Jupyter and run all cells top to bottom.

**Dashboard:**
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## What I learned building this

- Unsupervised models need to be evaluated carefully — accuracy is meaningless when anomalies are rare, so precision/recall/F1 (and the precision-recall curve) matter much more.
- Picking a good decision threshold isn't a guess — it can be optimized directly from data, and it made a huge difference here.
- Real anomaly data is messy: some flagged points weren't part of the "official" labeled events but were still statistically unusual. That's not a bug — it's a realistic tension in anomaly detection, and worth explaining rather than hiding.

---

## Author

**Areeba Aijaz**
Automation Engineer & ML Learner
