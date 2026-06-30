import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

model = joblib.load("model/churn_model.pkl")

st.title("📊 Customer Churn Predictor")
st.markdown("Predict which customers are likely to churn — single customer or bulk CSV upload.")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Upload", "📈 Feature Importance"])

# ---------------- TAB 1: Single Prediction ----------------
with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Partner = st.selectbox("Has Partner", ["No", "Yes"])
        Dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        PaymentMethod = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    if st.button("Predict Churn", type="primary"):
        # Encode inputs same way as training
        encode_map = {
            "gender": {"Female": 0, "Male": 1},
            "SeniorCitizen": {"No": 0, "Yes": 1},
            "Partner": {"No": 0, "Yes": 1},
            "Dependents": {"No": 0, "Yes": 1},
            "PhoneService": {"No": 0, "Yes": 1},
            "PaperlessBilling": {"No": 0, "Yes": 1},
            "MultipleLines": {"No phone service": 0, "No": 1, "Yes": 2},
            "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
            "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
            "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
            "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
            "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
            "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
            "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
            "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
            "PaymentMethod": {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1, 
                               "Electronic check": 2, "Mailed check": 3},
        }

        input_data = pd.DataFrame([{
            "gender": encode_map["gender"][gender],
            "SeniorCitizen": encode_map["SeniorCitizen"][SeniorCitizen],
            "Partner": encode_map["Partner"][Partner],
            "Dependents": encode_map["Dependents"][Dependents],
            "tenure": tenure,
            "PhoneService": encode_map["PhoneService"][PhoneService],
            "MultipleLines": encode_map["MultipleLines"][MultipleLines],
            "InternetService": encode_map["InternetService"][InternetService],
            "OnlineSecurity": encode_map["OnlineSecurity"][OnlineSecurity],
            "OnlineBackup": encode_map["OnlineBackup"][OnlineBackup],
            "DeviceProtection": encode_map["DeviceProtection"][DeviceProtection],
            "TechSupport": encode_map["TechSupport"][TechSupport],
            "StreamingTV": encode_map["StreamingTV"][StreamingTV],
            "StreamingMovies": encode_map["StreamingMovies"][StreamingMovies],
            "Contract": encode_map["Contract"][Contract],
            "PaperlessBilling": encode_map["PaperlessBilling"][PaperlessBilling],
            "PaymentMethod": encode_map["PaymentMethod"][PaymentMethod],
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }])

        prob = model.predict_proba(input_data)[0][1]
        pred = model.predict(input_data)[0]

        risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
        color = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Prediction", "Yes" if pred == 1 else "No")
        c2.metric("Churn Probability", f"{prob*100:.1f}%")
        c3.metric("Risk Level", f"{color} {risk}")

# ---------------- TAB 2: Batch Upload ----------------
with tab2:
    st.subheader("Upload Customer CSV")
    st.caption("CSV must already be encoded the same way as training data (numeric columns).")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)

        df['Churn_Prediction'] = preds
        df['Churn_Probability'] = (probs * 100).round(1)
        df['Risk_Level'] = df['Churn_Probability'].apply(
            lambda p: "🔴 High" if p >= 70 else "🟡 Medium" if p >= 40 else "🟢 Low"
        )

        st.success(f"Processed {len(df)} customers")
        st.dataframe(df.sort_values('Churn_Probability', ascending=False), use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button("Download Results", csv, "churn_predictions.csv", "text/csv")

# ---------------- TAB 3: Feature Importance ----------------
with tab3:
    st.subheader("What Drives Churn?")

    feature_names = model.named_steps['model'].feature_names_in_ if hasattr(model.named_steps['model'], 'feature_names_in_') else None
    coefficients = model.named_steps['model'].coef_[0]

    if feature_names is None:
        feature_names = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
                          'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                          'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                          'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']

    imp_df = pd.DataFrame({'Feature': feature_names, 'Impact': coefficients})
    imp_df = imp_df.sort_values('Impact')

    fig = px.bar(imp_df, x='Impact', y='Feature', orientation='h',
                 color='Impact', color_continuous_scale='RdBu_r',
                 title="Feature Impact on Churn (Logistic Regression Coefficients)")
    st.plotly_chart(fig, use_container_width=True)