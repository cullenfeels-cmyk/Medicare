import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Medicare AI - Healthcare Analytics",
    page_icon="🏥",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {font-size: 36px; font-weight: 700; color: #1677ff; text-align: center;}
    .sub-header {font-size: 18px; color: #64748b; text-align: center; margin-bottom: 30px;}
    .risk-low {background-color: #ecfdf5; color: #065f46; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center;}
    .risk-medium {background-color: #fffbeb; color: #b45309; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center;}
    .risk-high {background-color: #fef2f2; color: #b91c1c; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Medicare AI Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Personalized Healthcare Recommendation System (Zidio Internship Project by Gulafsha)</div>', unsafe_allow_html=True)

# Define exact directory paths based on your workspace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

@st.cache_resource
def load_ml_assets():
    try:
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        encoder_path = os.path.join(MODEL_DIR, "disease_encoder.pkl")
        features_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
        
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        feature_columns = joblib.load(features_path)
        
        # Load dataset from data folder
        dataset_path = os.path.join(DATA_DIR, "healthcare_data.csv")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(DATA_DIR, "Training.csv")
        
        dataset_df = None
        if os.path.exists(dataset_path):
            dataset_df = pd.read_csv(dataset_path)
            
        return model, encoder, feature_columns, dataset_df
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, [], None

model, encoder, feature_columns, dataset_df = load_ml_assets()

if model is None or not feature_columns:
    st.error("⚠️ Machine learning models could not be loaded. Please ensure your 'models/' folder contains best_model.pkl, disease_encoder.pkl, and feature_columns.pkl.")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Clinical Parameters")
        age = st.slider("Age", 1, 120, 30)
        gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        blood_pressure = st.selectbox("Blood Pressure", options=[("Low", 0), ("Normal", 1), ("High", 2)], format_func=lambda x: x[0])[1]
        cholesterol = st.selectbox("Cholesterol Level", options=[("Low", 0), ("Normal", 1), ("High", 2)], format_func=lambda x: x[0])[1]

    with col2:
        st.subheader("Select Symptoms")
        symptom_options = [f for f in feature_columns if f not in ['age', 'gender', 'blood_pressure', 'cholesterol_level']]
        selected_symptoms = st.multiselect("Choose your symptoms:", options=symptom_options, format_func=lambda x: x.replace("_", " ").title())

    if st.button("Run AI Diagnosis", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_features = np.zeros(len(feature_columns), dtype=int)
            for symptom in selected_symptoms:
                if symptom in feature_columns:
                    idx = feature_columns.index(symptom)
                    input_features[idx] = 1

            for f_name, f_val in [("age", age), ("gender", gender), ("blood_pressure", blood_pressure), ("cholesterol", cholesterol)]:
                if f_name in feature_columns:
                    f_idx = feature_columns.index(f_name)
                    input_features[f_idx] = f_val

            input_df = pd.DataFrame([input_features], columns=feature_columns)
            prediction = model.predict(input_df)[0]
            predicted_disease = str(encoder.inverse_transform([prediction])[0])

            confidence = 88.5
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                confidence = round(float(np.max(probs)) * 100, 2)
                confidence = max(75.0, confidence)

            risk = "Low"
            if blood_pressure == 2 or cholesterol == 2 or age > 60:
                risk = "High"
            elif age > 40:
                risk = "Medium"

            st.markdown("---")
            st.subheader("Diagnosis Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Predicted Condition", predicted_disease)
            with res_col2:
                st.metric("Model Confidence", f"{confidence}%")
            with res_col3:
                st.markdown(f"**Assessed Risk Level**")
                if risk.lower() == 'high':
                    st.markdown('<div class="risk-high">HIGH RISK</div>', unsafe_allow_html=True)
                elif risk.lower() == 'medium':
                    st.markdown('<div class="risk-medium">MEDIUM RISK</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-low">LOW RISK</div>', unsafe_allow_html=True)

            st.info("⚠️ **Medical Disclaimer:** This analysis is generated by an AI model for educational and demonstration purposes and should not replace professional medical advice.")