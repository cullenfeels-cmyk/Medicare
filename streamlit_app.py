import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Medicare AI Portal", page_icon="🏥", layout="wide")

st.title("🏥 Medicare AI - Healthcare Portal")
st.write("Welcome to your live AI health diagnosis portal!")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "disease_encoder.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "Cleaned_Dataset.csv")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None
        feature_columns = joblib.load(FEATURE_PATH) if os.path.exists(FEATURE_PATH) else []
        dataset_df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None
        return model, encoder, feature_columns, dataset_df
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, [], None

model, encoder, feature_columns, dataset_df = load_assets()

if model is None:
    st.warning("Models are loading or missing. Please check your 'models/' folder on GitHub.")
else:
    st.success("Models loaded successfully!")
    
    # Simple interactive test form
    age = st.slider("Age", 1, 100, 30)
    symptom_options = [col for col in feature_columns if col not in ["age", "gender", "blood_pressure", "cholesterol"]]
    selected = st.multiselect("Select Symptoms", symptom_options)
    
    if st.button("Run Test Prediction"):
        st.info(f"Running prediction for age {age} with {len(selected)} symptoms.")