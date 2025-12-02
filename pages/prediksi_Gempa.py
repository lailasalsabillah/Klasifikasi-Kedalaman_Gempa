import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# TITLE
# -------------------------------
st.title("🌍 Prediksi Kategori Kedalaman Gempa")
st.write("Model menggunakan XGBoost (versi ringan untuk Streamlit Cloud)")


# -------------------------------
# LOAD MODEL
# -------------------------------
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

# -------------------------------
# FUNGSI PREDIKSI
# -------------------------------
def predict_depth(df):
    scaled = scaler.transform(df)
    pred = xgb_model.predict(scaled)[0]
    return label_map[pred]


# -------------------------------
# INP
