import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------------
# TITLE
# --------------------------------
st.title("🌍 Prediksi Kategori Kedalaman Gempa")
st.write("Model menggunakan XGBoost (versi ringan untuk Streamlit Cloud)")


# --------------------------------
# LOAD MODEL
# --------------------------------
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}


# --------------------------------
# FUNGSI PREDIKSI
# --------------------------------
def predict_depth(df):
    scaled = scaler.transform(df)
    proba = xgb_model.predict_proba(scaled)[0]
    pred_class = np.argmax(proba)
    return label_map[pred_class], proba


# --------------------------------
# INPUT MODE 1 – PREDIKSI SATU DATA
# --------------------------------
st.header("🔎 Prediksi Kedalaman Gempa (Input Satu Data)")

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", -20.0, 20.0, 0.0)
    longitude = st.number_input("Longitude", 80.0, 150.0, 100.0)
    mag = st.number_input("Magnitude", 3.0, 10.0, 5.0)
    gap = st.number_input("Gap", 0, 300, 40)
    dmin = st.number_input("Dmin", 0.0, 30.0, 2.0)

with col2:
    rms = st.number_input("RMS", 0.0, 3.0, 1.0)
    herror = st.number_input("Horizontal Error", 0.0, 50.0, 5.0)
    derror = st.number_input("Depth Error", 0.0, 30.0, 5.0)
    magerr = st.number_input("Magnitude Error", 0.0, 1.0, 0.1)
    year = st.number_input("Year", 2000, 2030, 2020)


# DataFrame input
input_df = pd.DataFrame([[
    latitude, longitude, mag, gap, dmin,
    rms, herror, derror, magerr, year
]], columns=[
    "latitude", "longitude", "mag", "gap", "dmin",
    "rms", "horizontal_error", "depth_error", "mag_error", "year"
])

st.write("📘 **Data Input Anda:**")
st.dataframe(input_df)


# --------------------------------
# PREDIKSI + GRAFIK PROBABILITAS + SCATTER
# --------------------------------
if st.button("Prediksi Kedalaman Gempa"):

    hasil, proba = predict_depth(input_df)
    st.success(f"📌 **Hasil Prediksi: {hasil}**")

    # -----------------------------
    # GRAFIK PROBABILITAS
    # -----------------------------
    st.subheader("📊 Grafik Probabilitas Kedalaman Gempa")

    kelas = ["Shallow", "Intermediate", "Deep"]

    fig, ax = plt.subplots()
    bars = ax.bar(kelas, proba)

    ax.set_ylabel("Probabilitas")
    ax.set_title("Distribusi Probabilitas Prediksi")
    ax.bar_label(bars, fmt="%.2f")

    st.pyplot(fig)

    # -----------------------------
    # SCATTER PLOT LOKASI GEMPA
    # -----------------------------
    st.subheader("📍 Scatter Plot Lokasi Gempa")

    color_map = {
        "Shallow (<70 km)": "blue",
        "Intermediate (70–300 km)": "orange",
        "Deep (>300 km)": "red"
    }

    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.scatter(
        input_df["longitude"],
        input_df["latitude"],
        s=input_df["mag"] * 30,             # ukuran titik berdasarkan magnitude
        c=color_map[hasil],
        alpha=0.7,
        edgecolors="black"
    )

    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Scatter Plot Lokasi Gempa")
    ax3.grid(True)

    st.pyplot(fig3)


# --------------------------------
# MODE RENTANG (SIDEBAR)
# --------------------------------
with st.sidebar:
    st.header("📊 Prediksi Rentang Parameter")

    lat_range = st.slider("Latitude", -20.0, 20.0, (-10.0, 10.0))
    lon_range = st.slider("Longitude", 80.0, 150.0, (100.0, 120.0))
    mag_range = st.slider("Magnitude", 3.0, 10.0, (4.0, 6.0))
    gap_range = st.slider("Gap", 0, 300, (20, 80))
    dmin_range = st.slider("Dmin", 0.0, 30.0, (1.0, 5.0))
    rms_range = st.slider("RMS", 0.0, 3.0, (0.5, 1.5))
    herror_range = st.slider("Horizontal Error", 0.0, 50.0, (5.0, 10.0))
    derror_range = st.slider("Depth Error", 0.0, 30.0, (3.0, 8.0))
    magerr_range = st.slider("Magnitude Error", 0.0, 1.0, (0.05, 0.2))
    year_range = st.slider("Year", 2000, 2030, (2015, 2025))

    if st.button("Prediksi Dari Rentang"):

        data_avg = pd.DataFrame([[
            np.mean(lat_range), np.mean(lon_range), np.mean(mag_range),
            np.mean(gap_range), np.mean(dmin_range), np.mean(rms_range),
            np.mean(herror_range), np.mean(derror_range),
            np.mean(magerr_range), np.mean(year_range)
        ]], columns=[
            "latitude", "longitude", "mag", "gap", "dmin",
            "rms", "horizontal_error", "depth_error", "mag_error", "year"
        ])

        hasil_rentang, proba_rentang = predict_depth(data_avg)

        st.write("📘 **Data Rata-Rata Rentang:**")
        st.dataframe(data_avg)

        st.success(f"📌 **Prediksi Rentang: {hasil_rentang}**")

        # Grafik probabilitas rentang
        st.subheader("📊 Grafik Probabilitas Rentang")

        fig2, ax2 = plt.subplots()
        bars2 = ax2.bar(kelas, proba_rentang)
        ax2.set_ylabel("Probabilitas")
        ax2.set_title("Distribusi Probabilitas Prediksi Rentang")
        ax2.bar_label(bars2, fmt="%.2f")
        st.pyplot(fig2)

        # Scatter plot rentang
        st.subheader("📍 Scatter Plot Lokasi Gempa (Rata-Rata Rentang)")

        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.scatter(
            data_avg["longitude"], data_avg["latitude"],
            s=data_avg["mag"] * 30,
            c=color_map[hasil_rentang],
            alpha=0.7,
            edgecolors="black"
        )

        ax4.set_xlabel("Longitude")
        ax4.set_ylabel("Latitude")
        ax4.set_title("Scatter Plot Lokasi Gempa (Mean dari Rentang Input)")
        ax4.grid(True)

        st.pyplot(fig4)
