import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================
# ✨ TITLE
# ===========================================
st.title("🌍 Prediksi Kategori Kedalaman Gempa")
st.write("Model menggunakan XGBoost — Cepat, Akurat, dan Stabil di Streamlit Cloud.")

# ===========================================
# 📌 LOAD MODEL
# ===========================================
scaler = joblib.load("models/scaler.pkl")
xgb_model = joblib.load("models/xgb_depth_class.pkl")

label_map = {
    0: "Shallow (<70 km)",
    1: "Intermediate (70–300 km)",
    2: "Deep (>300 km)"
}

color_map = {
    0: "blue",
    1: "orange",
    2: "red"
}

# Kolom wajib untuk prediksi
required_cols = [
    "latitude", "longitude", "mag", "gap", "dmin",
    "rms", "horizontal_error", "depth_error", "mag_error", "year"
]

# ===========================================
# 📌 FUNGSI PREDIKSI
# ===========================================
def predict_depth(df):
    scaled = scaler.transform(df)
    proba = xgb_model.predict_proba(scaled)[0]
    pred_class = np.argmax(proba)
    return label_map[pred_class], pred_class, proba

# ===========================================
# 🔎 MODE 1 — PREDIKSI SATU DATA
# ===========================================
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

# DF Input Tunggal
input_df = pd.DataFrame([[
    latitude, longitude, mag, gap, dmin,
    rms, herror, derror, magerr, year
]], columns=required_cols)

st.write("📘 **Data Input Anda:**")
st.dataframe(input_df)

if st.button("Prediksi Kedalaman Gempa"):

    hasil, pred_class, proba = predict_depth(input_df)
    st.success(f"📌 **Kategori Gempa: {hasil}**")

    # ============================================================
    # 📊 GRAFIK PROBABILITAS
    # ============================================================
    st.subheader("📊 Grafik Probabilitas Prediksi")

    kelas = ["Shallow", "Intermediate", "Deep"]
    fig, ax = plt.subplots()
    bars = ax.bar(kelas, proba, color=["blue", "orange", "red"])

    ax.set_ylabel("Probabilitas")
    ax.set_title("Distribusi Probabilitas")
    ax.bar_label(bars, fmt="%.2f")
    st.pyplot(fig)

    # ============================================================
    # 📍 SCATTER PLOT SATU TITIK
    # ============================================================
    st.subheader("📍 Scatter Plot Lokasi Gempa (Satu Data)")

    fig3, ax3 = plt.subplots(figsize=(6,4))

    ax3.scatter(
        input_df["longitude"],
        input_df["latitude"],
        s=input_df["mag"] * 30,
        c=color_map[pred_class],
        alpha=0.7,
        edgecolors="black"
    )

    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Lokasi Gempa Berdasarkan Input")
    ax3.grid(True)
    st.pyplot(fig3)


# ===========================================
# 🧮 MODE 2 — PREDIKSI RENTANG
# ===========================================
with st.sidebar:
    st.header("📊 Prediksi Dengan Rentang Parameter")

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
        ]], columns=required_cols)

        hasil_rentang, pred_class_rentang, proba_rentang = predict_depth(data_avg)

        st.write("📘 **Data Rata-Rata Rentang:**")
        st.dataframe(data_avg)

        st.success(f"📌 **Prediksi Rentang: {hasil_rentang}**")

        # Grafik probabilitas rentang
        fig2, ax2 = plt.subplots()
        bars2 = ax2.bar(kelas, proba_rentang, color=["blue", "orange", "red"])
        ax2.set_ylabel("Probabilitas")
        ax2.set_title("Distribusi Probabilitas Rentang")
        ax2.bar_label(bars2, fmt="%.2f")
        st.pyplot(fig2)

        # Scatter plot rentang
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.scatter(
            data_avg["longitude"], data_avg["latitude"],
            s=data_avg["mag"] * 30,
            c=color_map[pred_class_rentang],
            alpha=0.7,
            edgecolors="black"
        )
        ax4.set_xlabel("Longitude")
        ax4.set_ylabel("Latitude")
        ax4.set_title("Scatter Plot Lokasi (Mean dari Rentang Input)")
        ax4.grid(True)
        st.pyplot(fig4)


# ===========================================
# 📥 MODE 3 — UPLOAD CSV UNTUK PREDIKSI MASSAL
# ===========================================
st.header("📥 Upload CSV untuk Prediksi Banyak Data")

uploaded_file = st.file_uploader("Upload file CSV:", type=["csv"])

if uploaded_file is not None:
    try:
        df_csv = pd.read_csv(uploaded_file)

        st.subheader("📄 Preview CSV")
        st.dataframe(df_csv.head())

        # Cek kolom wajib
        missing = [c for c in required_cols if c not in df_csv.columns]
        if missing:
            st.error(f"❌ CSV tidak memiliki kolom wajib: {missing}")
        else:
            st.success("✔ CSV valid!")

            # Prediksi massal
            scaled_csv = scaler.transform(df_csv[required_cols])
            pred_classes = xgb_model.predict(scaled_csv)
            pred_prob = xgb_model.predict_proba(scaled_csv)

            df_csv["pred_class"] = pred_classes
            df_csv["pred_label"] = df_csv["pred_class"].map(label_map)

            st.subheader("📊 Hasil Prediksi CSV")
            st.dataframe(df_csv)

            # ============================================================
            # 📍 SCATTER PLOT CSV
            # ============================================================
            st.subheader("📍 Scatter Plot Lokasi Gempa (CSV Upload)")

            fig5, ax5 = plt.subplots(figsize=(6,4))

            ax5.scatter(
                df_csv["longitude"],
                df_csv["latitude"],
                s=df_csv["mag"] * 20,
                c=[color_map[c] for c in pred_classes],
                alpha=0.6,
                edgecolors="black"
            )

            ax5.set_xlabel("Longitude")
            ax5.set_ylabel("Latitude")
            ax5.set_title("Lokasi Gempa Berdasarkan CSV Upload")
            ax5.grid(True)
            st.pyplot(fig5)

            # ============================================================
            # 🌋 HEATMAP INTENSITAS GEMPA
            # ============================================================
            st.subheader("🌋 Heatmap Intensitas Gempa (Magnitude-Based)")

            fig_hm, ax_hm = plt.subplots(figsize=(7, 5))

            sns.kdeplot(
                x=df_csv["longitude"],
                y=df_csv["latitude"],
                weights=df_csv["mag"],
                cmap="Reds",
                bw_adjust=0.6,
                fill=True,
                thresh=0.05,
                ax=ax_hm
            )

            ax_hm.set_title("Heatmap Intensitas Gempa")
            ax_hm.set_xlabel("Longitude")
            ax_hm.set_ylabel("Latitude")

            st.pyplot(fig_hm)

            # Tombol download CSV hasil
            csv_download = df_csv.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Hasil Prediksi CSV",
                csv_download,
                "hasil_prediksi_gempa.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"❌ Terjadi error saat membaca CSV: {e}")


# ===========================================
# 📊 CONFUSION MATRIX + ACCURACY MODEL
# ===========================================
st.header("📊 Evaluasi Model (Confusion Matrix & Accuracy)")

eval_file = st.file_uploader("Upload CSV dengan label asli (true_class):", type=["csv"], key="eval")

if eval_file is not None:
    try:
        eval_df = pd.read_csv(eval_file)

        if "true_class" not in eval_df.columns:
            st.error("❌ CSV harus memiliki kolom 'true_class'")
        else:
            from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

            # Prediksi ulang
            scaled_eval = scaler.transform(eval_df[required_cols])
            y_pred = xgb_model.predict(scaled_eval)
            y_true = eval_df["true_class"]

            # Accuracy
            acc = accuracy_score(y_true, y_pred)
            st.metric("📌 Accuracy Model", f"{acc * 100:.2f}%")

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")

            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=list(label_map.values()))
            disp.plot(ax=ax_cm, cmap="Blues", colorbar=True)

            st.pyplot(fig_cm)

    except Exception as e:
        st.error(f"❌ Error evaluasi model: {e}")
