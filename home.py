
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Prediksi Curah Hujan Bandung",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df_historis = pd.read_csv('data_historis.csv', parse_dates=['tanggal'])
    df_prediksi = pd.read_csv('semua_prediksi.csv', parse_dates=['tanggal'])
    return df_historis, df_prediksi

df_historis, df_prediksi = load_data()

st.title("️ Prediksi Curah Hujan Kota Bandung")
st.markdown("""
Selamat datang di aplikasi **Prediksi Curah Hujan Kota Bandung 2024–2027**!

Aplikasi ini menggunakan algoritma **Random Forest Regressor** untuk memprediksi
curah hujan bulanan di Kota Bandung berdasarkan data historis 2017–2023.
""")

st.divider()

st.subheader("📊 Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Data Historis", value=f"{len(df_historis)} bulan")
with col2:
    st.metric(label="Data Prediksi", value=f"{len(df_prediksi)} bulan")
with col3:
    avg_hujan = df_historis['jumlah_curah_hujan'].mean()
    st.metric(label="Rata-rata Curah Hujan", value=f"{avg_hujan:.1f} MM")
with col4:
    max_hujan = df_historis['jumlah_curah_hujan'].max()
    st.metric(label="Curah Hujan Tertinggi", value=f"{max_hujan:.1f} MM")

st.divider()

st.subheader("📈 Tren Curah Hujan Historis (2017–2023)")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_historis['tanggal'], df_historis['jumlah_curah_hujan'],
        marker='o', linestyle='-', color='#2E86AB', markersize=4, linewidth=1.5)
ax.set_title('Curah Hujan Bulanan Kota Bandung (2017–2023)', fontsize=14, fontweight='bold')
ax.set_xlabel('Tahun', fontsize=12)
ax.set_ylabel('Curah Hujan (MM)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

st.subheader(" Navigasi")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("### 📊 EDA")
    st.markdown("Eksplorasi data historis, pola musiman, dan distribusi curah hujan.")
    st.page_link("pages/1_EDA.py", label="Buka halaman EDA →", icon="📊")

with nav_col2:
    st.markdown("### 🎯 Prediksi")
    st.markdown("Lihat hasil prediksi curah hujan 2024–2027.")
    st.page_link("pages/2_Prediksi.py", label="Buka halaman Prediksi →", icon="🎯")

with nav_col3:
    st.markdown("###  Tentang Model")
    st.markdown("Informasi model Random Forest dan metrik evaluasi.")
    st.page_link("pages/3_Tentang_Model.py", label="Buka halaman Model →", icon="🤖")

st.divider()
st.caption("🎓 Proyek PML Kelompok 3 — Prediksi Curah Hujan Kota Bandung")