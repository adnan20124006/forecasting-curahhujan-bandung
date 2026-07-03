%%writefile forecasting-curahhujan-bandung-apps/pages/2_Prediksi.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi — Curah Hujan Bandung", page_icon="", layout="wide")

@st.cache_data
def load_data():
    df_historis = pd.read_csv('data_historis.csv', parse_dates=['tanggal'])
    df_prediksi = pd.read_csv('semua_prediksi.csv', parse_dates=['tanggal'])
    return df_historis, df_prediksi

df_historis, df_prediksi = load_data()

st.title("🎯 Prediksi Curah Hujan 2024–2027")
st.markdown("Hasil prediksi curah hujan Kota Bandung menggunakan model Random Forest Regressor.")
st.divider()

st.sidebar.header("⚙️ Filter Prediksi")
tahun_prediksi = st.sidebar.multiselect(
    "Pilih Tahun Prediksi",
    options=sorted(df_prediksi['tahun'].unique()),
    default=sorted(df_prediksi['tahun'].unique())
)
df_prediksi_filtered = df_prediksi[df_prediksi['tahun'].isin(tahun_prediksi)]

st.subheader("📈 Perbandingan Data Historis vs Prediksi")
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(df_historis['tanggal'], df_historis['jumlah_curah_hujan'],
        label='Data Historis (2017–2023)', color='#2E86AB', linewidth=2, marker='o', markersize=3)
ax.plot(df_prediksi_filtered['tanggal'], df_prediksi_filtered['prediksi_curah_hujan'],
        label='Prediksi (2024–2027)', color='#E74C3C', linewidth=2, linestyle='--', marker='s', markersize=3)
ax.axvline(pd.Timestamp('2024-01-01'), color='black', linestyle=':', linewidth=2, label='Batas Historis/Prediksi')
ax.set_title('Pola Curah Hujan Kota Bandung (2017–2027)', fontsize=15, fontweight='bold')
ax.set_xlabel('Tahun', fontsize=12)
ax.set_ylabel('Curah Hujan (MM)', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

for tahun in sorted(df_prediksi_filtered['tahun'].unique()):
    st.subheader(f"📅 Prediksi Tahun {tahun}")
    df_tahun = df_prediksi_filtered[df_prediksi_filtered['tahun'] == tahun]
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(df_tahun[['tanggal', 'bulan', 'prediksi_curah_hujan']],
                     use_container_width=True, hide_index=True)
    with col2:
        st.metric("Rata-rata", f"{df_tahun['prediksi_curah_hujan'].mean():.1f} MM")
        st.metric("Tertinggi", f"{df_tahun['prediksi_curah_hujan'].max():.1f} MM")
        st.metric("Terendah", f"{df_tahun['prediksi_curah_hujan'].min():.1f} MM")
    st.divider()

st.caption("🎓 Proyek PML Kelompok 3 — Prediksi Curah Hujan Kota Bandung")