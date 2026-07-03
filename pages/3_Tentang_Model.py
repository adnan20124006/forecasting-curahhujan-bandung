import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Tentang Model", page_icon="🤖", layout="wide")

st.title("🤖 Tentang Model")
st.markdown("Informasi tentang model **Random Forest Regressor** untuk prediksi curah hujan.")
st.divider()

st.subheader("️ Konfigurasi Model")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🌳 **Algoritma**: Random Forest Regressor")
    st.info("🌲 **n_estimators**: 100 pohon")
    st.info(" **max_depth**: 10")
with col2:
    st.info("🔀 **min_samples_split**: 5")
    st.info("🍃 **min_samples_leaf**: 3")
    st.info("🎯 **max_features**: sqrt")
with col3:
    st.info("📊 **Training Data**: 81 bulan")
    st.info("🧪 **Testing Data**: 27 bulan")
    st.info(" **random_state**: 42")

st.divider()

st.subheader("📊 Metrik Evaluasi Model")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="MAE", value="50.09 MM", help="Rata-rata selisih absolut")
    st.caption("Semakin kecil = semakin baik")
with col2:
    st.metric(label="RMSE", value="58.75 MM", help="Akar dari rata-rata kuadrat error")
    st.caption("Memberi penalti lebih besar untuk error besar")
with col3:
    st.metric(label="R² Score", value="0.7651", help="Proporsi variansi yang dijelaskan model")
    st.caption("✅ Model SANGAT BAIK (R² > 0.7)")

st.divider()

st.subheader("📊 Feature Importance")
fitur_list = ['tahun', 'bulan_angka', 'lag_1', 'lag_2', 'lag_3',
              'rolling_mean_3', 'rolling_std_3', 'bulan_sin', 'bulan_cos']
importances = [0.0195, 0.1373, 0.1695, 0.0654, 0.0664, 0.2166, 0.1646, 0.0623, 0.0985]

fig, ax = plt.subplots(figsize=(12, 6))
y_pos = np.arange(len(fitur_list))
colors = plt.cm.Set2(np.linspace(0, 1, len(fitur_list)))
bars = ax.barh(y_pos, importances, color=colors, edgecolor='black', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(fitur_list, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for bar, imp in zip(bars, importances):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.info("💡 **Insight**: Fitur `rolling_mean_3` memiliki pengaruh terbesar, diikuti oleh `lag_1` dan `rolling_std_3`.")

st.divider()

st.subheader("📝 Kesimpulan")
st.markdown("""
1. **Pola Musiman**: Curah hujan Kota Bandung memiliki pola musiman yang jelas,
   dengan puncak pada November–Maret (musim hujan) dan terendah pada Juni–September (musim kemarau).

2. **Performa Model**: Model Random Forest Regressor menunjukkan performa yang **BAIK**
   dengan R² Score sebesar **0.7651**, yang berarti model mampu menjelaskan sekitar
   **76.5%** variansi dalam data curah hujan.

3. **Prediksi 2024–2027**: Model memprediksi curah hujan akan tetap mengikuti pola
   musiman historis, dengan fluktuasi yang lebih stabil dibandingkan data aktual.
""")

st.caption(" Proyek PML Kelompok 3 — Prediksi Curah Hujan Kota Bandung")