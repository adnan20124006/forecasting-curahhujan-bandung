import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA — Curah Hujan Bandung", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('model_streamlit/data_historis.csv', parse_dates=['tanggal'])
    return df

df = load_data()

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Eksplorasi data historis curah hujan Kota Bandung 2017–2023.")
st.divider()

st.sidebar.header("⚙️ Filter Data")
tahun_filter = st.sidebar.multiselect(
    "Pilih Tahun",
    options=sorted(df['tahun'].unique()),
    default=sorted(df['tahun'].unique())
)
df_filtered = df[df['tahun'].isin(tahun_filter)]

st.subheader("📋 Statistik Deskriptif")
col1, col2 = st.columns(2)

with col1:
    st.info(f"""
- **Total Data**: {len(df_filtered)} bulan
- **Periode**: {df_filtered['tanggal'].min().strftime('%b %Y')} — {df_filtered['tanggal'].max().strftime('%b %Y')}
- **Rata-rata**: {df_filtered['jumlah_curah_hujan'].mean():.1f} MM
- **Median**: {df_filtered['jumlah_curah_hujan'].median():.1f} MM
- **Std Dev**: {df_filtered['jumlah_curah_hujan'].std():.1f} MM
""")

with col2:
    st.markdown("### Distribusi Curah Hujan")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df_filtered['jumlah_curah_hujan'], bins=20, color='#2E86AB', edgecolor='black', alpha=0.7)
    ax.set_title('Distribusi Curah Hujan', fontsize=12, fontweight='bold')
    ax.set_xlabel('Curah Hujan (MM)')
    ax.set_ylabel('Frekuensi')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

st.subheader(" Visualisasi Data")
tab1, tab2, tab3 = st.tabs(["📉 Tren Waktu", "📊 Pola Musiman", "🗓️ Heatmap Tahunan"])

with tab1:
    st.markdown("### Tren Curah Hujan Bulanan")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_filtered['tanggal'], df_filtered['jumlah_curah_hujan'],
            marker='o', linestyle='-', color='#2E86AB', markersize=4, linewidth=1.5)
    ax.set_title(f'Tren Curah Hujan ({tahun_filter[0]}–{tahun_filter[-1]})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tahun', fontsize=12)
    ax.set_ylabel('Curah Hujan (MM)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    st.markdown("### Rata-rata Curah Hujan per Bulan")
    bulan_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'Mei', 6:'Jun',
                 7:'Jul', 8:'Agu', 9:'Sep', 10:'Okt', 11:'Nov', 12:'Des'}
    rata_bulanan = df_filtered.groupby('bulan_angka')['jumlah_curah_hujan'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(1, 13), rata_bulanan.values, color='#E74C3C', edgecolor='black', alpha=0.8)
    for bar, v in zip(bars, rata_bulanan.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{v:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title('Rata-rata Curah Hujan per Bulan', fontsize=14, fontweight='bold')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Rata-rata Curah Hujan (MM)')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([bulan_map[i] for i in range(1, 13)])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    st.markdown("### Heatmap Curah Hujan per Tahun dan Bulan")
    pivot = df_filtered.pivot_table(values='jumlah_curah_hujan', index='tahun', columns='bulan_angka', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5, ax=ax)
    ax.set_title('Heatmap Curah Hujan (MM)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Tahun')
    ax.set_xticklabels([bulan_map[i] for i in range(1, 13)])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()
st.subheader("📋 Tabel Data Historis")
st.dataframe(df_filtered[['tanggal', 'tahun', 'bulan_angka', 'jumlah_curah_hujan']],
             use_container_width=True, hide_index=True)

st.caption("🎓 Proyek PML Kelompok 3 — Prediksi Curah Hujan Kota Bandung")