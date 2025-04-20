import streamlit as st
import pandas as pd

# Styling for improved aesthetics
st.markdown("""
<style>
.center {
    text-align: center;
    font-weight: bold;
    font-size: 24px;
}
.subheader {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.boxed-text {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 15px;
    background-color: #f9f9f9;
    margin: 20px 0;
    font-size: 16px;
    line-height: 1.6;
}
.instructions {
    background-color: #eef;
    padding: 10px;
    border-radius: 10px;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="center">Prediksi Harga Penutupan Saham PT Kalbe Farma Tbk Menggunakan Model Extreme Gradient Boosting (XGBoost)</h1>', unsafe_allow_html=True)

# Pengenalan GUI
st.markdown('<p class="subheader">Sistem ini dirancang untuk memprediksi harga penutupan saham PT Kalbe Farma Tbk (KLBF) menggunakan model Extreme Gradient Boosting (XGBoost).</p>', unsafe_allow_html=True)

st.markdown("""
<div class='boxed-text'>
Sistem ini memungkinkan pengguna untuk melakukan prediksi harga penutupan saham berdasarkan berbagai parameter historis yang diperoleh dari dataset harga saham harian. Prediksi dilakukan menggunakan beberapa metode Machine Learning, yaitu:

- **XGBoost-Default** (model tanpa optimasi parameter)
- **XGBoost-GridSearchCV** (model dengan optimasi hyperparameter menggunakan GridSearchCV)
- **XGBoost-PSO** (model dengan optimasi hyperparameter menggunakan Particle Swarm Optimization)

Sistem ini memiliki antarmuka interaktif yang sederhana dan mudah digunakan untuk mendapatkan hasil prediksi secara real-time serta menampilkan visualisasi data dalam bentuk grafik interaktif.
</div>
""", unsafe_allow_html=True)

# Panduan Penggunaan
st.markdown("""
### Panduan Penggunaan
<div class='instructions'>
<b>1. Memilih Model</b><br>
Pengguna dapat memilih salah satu model yang tersedia di menu **Prediksi Saham** untuk melakukan prediksi harga saham berdasarkan model yang dipilih.

<b>2. Menginput Data</b><br>
- **Input Manual:** Pengguna dapat memasukkan harga **Open, High, Low, dan Close** secara langsung melalui formulir input.
- **Upload CSV:** Alternatif lain adalah mengunggah file CSV yang berisi data harga saham untuk diprediksi secara otomatis.

<b>3. Menampilkan Hasil Prediksi</b><br>
Setelah input data dimasukkan, sistem akan melakukan perhitungan dan menampilkan prediksi harga saham di layar. Hasil prediksi ini ditampilkan dalam bentuk tabel dan dapat diunduh untuk analisis lebih lanjut.

<b>4. Visualisasi Data</b><br>
Pada menu **Visualisasi Data**, pengguna dapat melihat grafik interaktif yang membandingkan harga aktual dengan hasil prediksi dari model yang dipilih.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
---
© 2025 - Sistem Prediksi Harga Saham KLBF | Dikembangkan untuk tujuan penelitian dan analisis data.
""")
