import streamlit as st
import pandas as pd

# Styling for alignment and layout adjustments
st.markdown("""
<style>
.center {
    text-align: center;
}
.adjusted-left {
    margin-left: 20%; /* Geser sedikit ke tengah kiri */
    margin-top: 20px;
    line-height: 1.8;
    text-align: center;
}
.adjusted-right {
    margin-right: 20%; /* Geser sedikit ke tengah kanan */
    margin-top: 20px;
    line-height: 1.8;
    text-align: center;
}
.space-below {
    margin-bottom: 5px;
}
.space-between {
    margin-top: 20px; /* Space between NIM and dosen pembimbing */
}
.space-name {
    margin-top: 10px; /* Tambahkan jarak antara judul dan nama */
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# Logo placement
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('logo.png', width=200,)

with col3:
    st.write(' ')

# Title and information
st.markdown('<h1 class="center">Prediksi Harga Penutupan Saham PT Kalbe Farma Tbk Menggunakan Model Extreme Gradient Boosting (XGBoost)</h1>', unsafe_allow_html=True)
st.markdown('<p class="center space-below">Oleh:</p>', unsafe_allow_html=True)
st.markdown('<p class="center space-below"><b>Chandra Putra Ciptaningtyas</b></p>', unsafe_allow_html=True)
st.markdown('<p class="center space-below">NIM. 24050121140106</p>', unsafe_allow_html=True)

# Additional space between NIM and dosen pembimbing
st.markdown('<div class="space-between"></div>', unsafe_allow_html=True)

# Display dosen pembimbing information
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('''
    <p class="adjusted-left">
        Dosen Pembimbing 1<br><br>
        <span><b>Dr. Triastuti Wuryandari, S.Si., M.Si.</b></span><br>
        <span>NIP. 197109061998032001</span>
    </p>
    ''', unsafe_allow_html=True)

with col_right:
    st.markdown('''
    <p class="adjusted-right">
        Dosen Pembimbing 2<br><br>
        <span><b>Miftahul Jannah, S.Si., M.Si.</b></span><br>
        <span>NIP. 199403092024062005</span>
    </p>
    ''', unsafe_allow_html=True)
