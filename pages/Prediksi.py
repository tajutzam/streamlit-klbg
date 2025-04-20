import streamlit as st
import pandas as pd
import numpy as np
from predict import load_model, predict

# Load daftar model
MODELS = ["Model XGBoost Default", "Model XGBoost GridSearchCV", "Model XGBoost PSO"]

# Halaman Prediksi
st.title("Prediksi Saham Kalbe Farma (KLBF)")
st.write("Memprediksi **harga penutupan** dari saham berdasarkan harga **open**, **high**, dan **low**.")

# Dropdown untuk memilih model
selected_model_name = st.selectbox("Pilih Model Prediksi", MODELS)

# Informasi performa model berdasarkan pilihan
if selected_model_name == "Model XGBoost GridSearchCV":
    st.text('''
    Mean Squared Error (MSE): 1082.06
    Root Mean Squared Error (RMSE): 32.89
    Mean Absolute Error (MAE): 24.96
    Mean Absolute Percentage Error (MAPE): 1.59
    R-squared: 0.91
''')
elif selected_model_name == "Model XGBoost PSO":
    st.text('''
    Mean Squared Error (MSE): 933.66
    Root Mean Squared Error (RMSE): 30.56
    Mean Absolute Error (MAE): 22.98
    Mean Absolute Percentage Error (MAPE): 1.47
    R-squared: 0.92
''')
elif selected_model_name == "Model XGBoost Default":
    st.text('''
    Mean Squared Error (MSE): 1216.80
    Root Mean Squared Error (RMSE): 34.88
    Mean Absolute Error (MAE): 27.48
    Mean Absolute Percentage Error (MAPE): 1.76
    R-squared: 0.90
''')

# Load model berdasarkan pilihan
model = load_model(selected_model_name)

# Menu input data
st.sidebar.header("Input Data Prediksi")
input_method = st.sidebar.radio("Pilih Metode Input", ["Manual", "Upload CSV"])

if input_method == "Manual":
    # Input manual
    open_price = st.number_input("Open Price", min_value=1530, step=50)
    high_price = st.number_input("High Price", min_value=1550, step=50)
    low_price = st.number_input("Low Price", min_value=1500, step=50)
    close_price = st.number_input("Close Price", min_value=1500, step=50)

    # Tombol prediksi
    if st.button("Predict"):
        if high_price >= low_price:
            # Prediksi
            predicted_close = predict(
                model, open_price, high_price, low_price, close_price)
            st.success(f"Prediksi Harga Penutupan: ${predicted_close:.2f}")
        else:
            st.error("High price harus lebih besar atau sama dengan low price.")

elif input_method == "Upload CSV":
    # Input melalui file CSV
    uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # Membaca file CSV
            df = pd.read_csv(uploaded_file)
            st.write("Data dari File CSV:")
            st.write(df.head())

            # Mengambil data terakhir untuk prediksi
            last_row = df.tail(1)
            open_price = last_row["Open"].values[0]
            high_price = last_row["High"].values[0]
            low_price = last_row["Low"].values[0]
            close_price = last_row["Close"].values[0]

            # Tombol prediksi
            if st.button("Predict"):
                if high_price >= low_price:
                    # Prediksi
                    predicted_close = predict(
                        model, open_price, high_price, low_price, close_price)
                    st.success(f"Prediksi Harga Penutupan: ${predicted_close:.2f}")
                else:
                    st.error("High price harus lebih besar atau sama dengan low price.")
        except KeyError:
            st.error("Pastikan file CSV memiliki kolom: Open, High, Low, dan Close.")