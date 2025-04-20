import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor

# Konstanta nilai minimum dan maksimum
MIN_VALUE = 784.91
MAX_VALUE = 2271.54
from datetime import datetime


# Path model dari CSV
MODELS_PATH = {
    "Model XGBoost Default": "models/xgboost_model_default_params.csv",
    "Model XGBoost GridSearchCV": "models/xgboost_model_gridsearchcv_params.csv", 
    "Model XGBoost PSO": "models/xgboost_pso_params.csv"
}

TRAIN_DATA_PATH = "models/train_data.csv"
TEST_DATA_PATH = "models/test_data.csv"


def custom_min_max_scaler(value, min_value=MIN_VALUE, max_value=MAX_VALUE):
    return 1 - ((value - min_value) / (max_value - min_value))

def load_model(model_name):
    model_path = MODELS_PATH.get(model_name)
    if not model_path:
        raise ValueError("Model tidak ditemukan dalam daftar MODELS_PATH.")
    
    try:
        # Muat parameter dari CSV
        model_params = pd.read_csv(model_path)
        params_dict = model_params.set_index("Parameter")["Value"].to_dict()
        
        # Konversi tipe data parameter
        params_dict["max_depth"] = int(float(params_dict.get("max_depth", 6)))
        params_dict["n_estimators"] = int(float(params_dict.get("n_estimators", 100)))
        params_dict["min_child_weight"] = float(params_dict.get("min_child_weight", 1))
        params_dict["gamma"] = float(params_dict.get("gamma", 0))
        params_dict["reg_lambda"] = float(params_dict.get("reg_lambda", 1))
        params_dict["learning_rate"] = float(params_dict.get("learning_rate", 0.3))
        params_dict["subsample"] = float(params_dict.get("subsample", 1))
        params_dict["colsample_bytree"] = float(params_dict.get("colsample_bytree", 1))
        
        # Buat model dengan parameter
        model = XGBRegressor(**params_dict)
        
        # Muat data training
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        
        # Pisahkan fitur dan target
        X_train = train_data[[
            "('Open', 'KLBF.JK')", 
            "('High', 'KLBF.JK')", 
            "('Low', 'KLBF.JK')", 
            "('Close', 'KLBF.JK')"
        ]]
        y_train = train_data['Next_Day_Close']
        
        # Normalisasi data training
        X_train_normalized = X_train.apply(lambda x: custom_min_max_scaler(x))
        
        # Fit model
        model.fit(X_train_normalized, y_train)
        
        return model
    except Exception as e:
        raise ValueError(f"❌ Terjadi kesalahan: {str(e)}")

def predict(model, open_price, high_price, low_price, close_price):
    input_data = np.array([
        [
            custom_min_max_scaler(float(open_price)),
            custom_min_max_scaler(float(high_price)),
            custom_min_max_scaler(float(low_price)),
            custom_min_max_scaler(float(close_price))
        ]
    ])
    
    prediction = model.predict(input_data)
    return prediction[0]

from sklearn.metrics import mean_squared_error

def predict_data_test(model):
    try:
        # Load data test
        test_data = pd.read_csv(TEST_DATA_PATH)  # ganti dengan path sebenarnya

        # Pilih fitur dengan benar
        X_test = test_data[[
            "('Open', 'KLBF.JK')", 
            "('High', 'KLBF.JK')", 
            "('Low', 'KLBF.JK')", 
            "('Close', 'KLBF.JK')"
        ]]
        y_test = test_data['Next_Day_Close']

        print(test_data.columns)


        # Periksa dimensi X_test
        print(X_test.shape)  # Pastikan ini menunjukkan (n_samples, 4)

        # Normalisasi fitur
        X_test_normalized = X_test.apply(lambda x: custom_min_max_scaler(x))

        # Periksa dimensi setelah normalisasi
        print(X_test_normalized.shape)  # Pastikan ini tetap (n_samples, 4)

        # Prediksi menggunakan model
        y_pred = model.predict(X_test_normalized)

        # Hitung RMSE secara manual
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Buat DataFrame untuk hasil prediksi dan aktual
        results_df = pd.DataFrame({
            "Tanggal": test_data["Date"] if "Date" in test_data.columns else pd.date_range(end=datetime.today(), periods=len(y_test)),
            "Harga Aktual": y_test,
            "Harga Prediksi": y_pred
        })

        return results_df, rmse

    except Exception as e:
        st.error(f"Gagal memproses data test: {e}")
        return None, None




# --- Streamlit UI ---
st.title("Prediksi Harga Saham KLBF dengan Model XGBoost")

model_option = st.selectbox("Pilih Model:", list(MODELS_PATH.keys()))

if st.button("Muat Model"):
    try:
        model = load_model(model_option)
        st.session_state['model'] = model  # Simpan model di session state
        st.success(f"Model {model_option} berhasil dimuat!")
    except Exception as e:
        st.error(f"Error: {str(e)}")

open_price = st.number_input("Harga Open:", min_value=0.0, format="%.2f")
high_price = st.number_input("Harga High:", min_value=0.0, format="%.2f")
low_price = st.number_input("Harga Low:", min_value=0.0, format="%.2f")
close_price = st.number_input("Harga Close:", min_value=0.0, format="%.2f")

if st.button("Prediksi Harga Close"):
    if 'model' in st.session_state:
        prediksi_harga = predict(st.session_state['model'], open_price, high_price, low_price, close_price)
        st.success(f"Prediksi Harga Close: {prediksi_harga:.2f}")
    else:
        st.error("Harap pilih dan muat model terlebih dahulu.")
