import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Fungsi untuk memuat data dan memprosesnya
def load_data():
    df = pd.read_excel('DATA SISTEM CAPASITOR BANK (1).xlsx')
    df_cleaned = df.dropna()

    # Mengonversi semua nilai dalam kolom TEMPERATURE menjadi string terlebih dahulu
    df_cleaned.loc[:, 'TEMPERATURE'] = df_cleaned['TEMPERATURE'].astype(str)

    # Mengekstraksi angka dari nilai string dan mengonversinya menjadi numerik
    df_cleaned.loc[:, 'TEMPERATURE'] = pd.to_numeric(df_cleaned['TEMPERATURE'].str.extract('(\d+)')[0], errors='coerce')

    # Memisahkan fitur dan label
    columns_to_exclude = [f'MODUL_{i}' for i in range(1, 13)] + ['TANGGAL_PELAKSANAAN']
    X = df_cleaned.drop(columns=columns_to_exclude)
    labels = [f'MODUL_{i}' for i in range(1, 13)]
    y = df_cleaned[labels]

    # Menangani nilai NaN pada fitur dengan menggunakan SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)

    return X, y, imputer

# Memuat data
X, y, imputer = load_data()

# Menggunakan SMOTE dan melatih model Random Forest untuk setiap modul
models = {}
smote = SMOTE(random_state=42)

for label in y.columns:
    X_resampled, y_resampled = smote.fit_resample(X, y[label])
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_resampled, y_resampled)
    models[label] = rf

# Judul aplikasi
st.title("Prediksi Status Modul Kapasitor")

# Membagi inputan menjadi dua kolom
col1, col2 = st.columns(2)

with col1:
    faktor_daya = st.number_input('FAKTOR DAYA', step=0.01, format="%.2f")
    line_voltage_ln = st.number_input('LINE VOLTAGE LN', step=1)
    apparent_current = st.number_input('APPARENT CURRENT', step=1)
    reactive_power = st.number_input('REACTIVE POWER', step=1)
    active_power = st.number_input('ACTIVE POWER', step=1)
    apparent_power = st.number_input('APPARENT POWER', step=1)
    diff_to_pf = st.number_input('DIFF TO PF', step=0.01, format="%.2f")

with col2:
    frequency = st.number_input('FREQUENCY', step=1)
    temperature = st.number_input('TEMPERATURE', step=1)
    harmonics_v = st.number_input('HARMONICS V', step=0.1, format="%.1f")
    harmonics_i = st.number_input('HARMONICS I', step=0.1, format="%.1f")
    harmonics_thd_v = st.number_input('HARMONICS THD V', step=0.1, format="%.1f")
    harmonics_thd_i = st.number_input('HARMONICS THD I', step=0.1, format="%.1f")

# Tombol prediksi
if st.button('Prediksi'):
    # Menyusun data baru untuk prediksi
    new_data = np.array([[faktor_daya, line_voltage_ln, apparent_current, reactive_power, active_power, 
                        apparent_power, diff_to_pf, frequency, temperature, harmonics_v, harmonics_i, 
                        harmonics_thd_v, harmonics_thd_i]])

    # Menangani nilai NaN pada data baru (menggunakan mean dari data yang dilatih jika ada NaN)
    new_data_imputed = imputer.transform(new_data)

    # Mengonversi kembali ke DataFrame dengan nama kolom yang sama
    new_data_df = pd.DataFrame(new_data_imputed, columns=X.columns)

    # Melakukan prediksi untuk setiap modul
    predictions = {}
    for label, model in models.items():
        prediction = model.predict(new_data_df)
        predictions[label] = prediction[0]


    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    for label, prediction in predictions.items():
        status = "Active" if prediction == 1 else "Inactive"
        st.write(f"{label}: {status}")
