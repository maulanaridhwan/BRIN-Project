{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOyKSqrLwycmJZfMRWa+C58",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/maulanaridhwan/BRIN_Project/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "Qxf3GV4NuocA"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.impute import SimpleImputer\n",
        "from imblearn.over_sampling import SMOTE\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# Fungsi untuk memuat data dan memprosesnya\n",
        "def load_data():\n",
        "    df = pd.read_excel('/content/DATA SISTEM CAPASITOR BANK (1).xlsx')\n",
        "    df_cleaned = df.dropna()\n",
        "\n",
        "    # Mengonversi semua nilai dalam kolom TEMPERATURE menjadi string terlebih dahulu\n",
        "    df_cleaned.loc[:, 'TEMPERATURE'] = df_cleaned['TEMPERATURE'].astype(str)\n",
        "\n",
        "    # Mengekstraksi angka dari nilai string dan mengonversinya menjadi numerik\n",
        "    df_cleaned.loc[:, 'TEMPERATURE'] = pd.to_numeric(df_cleaned['TEMPERATURE'].str.extract('(\\d+)')[0], errors='coerce')\n",
        "\n",
        "    # Memisahkan fitur dan label\n",
        "    columns_to_exclude = [f'MODUL_{i}' for i in range(1, 13)] + ['TANGGAL_PELAKSANAAN']\n",
        "    X = df_cleaned.drop(columns=columns_to_exclude)\n",
        "    labels = [f'MODUL_{i}' for i in range(1, 13)]\n",
        "    y = df_cleaned[labels]\n",
        "\n",
        "    # Menangani nilai NaN pada fitur dengan menggunakan SimpleImputer\n",
        "    imputer = SimpleImputer(strategy='mean')\n",
        "    X_imputed = imputer.fit_transform(X)\n",
        "    X = pd.DataFrame(X_imputed, columns=X.columns)\n",
        "\n",
        "    return X, y, imputer\n",
        "\n",
        "# Memuat data\n",
        "X, y, imputer = load_data()\n",
        "\n",
        "# Menggunakan SMOTE dan melatih model Random Forest untuk setiap modul\n",
        "models = {}\n",
        "smote = SMOTE(random_state=42)\n",
        "\n",
        "for label in y.columns:\n",
        "    X_resampled, y_resampled = smote.fit_resample(X, y[label])\n",
        "    rf = RandomForestClassifier(random_state=42)\n",
        "    rf.fit(X_resampled, y_resampled)\n",
        "    models[label] = rf\n",
        "\n",
        "# Judul aplikasi\n",
        "st.title(\"Prediksi Status Modul Kapasitor\")\n",
        "\n",
        "# Membagi inputan menjadi dua kolom\n",
        "col1, col2 = st.columns(2)\n",
        "\n",
        "with col1:\n",
        "    faktor_daya = st.number_input('FAKTOR DAYA', step=0.01, format=\"%.2f\")\n",
        "    line_voltage_ln = st.number_input('LINE VOLTAGE LN', step=1)\n",
        "    apparent_current = st.number_input('APPARENT CURRENT', step=1)\n",
        "    reactive_power = st.number_input('REACTIVE POWER', step=1)\n",
        "    active_power = st.number_input('ACTIVE POWER', step=1)\n",
        "    apparent_power = st.number_input('APPARENT POWER', step=1)\n",
        "    diff_to_pf = st.number_input('DIFF TO PF', step=0.01, format=\"%.2f\")\n",
        "\n",
        "with col2:\n",
        "    frequency = st.number_input('FREQUENCY', step=1)\n",
        "    temperature = st.number_input('TEMPERATURE', step=1)\n",
        "    harmonics_v = st.number_input('HARMONICS V', step=0.1, format=\"%.1f\")\n",
        "    harmonics_i = st.number_input('HARMONICS I', step=0.1, format=\"%.1f\")\n",
        "    harmonics_thd_v = st.number_input('HARMONICS THD V', step=0.1, format=\"%.1f\")\n",
        "    harmonics_thd_i = st.number_input('HARMONICS THD I', step=0.1, format=\"%.1f\")\n",
        "\n",
        "# Tombol prediksi\n",
        "if st.button('Prediksi'):\n",
        "    # Menyusun data baru untuk prediksi\n",
        "    new_data = np.array([[faktor_daya, line_voltage_ln, apparent_current, reactive_power, active_power,\n",
        "                        apparent_power, diff_to_pf, frequency, temperature, harmonics_v, harmonics_i,\n",
        "                        harmonics_thd_v, harmonics_thd_i]])\n",
        "\n",
        "    # Menangani nilai NaN pada data baru (menggunakan mean dari data yang dilatih jika ada NaN)\n",
        "    new_data_imputed = imputer.transform(new_data)\n",
        "\n",
        "    # Mengonversi kembali ke DataFrame dengan nama kolom yang sama\n",
        "    new_data_df = pd.DataFrame(new_data_imputed, columns=X.columns)\n",
        "\n",
        "    # Melakukan prediksi untuk setiap modul\n",
        "    predictions = {}\n",
        "    for label, model in models.items():\n",
        "        prediction = model.predict(new_data_df)\n",
        "        predictions[label] = prediction[0]\n",
        "\n",
        "\n",
        "    # Menampilkan hasil prediksi\n",
        "    st.subheader(\"Hasil Prediksi:\")\n",
        "    for label, prediction in predictions.items():\n",
        "        status = \"Active\" if prediction == 1 else \"Inactive\"\n",
        "        st.write(f\"{label}: {status}\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "rWLTtp3MvRXQ"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}