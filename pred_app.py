{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPTJeq6gLGkNxNQayoDl49f",
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
        "<a href=\"https://colab.research.google.com/github/maulanaridhwan/BRIN_Project/blob/main/pred_app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "J1GsyOLOvr-I",
        "outputId": "6caed976-cb3c-4990-c019-cea1f74cd431"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.36.0-py2.py3-none-any.whl (8.6 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.6/8.6 MB\u001b[0m \u001b[31m28.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.3.3)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.25.2)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.1)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.0.3)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.1)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
            "  Downloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m22.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m75.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Collecting watchdog<5,>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-4.0.1-py3-none-manylinux2014_x86_64.whl (83 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m83.0/83.0 kB\u001b[0m \u001b[31m9.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m7.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2023.4)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.19.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
            "Installing collected packages: watchdog, smmap, pydeck, gitdb, gitpython, streamlit\n",
            "Successfully installed gitdb-4.0.11 gitpython-3.1.43 pydeck-0.9.1 smmap-5.0.1 streamlit-1.36.0 watchdog-4.0.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Qxf3GV4NuocA",
        "outputId": "6a9883cb-8835-4474-aa88-ed7cbd9e60b2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-07-18 02:50:19.184 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2024-07-18 02:50:19.187 Session state does not function when running a script without `streamlit run`\n"
          ]
        }
      ],
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