import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV

st.title("Aplikasi Prediksi Obesitas")

uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Informasi Dataset")
    st.text(df.info())

    st.subheader("Statistik Deskriptif")
    st.write(df.describe(include='all'))

    st.subheader("Distribusi Kelas Target")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x='NObeyesdad', order=df['NObeyesdad'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Distribusi Kelas Target (NObeyesdad)')
    st.pyplot(fig)

    st.subheader("Preprocessing Data")
    for col in ['Age', 'Height', 'Weight']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Age', 'Height', 'Weight'], inplace=True)

    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[column] >= lower) & (data[column] <= upper)]

    for col in ['Age', 'Height', 'Weight']:
        df = remove_outliers_iqr(df, col)

    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled
