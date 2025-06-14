import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.title("📊 Prediksi Obesitas Berdasarkan Data Gaya Hidup")

uploaded_file = st.file_uploader("Unggah file CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.dataframe(df.head())

    st.write("Jumlah duplikat:", df.duplicated().sum())
    df = df.drop_duplicates()

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

    st.subheader("Distribusi Kelas Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='NObeyesdad', order=df['NObeyesdad'].value_counts().index, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("Boxplot untuk Fitur Numerik")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig2, axes = plt.subplots(nrows=1, ncols=len(numerical_cols), figsize=(15,5))
    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig2)

    st.subheader("Pelatihan dan Evaluasi Model")

    # Encode kategori
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop('NObeyesdad', axis=1)
    y = df_encoded['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.markdown(f"### {name}")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

        st.write("Confusion Matrix:")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        st.pyplot(fig_cm)
