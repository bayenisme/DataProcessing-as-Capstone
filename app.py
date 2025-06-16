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

    label_encoder = LabelEncoder()
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    X = df_encoded.drop('NObeyesdad', axis=1)
    y = df_encoded['NObeyesdad']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier()
    }

    st.subheader("Evaluasi Model Sebelum Tuning")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"Model: {name}")
        st.write(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title(f'Confusion Matrix - {name}')
        st.pyplot(fig)

    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
    }

    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    st.subheader("Best Parameters - Random Forest")
    st.write(grid_rf.best_params_)
    st.write("Classification Report (Tuned Random Forest):")
    st.write(classification_report(y_test, y_pred_rf))

    comparison_scores = {
        'Model': ['Random Forest (Tuned)', 'Logistic Regression', 'KNN'],
        'Accuracy': [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, models['Logistic Regression'].predict(X_test)), accuracy_score(y_test, models['KNN'].predict(X_test))],
    }

    comparison_df = pd.DataFrame(comparison_scores)

    st.subheader("Perbandingan Akurasi Model")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=comparison_df, x='Model', y='Accuracy')
    plt.title('Perbandingan Akurasi Model')
    st.pyplot(fig)
