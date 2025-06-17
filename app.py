import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

st.title("📊 Prediksi Obesitas Berdasarkan Data Gaya Hidup")

# Upload CSV file
uploaded_file = st.file_uploader("Unggah file CSV dataset", type=["csv"])

if uploaded_file:
    # Membaca file CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Awal")
    st.dataframe(df.head())

    # Menampilkan informasi dataset
    st.write("Jumlah duplikat:", df.duplicated().sum())
    df = df.drop_duplicates()

    # Convert kolom 'Age', 'Height', 'Weight' menjadi numerik
    for col in ['Age', 'Height', 'Weight']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Menghapus missing values setelah konversi
    df.dropna(subset=['Age', 'Height', 'Weight'], inplace=True)

    # Menangani outlier menggunakan metode IQR
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[(data[column] >= lower) & (data[column] <= upper)]

    for col in ['Age', 'Height', 'Weight']:
        df = remove_outliers_iqr(df, col)

    # Visualisasi Distribusi Kelas Target
    st.subheader("Distribusi Kelas Target")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='NObeyesdad', order=df['NObeyesdad'].value_counts().index, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Visualisasi Boxplot untuk Fitur Numerik
    st.subheader("Boxplot untuk Fitur Numerik")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig2, axes = plt.subplots(nrows=1, ncols=len(numerical_cols), figsize=(15, 5))
    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
    st.pyplot(fig2)

    # Label Encoding untuk kolom kategorikal
    label_encoder = LabelEncoder()
    df_encoded = df.copy()

    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    X = df_encoded.drop('NObeyesdad', axis=1)
    y = df_encoded['NObeyesdad']

    st.write("\nDistribusi kelas setelah encoding:")
    st.write(y.value_counts())

    # Handling imbalance and normalization
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Membagi dataset menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

    # Pelatihan dan evaluasi model
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, penalty='l2'),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier()
    }

    for name, model in models.items():
        st.write(f"\nModel: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("\nClassification Report:")
        st.text(classification_report(y_test, y_pred))

    # Cek train score dan test score untuk mendeteksi overfitting
    st.write("\nTrain vs Test Accuracy (Untuk Deteksi Overfitting):")
    for name, model in models.items():
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        st.write(f"{name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Visualisasi Confusion Matrix
    def plot_confusion_matrix(model, X_test, y_test, title):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(df['NObeyesdad'].unique())

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f'Confusion Matrix - {title}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    for name, model in models.items():
        plot_confusion_matrix(model, X_test, y_test, name)

    # Hyperparameter tuning using GridSearchCV
    param_grid_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],  # Tambahkan pembatasan kedalaman pohon
        'min_samples_split': [2, 5, 10],  # Batasan minimum sampel untuk membagi
        'min_samples_leaf': [1, 2, 4]  # Batasan minimum sampel di setiap daun pohon
    }

    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    st.write("Best Parameters - Random Forest:")
    st.text(grid_rf.best_params_)
    st.write("\nClassification Report (Tuned Random Forest):")
    st.text(classification_report(y_test, y_pred_rf))

    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1: Manhattan, 2: Euclidean
    }

    grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_knn.fit(X_train, y_train)

    best_knn = grid_knn.best_estimator_
    y_pred_knn = best_knn.predict(X_test)

    st.write("Best Parameters - KNN:")
    st.text(grid_knn.best_params_)
    st.write("\nClassification Report (Tuned KNN):")
    st.text(classification_report(y_test, y_pred_knn))

    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    }

    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_lr.fit(X_train, y_train)

    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)

    st.write("Best Parameters - Logistic Regression:")
    st.text(grid_lr.best_params_)
    st.write("\nClassification Report (Tuned Logistic Regression):")
    st.text(classification_report(y_test, y_pred_lr))
