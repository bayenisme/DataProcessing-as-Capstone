#!/usr/bin/env python
# coding: utf-8

# # **Exploratory Data Analysis**

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[52]:

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


# In[53]:

subprocess.check_call(['gdown', '--id', '1D1dZRAQFmt4eiyRTxGPgkTZ0LlJWQOVg'])


# In[54]:


df = pd.read_csv('ObesityDataSet.csv')


# In[55]:


print("5 Data Pertama")
print(df.head())


# In[56]:


print("\nInformasi Dataset")
print(df.info())


# In[57]:


print("\nStatistik Deskriptif")
print(df.describe(include='all'))


# In[58]:


print("\nJumlah Missing Value")
print(df.isnull().sum())


# In[59]:


print("\nJumlah Data Duplikat")
print(df.duplicated().sum())


# # **Visualisasi**

# In[60]:


plt.figure(figsize=(10,5))
sns.countplot(data=df, x='NObeyesdad', order=df['NObeyesdad'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Distribusi Kelas Target (NObeyesdad)')
plt.show()


# In[61]:


numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


# # **Preprocessing Data**

# In[62]:


print("Missing values:")
print(df.isnull().sum())

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

print("\nMissing values after handling:")
print(df.isnull().sum())


# # **Encoding Kategorikal**

# In[63]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_encoded = df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

X = df_encoded.drop('NObeyesdad', axis=1)
y = df_encoded['NObeyesdad']

print("\nDistribusi kelas setelah encoding:")
print(y.value_counts())


# # **Handling Imbalance dan Normalisasi**

# In[64]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    print(f"\nModel: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# In[66]:


def plot_confusion_matrix(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(df['NObeyesdad'].unique())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(models['Logistic Regression'], X_test, y_test, 'Logistic Regression (Before Tuning)')
plot_confusion_matrix(models['Random Forest'], X_test, y_test, 'Random Forest (Before Tuning)')
plot_confusion_matrix(models['KNN'], X_test, y_test, 'KNN (Before Tuning)')


# ## **Visualisasi Model**

# In[67]:


model_scores = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    model_scores['Model'].append(name)
    model_scores['Accuracy'].append(accuracy_score(y_test, y_pred))
    model_scores['Precision'].append(precision_score(y_test, y_pred, average='weighted'))
    model_scores['Recall'].append(recall_score(y_test, y_pred, average='weighted'))
    model_scores['F1 Score'].append(f1_score(y_test, y_pred, average='weighted'))

score_df = pd.DataFrame(model_scores)


plt.figure(figsize=(10, 6))
score_df_melted = score_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(data=score_df_melted, x='Model', y='Score', hue='Metric')
plt.title('Perbandingan Performa Model')
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# ## **Hyperparameter Tuning**

# In[68]:


param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("Best Parameters - Random Forest:")
print(grid_rf.best_params_)
print("\nClassification Report (Tuned Random Forest):")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))


# In[69]:


param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1: Manhattan, 2: Euclidean
}

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_knn.fit(X_train, y_train)

best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

print("Best Parameters - KNN:")
print(grid_knn.best_params_)
print("\nClassification Report (Tuned KNN):")
print(classification_report(y_test, y_pred_knn))


# In[70]:


param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2']
}

grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr,
                       cv=3, scoring='f1_weighted', n_jobs=-1)
grid_lr.fit(X_train, y_train)

best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)

from sklearn.metrics import classification_report
print("Best Parameters - Logistic Regression:")
print(grid_lr.best_params_)
print("\nClassification Report (Tuned Logistic Regression):")
print(classification_report(y_test, y_pred_lr))


# ## **Visualisasi Hasil Tuning**

# In[71]:


def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

comparison_scores = {
    'Model': [],
    'Stage': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for name, model in models.items():
    metrics = get_metrics(model, X_test, y_test)
    comparison_scores['Model'].append(name)
    comparison_scores['Stage'].append('Before Tuning')
    comparison_scores['Accuracy'].append(metrics['Accuracy'])
    comparison_scores['Precision'].append(metrics['Precision'])
    comparison_scores['Recall'].append(metrics['Recall'])
    comparison_scores['F1 Score'].append(metrics['F1 Score'])

tuned_models = {
    'Random Forest': best_rf,
    'KNN': best_knn,
    'Logistic Regression': best_lr
}

for name, model in tuned_models.items():
    metrics = get_metrics(model, X_test, y_test)
    comparison_scores['Model'].append(name)
    comparison_scores['Stage'].append('After Tuning')
    comparison_scores['Accuracy'].append(metrics['Accuracy'])
    comparison_scores['Precision'].append(metrics['Precision'])
    comparison_scores['Recall'].append(metrics['Recall'])
    comparison_scores['F1 Score'].append(metrics['F1 Score'])

score_compare_df = pd.DataFrame(comparison_scores)

plt.figure(figsize=(12, 6))
score_melted = score_compare_df.melt(id_vars=['Model', 'Stage'], var_name='Metric', value_name='Score')

sns.barplot(data=score_melted, x='Model', y='Score', hue='Stage', palette='Set2', ci=None)
plt.title('Perbandingan Performa Sebelum vs Sesudah Tuning')
plt.ylim(0, 1)
plt.xlabel('Model')
plt.ylabel('Skor (0-1)')
plt.legend(title='Stage')
plt.tight_layout()
plt.show()


# In[72]:


get_ipython().system('jupyter nbconvert --to script Capstone_Project_14855.ipynb')


# ## **Kesimpulan**

# 1. Dataset yang digunakan memiliki 2111 data dengan 17 fitur dan 1 target yang merepresentasikan tingkat obesitas pada individu dari Meksiko, Peru, dan Kolombia
# 2. Dataset memiliki distribusi kelas target yang tidak seimbang dan terdapat outlier pada kolom Weight, Height, dan Age
# 3. Algoritma Random Forest dan KNN mendapatkan performa yang lebih tinggi dibanding Logistic Regression
# 4. Hyperparameter Tuning dibutuhkan agar performa model dapat ditingkatkan, khususnya pada Random Forest dan KNN
