# -*- coding: utf-8 -*-
"""
Proyecto Diabetes - PCA + MCA + KNN + Árbol de Decisión
Adaptado para Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE
import prince

# -------------------------------
# 1. Título y descripción
# -------------------------------
st.markdown("""
# Proyecto Diabetes - Datos clínicos de 130 hospitales y centros de atención integrados en EE.UU

El conjunto de datos contiene información de readmisiones hospitalarias por diabetes de atención clínica en 130 hospitales de EE. UU. y redes de prestación integradas.
Periodo: 10 años (1999–2008)

Objetivo: Analizar la readmisión de pacientes diabéticos, es decir, si después del alta regresan al hospital dentro de 30 días, después de 30 días o nunca.
""")

# -------------------------------
# 2. Cargar datos
# -------------------------------
df = pd.read_csv("diabetic_data.csv")
st.markdown("### Vista rápida del dataset")
st.dataframe(df.head())

# -------------------------------
# 3. Limpieza y mapeo de variables
# -------------------------------
df["admission_type_id"] = df["admission_type_id"].astype(str)
df["discharge_disposition_id"] = df["discharge_disposition_id"].astype(str)
df["admission_source_id"] = df["admission_source_id"].astype(str)

map_admission_type = {
    "1": "Emergencia", "2": "Urgente", "3": "Electivo", "4": "Recién nacido",
    "5": "Sin_info", "6": "Sin_info", "7": "Centro de trauma", "8": "Sin_info"
}

map_discharge_disposition = {
    "1": "Alta a casa", "2": "Alta a otro hospital a corto plazo", 
    "3": "Alta a centro de enfermería especializada", "4": "Alta a centro de cuidados intermedios",
    "5": "Alta a otro tipo de atención hospitalaria", "6": "Cuidados de salud en casa",
    "7": "Salida contra recomendación médica", "8": "Alta a casa con cuidados",
    "9": "Admitido como paciente hospitalizado", "10": "Cuidados paliativos en casa",
    "11": "Cuidados paliativos en centro médico", "12": "Alta a hospital psiquiátrico",
    "13": "Alta a otra instalación de rehabilitación", "14": "Sin_info", "15": "Sin_info",
    "16": "Alta a hospital federal", "17": "Alta a otra institución", "18": "Alta a custodia policial",
    "19": "Sin_info", "20": "Alta por orden judicial", "21": "Sin_info", "22": "Falleció en casa",
    "23": "Falleció en instalación médica", "24": "Falleció en lugar desconocido",
    "25": "Falleció en cuidados paliativos", "28": "Falleció en centro de enfermería especializada"
}

map_admission_source = {
    "1": "Referencia médica", "2": "Referencia desde clínica", "3": "Referencia desde aseguradora HMO",
    "4": "Transferencia desde hospital", "5": "Transferencia desde centro de enfermería especializada",
    "6": "Transferencia desde otro centro de salud", "7": "Sala de emergencias", "8": "Corte o custodia policial",
    "9": "Sin_info", "10": "Transferencia desde hospital de acceso crítico", "11": "Parto normal",
    "12": "Sin_info", "13": "Nacido en hospital", "14": "Nacido fuera de hospital",
    "15": "Sin_info", "17": "Sin_info", "20": "Sin_info", "22": "Sin_info", "25": "Sin_info"
}

df["admission_type_id"] = df["admission_type_id"].map(lambda x: map_admission_type.get(x, "Desconocido"))
df["discharge_disposition_id"] = df["discharge_disposition_id"].map(lambda x: map_discharge_disposition.get(x, "Desconocido"))
df["admission_source_id"] = df["admission_source_id"].map(lambda x: map_admission_source.get(x, "Desconocido"))

# -------------------------------
# 4. Manejo de faltantes y duplicados
# -------------------------------
missing_vals = ["None", "?"]
df = df.replace(missing_vals, pd.NA)

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna("Sin_info")

df = df.drop(columns=["encounter_id", "patient_nbr"])

# -------------------------------
# 5. Variables PCA y MCA
# -------------------------------
num_cols_pca = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

cat_cols_mca = [
    "race", "gender", "age", "weight", "payer_code",
    "medical_specialty", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed",
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
]

# -------------------------------
# 6. Visualizaciones iniciales
# -------------------------------
st.markdown("### Distribución de la variable objetivo (readmitted)")
y = df["readmitted"]
counts = y.value_counts()
labels = ['NO', '<30', '>30']
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(labels, counts)
ax.set_xlabel("Clase")
ax.set_ylabel("Cantidad de observaciones")
st.pyplot(fig)
plt.clf()

st.markdown("### Histograma de variables numéricas")
fig, ax = plt.subplots(figsize=(12,6))
df[num_cols_pca].hist(bins=30, edgecolor="black", ax=ax)
st.pyplot(fig)
plt.clf()

st.markdown("### Boxplot tiempo en hospital según readmisión")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x=y, y=df["time_in_hospital"], palette="Set3", ax=ax)
st.pyplot(fig)
plt.clf()

# -------------------------------
# 7. Split de datos
# -------------------------------
x = df.drop("readmitted", axis=1)
y = df["readmitted"]

x_train, x_test, y_train, y_test  = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

# Escalado de variables numéricas para PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train[num_cols_pca])
X_test_scaled  = scaler.transform(x_test[num_cols_pca])

# -------------------------------
# 8. PCA
# -------------------------------
pca = PCA()
X_pca = pca.fit_transform(X_train_scaled)
explained_var = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(explained_var)+1), explained_var, marker='o', linestyle='--')
ax.axhline(y=0.85, color='r', linestyle='-')
ax.set_xlabel("Número de componentes principales")
ax.set_ylabel("Varianza acumulada explicada")
ax.set_title("Varianza acumulada PCA")
st.pyplot(fig)
plt.clf()

# -------------------------------
# 9. MCA
# -------------------------------
X_train_cat = x_train[cat_cols_mca].astype(str)
mca_model = prince.MCA(n_components=15, random_state=42)
mca_model = mca_model.fit(X_train_cat)
X_mca = mca_model.transform(X_train_cat)
eigvals = mca_model.eigenvalues_
var_exp = eigvals / eigvals.sum()
cum_var_exp = np.cumsum(var_exp)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(cum_var_exp)+1), cum_var_exp, marker='o', linestyle='--')
ax.axhline(y=0.85, color='r', linestyle='-')
ax.set_xlabel("Dimensiones MCA")
ax.set_ylabel("Varianza acumulada explicada")
ax.set_title("Varianza acumulada MCA")
st.pyplot(fig)
plt.clf()

# -------------------------------
# 10. Reducir dimensiones y combinar
# -------------------------------
n_pca = np.argmax(explained_var >= 0.85) + 1
X_pca_reduced = X_pca[:, :n_pca]

n_mca = np.argmax(cum_var_exp >= 0.85) + 1
X_mca_reduced = X_mca.iloc[:, :n_mca].values

X_reduced = np.hstack((X_pca_reduced, X_mca_reduced))
pca_cols = [f"PCA_{i+1}" for i in range(n_pca)]
mca_cols = [f"MCA_{i+1}" for i in range(n_mca)]
X_reduced_df = pd.DataFrame(X_reduced, columns=pca_cols + mca_cols, index=x_train.index)

# -------------------------------
# 11. Preparar datos para KNN y Árbol
# -------------------------------
X_reduced_df["target"] = y_train.values
X_final = X_reduced_df.drop("target", axis=1)
y_final = X_reduced_df["target"]

le = LabelEncoder()
y_final_encoded = le.fit_transform(y_final)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final_encoded, test_size=0.2, stratify=y_final_encoded, random_state=42
)

# -------------------------------
# 12. Balanceo con SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

st.markdown("### Datos listos para KNN y Árbol de decisión")
st.write("Filas de entrenamiento balanceadas:", X_train_res.shape[0])
st.write("Número de columnas:", X_train_res.shape[1])
