# ============================
# APP Streamlit - Análisis Diabetes (130 hospitales EE.UU)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import prince
from imblearn.over_sampling import SMOTE

# ============================
# Carga de datos
# ============================

st.title("Análisis de Datos de Diabetes - 130 Hospitales (EE.UU.)")

# Cargar dataset desde GitHub (debe estar en el mismo repo)
df = pd.read_csv("diabetic_data.csv")

st.subheader("Vista general de los datos")
st.write(df.head())

st.subheader("Información del DataFrame")
buffer = []
df.info(buf=buffer)
st.text("\n".join(buffer))

# ============================
# Limpieza de datos
# ============================

st.subheader("Limpieza de datos")

# Reemplazar '?' por NaN
df = df.replace('?', np.nan)

# Convertir numéricas a entero donde corresponde
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
    "change", "diabetesMed"
]

categoricas_codigos = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
]

cat_cols_mca.extend(categoricas_codigos)

st.write("Variables numéricas (para PCA):", num_cols_pca)
st.write("Variables categóricas (para MCA):", cat_cols_mca)

# ============================
# Distribución de la variable objetivo
# ============================

st.subheader("Distribución de readmisiones")

fig, ax = plt.subplots(figsize=(6,4))
df["readmitted"].value_counts().plot(kind="bar", ax=ax, color=["#66c2a5","#fc8d62","#8da0cb"])
ax.set_title("Distribución de readmisiones")
ax.set_ylabel("Número de pacientes")
ax.set_xlabel("Readmisión")
st.pyplot(fig)

# ============================
# Boxplot: tiempo en hospital según readmisión
# ============================

st.subheader("Tiempo en hospital según readmisión")

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x=df["readmitted"], y=df["time_in_hospital"], palette="Set3", ax=ax)
ax.set_title("Tiempo en hospital según readmisión")
st.pyplot(fig)

# ============================
# PCA - Variables numéricas
# ============================

st.subheader("PCA (Análisis de Componentes Principales) - Variables Numéricas")

X_num = df[num_cols_pca].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.3)
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.set_title("PCA - Variables numéricas")
st.pyplot(fig)

st.write("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
st.write("Varianza total explicada:", pca.explained_variance_ratio_.sum())

# ============================
# MCA - Variables categóricas
# ============================

st.subheader("MCA (Análisis de Correspondencias Múltiples) - Variables Categóricas")

X_cat = df[cat_cols_mca].dropna()

mca = prince.MCA(n_components=2, random_state=42)
mca = mca.fit(X_cat)

fig, ax = plt.subplots()
X_mca = mca.transform(X_cat)
ax.scatter(X_mca[0], X_mca[1], alpha=0.3)
ax.set_xlabel("Dimensión 1")
ax.set_ylabel("Dimensión 2")
ax.set_title("MCA - Variables categóricas")
st.pyplot(fig)

# ============================
# Explicaciones
# ============================

st.subheader("Interpretación de PCA y MCA")

st.markdown("""
- **PCA (numéricas):** nos permite reducir la dimensionalidad de las variables continuas como número de procedimientos, medicamentos, tiempo en hospital, etc.  
  - El **Componente 1** concentra la mayor variabilidad de los pacientes según uso de procedimientos y hospitalización.  
  - El **Componente 2** puede asociarse a la frecuencia de reingresos y número de diagnósticos.  
  - Así, PCA nos ayuda a identificar perfiles de pacientes con mayor consumo de recursos clínicos.

- **MCA (categóricas):** se aplica a variables como raza, género, edad, tipo de ingreso, diagnóstico principal/secundario, uso de medicamentos.  
  - La **Dimensión 1** suele separar pacientes por edad y tipo de tratamiento recibido.  
  - La **Dimensión 2** captura diferencias según diagnósticos y tipo de atención (urgencias, hospitalización, ambulatorio).  
  - Esto permite identificar perfiles de riesgo en función de características demográficas y clínicas categóricas.
""")

