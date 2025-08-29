# Proyecto Análisis Diabetes - Datos clínicos en 130 hospitales de EE.UU
# ---------------------------------------------------------------------------------
# El conjunto de datos representa diez años (1999-2008) de atención clínica
# en 130 hospitales y redes de prestación integradas en EE.UU.
#
# Variables disponibles incluyen: raza, género, edad, peso, códigos de diagnóstico,
# estancia hospitalaria, número de procedimientos, resultados de laboratorio,
# medicaciones, especialidad médica, y reingresos hospitalarios.
#
# El análisis se centra en:
#   - Limpieza de datos
#   - Análisis descriptivo
#   - PCA (variables numéricas)
#   - MCA (variables categóricas)
#   - Visualizaciones exploratorias
#
# Este script está adaptado para ejecutarse en Streamlit.

# =========================
# Librerías
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io
from sklearn.decomposition import PCA
from prince import MCA

# =========================
# Carga de datos
# =========================
st.title("Proyecto de Análisis de Diabetes")

df = pd.read_csv("diabetic_data.csv")

st.subheader("Vista previa de los datos")
st.write(df.head())

st.subheader("Información del DataFrame")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Descripción estadística")
st.write(df.describe(include="all"))

# =========================
# Variables para PCA y MCA
# =========================
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

st.write("Variables numéricas para PCA:", num_cols_pca)
st.write("Variables categóricas para MCA:", cat_cols_mca)

# =========================
# Distribución de readmisiones
# =========================
st.subheader("Distribución de readmisiones")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x=df["readmitted"], order=df["readmitted"].value_counts().index, palette="Set2", ax=ax)
ax.set_title("Distribución de readmisiones")
ax.set_ylabel("Número de pacientes")
ax.set_xlabel("Readmisión")
st.pyplot(fig)

# =========================
# Boxplot de tiempo en hospital según readmisión
# =========================
st.subheader("Tiempo en hospital según readmisión")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x=df["readmitted"], y=df["time_in_hospital"], palette="Set3", ax=ax)
ax.set_title("Tiempo en hospital según readmisión")
st.pyplot(fig)

# =========================
# PCA (Análisis de Componentes Principales)
# =========================
st.subheader("PCA - Variables numéricas")

X_num = df[num_cols_pca].fillna(0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_num)

st.write("Varianza explicada por cada componente:", pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.3)
ax.set_title("PCA - Proyección en 2D")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

# =========================
# MCA (Análisis de Correspondencias Múltiples)
# =========================
st.subheader("MCA - Variables categóricas")

X_cat = df[cat_cols_mca].astype(str).fillna("Missing")
mca = MCA(n_components=2, random_state=42)
X_mca = mca.fit(X_cat)

st.write("Inercia explicada:", mca.explained_inertia_)

fig, ax = plt.subplots(figsize=(6,4))
mca.plot_coordinates(X_cat, ax=ax, show_row_points=False, show_column_points=True)
ax.set_title("MCA - Proyección de categorías")
st.pyplot(fig)

# =========================
# Integración PCA + MCA
# =========================
st.subheader("Integración PCA + MCA")

X_reduced = np.hstack((X_pca, X_mca.transform(X_cat)))
st.write("Forma de la matriz reducida combinada:", X_reduced.shape)

# =========================
# Conclusiones
# =========================
st.subheader("Conclusiones")
st.write("""
- PCA permite resumir la variabilidad de las variables numéricas en pocos componentes principales.
- MCA ayuda a representar la estructura de las variables categóricas en un espacio reducido.
- La combinación PCA + MCA nos da una representación compacta y mixta de pacientes.
""")

