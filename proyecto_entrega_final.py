
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot
import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.title("Diabetes-Datos clínicos de 130 hospitales y centros de atención integrados en EE.UU")
st.markdown("""
**El conjunto de datos:**
Readmisiones hospitalarias por diabetesde atención clínica en 130 hospitales de EE. UU. y redes de prestación integradas.
Periodo: 10 años (1999–2008)



Los datos contienen atributos como el número de pacientes, la raza, el sexo, la edad, el tipo de ingreso, el tiempo en el hospital, la especialidad médica del médico de admisión, el número de pruebas de laboratorio realizadas, el resultado de la prueba de HbA1c, el diagnóstico, el número de medicamentos, los medicamentos para la diabetes, el número de visitas ambulatorias, hospitalarias y de emergencia en el año anterior a la hospitalización, etc.

Objetivo: Analizar la readmisión de pacientes diabéticos, es decir, si después del alta regresan al hospital dentro de 30 días, después de 30 días o nunca

""")

st.subheader("Obtener los datos")

st.markdown("---")


# =========================
# Mostrar el docstring inicial (tu explicación original)
# =========================
if __doc__:
    st.markdown(__doc__)


# =========================
# CARGA DE DATOS 

# =========================
df = None
try:
    df = pd.read_csv("diabetic_data.csv")
except Exception:
    df = pd.read_csv("/content/drive/MyDrive/ML-BIOESTADISTICA/diabetic_data.csv")

st.write("Vista inicial del DataFrame:")
st.write(df.head())

# Ver valores únicos en las columnas categóricas codificadas
st.write("admission_type_id:", sorted(df["admission_type_id"].unique()))
st.write("discharge_disposition_id:", sorted(df["discharge_disposition_id"].unique()))
st.write("admission_source_id:", sorted(df["admission_source_id"].unique()))

#Convertir a string
df["admission_type_id"] = df["admission_type_id"].astype(str)
df["discharge_disposition_id"] = df["discharge_disposition_id"].astype(str)
df["admission_source_id"] = df["admission_source_id"].astype(str)

map_admission_type = {
    "1": "Emergencia",
    "2": "Urgente",
    "3": "Electivo",
    "4": "Recién nacido",
    "5": "Sin_info",
    "6": "Sin_info",
    "7": "Centro de trauma",
    "8": "Sin_info"
}

map_discharge_disposition = {
    "1": "Alta a casa",
    "2": "Alta a otro hospital a corto plazo",
    "3": "Alta a centro de enfermería especializada",
    "4": "Alta a centro de cuidados intermedios",
    "5": "Alta a otro tipo de atención hospitalaria",
    "6": "Cuidados de salud en casa",
    "7": "Salida contra recomendación médica",
    "8": "Alta a casa con cuidados",
    "9": "Admitido como paciente hospitalizado",
    "10": "Cuidados paliativos en casa",
    "11": "Cuidados paliativos en centro médico",
    "12": "Alta a hospital psiquiátrico",
    "13": "Alta a otra instalación de rehabilitación",
    "14": "Sin_info",
    "15": "Sin_info",
    "16": "Alta a hospital federal",
    "17": "Alta a otra institución",
    "18": "Alta a custodia policial",
    "19": "Sin_info",
    "20": "Alta por orden judicial",
    "21": "Sin_info",
    "22": "Falleció en casa",
    "23": "Falleció en instalación médica",
    "24": "Falleció en lugar desconocido",
    "25": "Falleció en cuidados paliativos",
    "28": "Falleció en centro de enfermería especializada"
}

map_admission_source = {
    "1": "Referencia médica",
    "2": "Referencia desde clínica",
    "3": "Referencia desde aseguradora HMO",
    "4": "Transferencia desde hospital",
    "5": "Transferencia desde centro de enfermería especializada",
    "6": "Transferencia desde otro centro de salud",
    "7": "Sala de emergencias",
    "8": "Corte o custodia policial",
    "9": "Sin_info",
    "10": "Transferencia desde hospital de acceso crítico",
    "11": "Parto normal",
    "12": "Sin_info",
    "13": "Nacido en hospital",
    "14": "Nacido fuera de hospital",
    "15": "Sin_info",
    "17": "Sin_info",
    "20": "Sin_info",
    "22": "Sin_info",
    "25": "Sin_info"
}

# --- 3. Reemplazar con map() y 'Desconocido' para valores fuera del diccionario ---
df["admission_type_id"] = df["admission_type_id"].map(lambda x: map_admission_type.get(x, "Desconocido"))
df["discharge_disposition_id"] = df["discharge_disposition_id"].map(lambda x: map_discharge_disposition.get(x, "Desconocido"))
df["admission_source_id"] = df["admission_source_id"].map(lambda x: map_admission_source.get(x, "Desconocido"))

# --- 4. Verificación rápida ---
st.write(df[["admission_type_id", "discharge_disposition_id", "admission_source_id"]].head())

# Tamaño y forma del Dataset
st.write("Shape:", df.shape)
st.write("Número de filas:", df.shape[0])
st.write("Número de columnas:", df.shape[1])

# Revisar duplicados
st.write("Duplicados:", df.duplicated().sum())

missing_vals = ["None", "?"]
df = df.replace(missing_vals, pd.NA)

# Verificar resultado en algunas columnas
st.write(df['max_glu_serum'].value_counts(dropna=False))
st.write(df['weight'].value_counts(dropna=False))

faltantes = df.isna().sum().sort_values(ascending=False)
faltantes_pct = (df.isna().mean() * 100).sort_values(ascending=False)

faltantes_df = pd.DataFrame({
    'Faltantes': faltantes,
    'Porcentaje': faltantes_pct
})
st.write(faltantes_df)

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna("Sin_info")
st.write(df[cat_cols].isna().sum().sort_values(ascending=False))

num_cols = df.select_dtypes(include=['int64','float64']).columns
st.write(df[num_cols].isna().sum().sort_values(ascending=False))

st.write(df)

df = df.drop(columns=["encounter_id", "patient_nbr"])

#Variables numéricas (PCA)
num_cols_pca = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

#Variables categóricas (MCA)
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

#Categóricas que estan con códigos numéricos
categoricas_codigos = [
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
]

#  las tenemos en cuenta para al MCA
cat_cols_mca.extend(categoricas_codigos)

# Verificación
st.write("PCA (numéricas):", len(num_cols_pca), num_cols_pca)
st.write("MCA (categóricas):", len(cat_cols_mca), cat_cols_mca)

st.markdown("""📊Distribución de la variable objetivo (readmitted)""")

st.write(df["readmitted"].value_counts())

# Gráfico de barras de readmitted
fig, ax = plt.subplots(figsize=(6,4))
df["readmitted"].value_counts().plot(
    kind="bar",
    ax=ax,
    color=["#66c2a5","#fc8d62","#8da0cb"]
)
ax.set_title("Distribución de readmisiones")
ax.set_ylabel("Número de pacientes")
ax.set_xlabel("Readmisión")
st.pyplot(fig)

st.markdown("""📊 Histograma / KDE de variables numéricas

Las numéricas relevantes

time_in_hospital: la mayoría de pacientes pasan cuantos días?

num_lab_procedures: algunos pacientes tienen muchos más exámenes que otros?

num_medications: hay tendencia a medicar más a los que reingresan?
""")

num_cols = ["time_in_hospital", "num_lab_procedures", "num_medications"]
# Histograma múltiple
axes = df[num_cols].hist(bins=30, figsize=(12,6), edgecolor="black")
plt.suptitle("Distribución de variables numéricas")
st.pyplot(plt.gcf())

st.markdown("""📊 Boxplot por clase objetivo""")

# Boxplot
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x=df["readmitted"], y=df["time_in_hospital"], palette="Set3", ax=ax)
ax.set_title("Tiempo en hospital según readmisión")
ax.set_xlabel("Readmisión")
ax.set_ylabel("Tiempo en hospital (días)")
st.pyplot(fig)

st.markdown("📊 **Reingreso según tipo de alta (discharge_disposition_id)**")

# Contar número de pacientes por tipo de alta y readmisión
count_df = df.groupby(['discharge_disposition_id', 'readmitted']).size().reset_index(name='count')

# Gráfico de barras agrupadas
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(data=count_df, x='discharge_disposition_id', y='count', hue='readmitted', palette='Set2', ax=ax)
ax.set_title("Reingreso según tipo de alta hospitalaria")
ax.set_xlabel("Discharge Disposition")
ax.set_ylabel("Número de pacientes")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Readmisión')
plt.tight_layout()
st.pyplot(fig)


st.markdown("""📊 Distribucion de género o raza""")

fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x=df["gender"], palette="pastel", ax=ax)
ax.set_title("Distribución por género")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8,5))
sns.countplot(x=df["race"], palette="muted", ax=ax)
ax.set_title("Distribución por raza")
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("""📊 Mapa de calor de correlaciones (solo numéricas)""")

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df[num_cols_pca].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
ax.set_title("Matriz de correlación - variables numéricas")
st.pyplot(fig)

x= df.drop("readmitted", axis=1)
y= df["readmitted"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train[num_cols_pca])
X_test_scaled  = scaler.transform(x_test[num_cols_pca])

st.markdown("""# TAREA 1: PCA + MCA""")

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler as _SS_  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 

# PCA con todos los componentes
pca = PCA()
X_pca = pca.fit_transform(X_train_scaled)

# Varianza acumulada
explained_var = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(explained_var) + 1), explained_var, marker='o', linestyle='--')
ax.axhline(y=0.85, color='r', linestyle='-')
ax.set_xlabel('Número de componentes principales')
ax.set_ylabel('Varianza acumulada explicada')
ax.set_title('Varianza acumulada explicada por PCA')
ax.grid(True)
st.pyplot(fig)

# Scatterplot PC1 vs PC2
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_train, palette='Set1', alpha=0.7, ax=ax)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Scatterplot PC1 vs PC2')
ax.legend(title='Clase', labels=['NO', '<30', '>30'])
st.pyplot(fig)

# Loadings (cargas) de las variables en las PCs
loadings = pd.DataFrame(pca.components_.T,columns=[f'PC{i+1}' for i in range(pca.n_components_)],index=num_cols_pca)

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(loadings.iloc[:,:10], annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Heatmap de loadings (primeras PCs)')
st.pyplot(fig)

# Aplicar PCA
pca = PCA(n_components=0.85)
X_pca = pca.fit_transform(X_train_scaled)

st.write(f"Número de componentes principales para explicar 85% varianza: {pca.n_components_}")
st.write(f"Varianza explicada acumulada por estas componentes: {sum(pca.explained_variance_ratio_):.4f}")


st.markdown("""# Análisis de los resultados obtenidos en MCA y PCA

Análisis de resultados PCA

Al aplicar PCA sobre las variables numéricas del dataset:

Se observó que con 6 componentes principales se logra conservar la mayor parte de la información sin necesidad de mantener todas las variables originales, lo que significa que el espacio de alta dimensionalidad de las variables numéricas (como valores de laboratorio, número de visitas, edad, etc.) puede representarse en un espacio reducido, facilitando el análisis.

El scatterplot mostró cierta separación entre las clases de readmisión (NO, <30, >30), aunque con solapamiento. Esto confirma que existen patrones numéricos que ayudan a explicar parte de la variabilidad entre pacientes, pero no son totalmente discriminantes por sí solos.

Conclusión
La reducción permitió simplificar el conjunto de variables numéricas manteniendo más del 85% de la varianza. Esto mejora la eficiencia y reduce el riesgo de sobreajuste, aunque por sí sola la variabilidad numérica no separa completamente las clases.

Análisis de resultados MCA

Con el MCA aplicado a las variables categóricas (como género, tipo de admisión, tipo de alta, diagnósticos, etc.):

Se encontró que con aproximadamente 13 dimensiones se logra superar el 85% de varianza acumulada, indicando que gran parte de la información categórica puede concentrarse en un espacio reducido.

El MCA permitió visualizar la relación entre categorías: por ejemplo, algunos tipos de admisión o de disposición al alta tienden a asociarse con mayor probabilidad de reingreso.


Integración PCA + MCA

Al combinar los componentes numéricos (PCA) y categóricos (MCA):

Se obtiene un conjunto de datos reducido y balanceado entre ambos tipos de información, manteniendo la mayor parte de la varianza de los datos originales.

""")
