# -*- coding: utf-8 -*-
df["readmitted"].value_counts().plot(kind="bar", ax=ax, color=["#66c2a5","#fc8d62","#8da0cb"])
ax.set_title("Distribución de readmisiones")
st.pyplot(fig)


# =========================
# Split de datos
# =========================
x= df.drop("readmitted", axis=1)
y= df["readmitted"]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=42)


# =========================
# PCA
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train[num_cols_pca])


pca = PCA()
X_pca = pca.fit_transform(X_train_scaled)


explained_var = np.cumsum(pca.explained_variance_ratio_)


fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(1, len(explained_var)+1), explained_var, marker='o')
ax.axhline(y=0.85, color='r')
ax.set_title('Varianza acumulada PCA')
st.pyplot(fig)


st.write(f"Componentes necesarios (85% varianza): {np.argmax(explained_var>=0.85)+1}")


# =========================
# MCA
# =========================
try:
import prince
except ImportError:
st.error("El paquete 'prince' es requerido para MCA. Agregue 'prince' a requirements.txt.")
st.stop()


X_train_cat = x_train[cat_cols_mca].astype(str)


try:
mca = prince.MCA(n_components=15, random_state=42)
mca = mca.fit(X_train_cat)
st.success("MCA ejecutado correctamente.")
except Exception as e:
st.error(f"Error en MCA: {e}")
st.stop()


X_mca = mca.transform(X_train_cat)


# =========================
# Integración PCA + MCA
# =========================
n_pca = np.argmax(explained_var>=0.85)+1
X_pca_reduced = X_pca[:,:n_pca]


cum_var_exp = np.cumsum(mca.eigenvalues_ / mca.eigenvalues_.sum())
n_mca = np.argmax(cum_var_exp>=0.85)+1
X_mca_reduced = X_mca.iloc[:,:n_mca].values


X_reduced = np.hstack((X_pca_reduced,X_mca_reduced))
col_names = [f"PCA_{i+1}" for i in range(n_pca)] + [f"MCA_{i+1}" for i in range(n_mca)]
X_reduced_df = pd.DataFrame(X_reduced, columns=col_names, index=x_train.index)


st.subheader("Datos reducidos PCA+MCA")
st.write(X_reduced_df.head())


# =========================
# KNN y Árbol (placeholder)
# =========================
X_reduced_df['target'] = y_train.values
st.write("Shape final:", X_reduced_df.shape)


st.success("El script se ejecutó hasta el final sin errores.")


