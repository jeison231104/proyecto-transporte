import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# 1. DATASET
# ---------------------------
np.random.seed(42)

n = 80

data = pd.DataFrame(
    {
        "hora_salida": np.random.randint(0, 24, n),
        "clima": np.random.choice(["Soleado", "Lluvia", "Nublado"], n),
        "trafico": np.random.choice(["Bajo", "Medio", "Alto"], n),
        "pasajeros": np.random.randint(20, 120, n),
    }
)

# Regla para generar retraso
data["retraso"] = (
    (data["trafico"] == "Alto")
    | ((data["clima"] == "Lluvia") & (data["hora_salida"].between(6, 10)))
).astype(int)

print("DATASET:")
print(data.head())


# ---------------------------
# 2. PREPROCESAMIENTO
# ---------------------------
le = LabelEncoder()

for col in ["clima", "trafico"]:
    data[col] = le.fit_transform(data[col])

X = data.drop("retraso", axis=1)
y = data["retraso"]


# ---------------------------
# 3. DATASET ENTRENAMIENTO Y PRUEBA
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nDATASET ENTRENAMIENTO:")
print(X_train.head())

print("\nDATASET PRUEBA:")
print(X_test.head())


# ---------------------------
# 4. MODELO (ÁRBOL DE DECISIÓN)
# ---------------------------
modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X_train, y_train)


# ---------------------------
# 5. PREDICCIÓN
# ---------------------------
y_pred = modelo.predict(X_test)


# ---------------------------
# 6. RESULTADOS
# ---------------------------
print("\nRESULTADOS:")
print("Precisión:", accuracy_score(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))


# ---------------------------
# 7. VISUALIZACIÓN DEL ÁRBOL
# ---------------------------
plt.figure(figsize=(12, 8))
plot_tree(
    modelo, feature_names=X.columns, class_names=["No retraso", "Retraso"], filled=True
)
plt.title("ÁRBOL DE DECISIÓN - TRANSPORTE")
plt.show()
