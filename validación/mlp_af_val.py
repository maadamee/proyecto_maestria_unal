import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score

def indice_acuerdo_ratio(a, d, N):
    return (a + d) / N if N != 0 else 0

def sensibilidad(a, c):
    return a / (a + c) if (a + c) != 0 else 0

def especificidad(d, b):
    return d / (b + d) if (b + d) != 0 else 0

def valor_predictivo_positivo(a, b):
    return a / (a + b) if (a + b) != 0 else 0

def valor_predictivo_negativo(d, c):
    return d / (c + d) if (c + d) != 0 else 0

def area_roc(sensibilidad, especificidad):
    return (sensibilidad + especificidad) / 2  # Aproximación de AUC ROC

# Datos de prueba
y_real = ["lu", "bfb", "adw", "pc", "tpf", "tbf", "fddf", "twdf", "tdf", "tdf", 
          "sbut", "sf", "sf", "adw", "pc", "bff", "bcf", "bfb", "bcf", "bff", 
          "bpf", "bpf", "sbut", "tpf", "adw", "bff", "tbf", "bff", "bfb", "tff"]
y_predicho = ["pc", "bcf", "ur", "abw", "lu", "tbf", "fddf", "twdf", "tdf", "tdf", 
              "sbut", "sf", "sf", "pc", "abw", "bff", "bcf", "bfb", "bcf", "bff", 
              "bpf", "bpf", "lu", "tpf", "sf", "bff", "tscc", "bff", "bff", "bff"]
categorias_esperadas = ["bfb", "tbf", "tdf", "twdf", "bff", "tff", "tffss", "sccub", 
                        "tscc", "bcf", "tcf", "fddf", "fdff", "bpf", "tpf", "sbub", 
                        "sbut", "dkws", "abw", "adw", "sf", "lu", "ur", "pc"]

# 1️⃣ **Crear matrices de confusión**
categorias_presentes = sorted(set(y_real + y_predicho))  # Clases que aparecen en los datos
mat_conf = confusion_matrix(y_real, y_predicho, labels=categorias_presentes)

# 2️⃣ **Evitar división por cero en la normalización**
sumas_filas = mat_conf.sum(axis=1, keepdims=True)
sumas_filas[sumas_filas == 0] = 1  # Evita división por cero
mat_conf_normalizada = mat_conf.astype('float') / sumas_filas

# 3️⃣ **Índice de acuerdo de grupo (iAp)**
iAp = np.trace(mat_conf) / np.sum(mat_conf)

# 4️⃣ **Índice de acuerdo ratio (iA)**
aciertos_totales = np.trace(mat_conf)  # Suma de la diagonal principal
total_predicciones = np.sum(mat_conf)  # Total de casos evaluados
iA = aciertos_totales / total_predicciones

# 5️⃣ **Cálculo de Kappa y Kappa ponderado**
kappa = cohen_kappa_score(y_real, y_predicho)
kappa_w = cohen_kappa_score(y_real, y_predicho, weights="quadratic")

# 6️⃣ **Convertir a variables binarias con pd.get_dummies()**
y_real_bin = pd.get_dummies(y_real, dtype=int)
y_pred_bin = pd.get_dummies(y_predicho, dtype=int)

# Asegurar que ambas matrices tengan las mismas columnas (agregar columnas faltantes con 0)
y_real_bin, y_pred_bin = y_real_bin.align(y_pred_bin, join="outer", axis=1, fill_value=0)

# 🔹 **Evitar el error eliminando clases con solo un valor (0 o 1)**
clases_validas = y_real_bin.columns[(y_real_bin.nunique() > 1)]
y_real_bin = y_real_bin[clases_validas]
y_pred_bin = y_pred_bin[clases_validas]

# 7️⃣ **Cálculo de ROC-AUC si hay clases válidas**
if not y_real_bin.empty:
    roc_auc = roc_auc_score(y_real_bin, y_pred_bin, average="macro", multi_class="ovr")
else:
    roc_auc = np.nan  # No se puede calcular ROC si no hay clases válidas

# 8️⃣ **Función para calcular métricas**
def calcular_metricas(matriz):
    diag = np.diag(matriz)  # Verdaderos positivos (TP)
    falsos_negativos = np.sum(matriz, axis=1) - diag  # FN
    falsos_positivos = np.sum(matriz, axis=0) - diag  # FP
    verdaderos_negativos = np.sum(matriz) - (diag + falsos_negativos + falsos_positivos)  # TN

    sensibilidad = diag / (diag + falsos_negativos + 1e-8)  # S
    especificidad = verdaderos_negativos / (verdaderos_negativos + falsos_positivos + 1e-8)  # E
    valor_predictivo_positivo = diag / (diag + falsos_positivos + 1e-8)  # p+
    valor_predictivo_negativo = verdaderos_negativos / (verdaderos_negativos + falsos_negativos + 1e-8)  # p-
    indice_falsos_negativos = falsos_negativos / (falsos_negativos + diag + 1e-8)  # FN Ratio

    return {
        "Sensibilidad": sensibilidad,
        "Especificidad": especificidad,
        "Valor Predictivo Positivo": valor_predictivo_positivo,
        "Valor Predictivo Negativo": valor_predictivo_negativo,
        "Índice de Falsos Negativos": indice_falsos_negativos
    }

metricas = calcular_metricas(mat_conf_normalizada)

# 9️⃣ **Mostrar resultados**
print(f"Índice de Acuerdo de Grupo (iAp): {iAp:.4f}")
print(f"Índice de Acuerdo Ratio (iA): {iA:.4f}")
print(f"Kappa: {kappa:.4f}")
print(f"Kappa Ponderado: {kappa_w:.4f}")
print(f"Área bajo la curva ROC: {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC no se puede calcular")

for key, values in metricas.items():
    print(f"{key}: {np.mean(values):.4f}")
    
categorias_presentes = sorted(set(y_real + y_predicho))  # Clases presentes en los datos

# 1️⃣ **Crear matriz de confusión**
mat_conf = confusion_matrix(y_real, y_predicho, labels=categorias_presentes)

# 2️⃣ **Normalizar la matriz de confusión**
sumas_filas = mat_conf.sum(axis=1, keepdims=True)
sumas_filas[sumas_filas == 0] = 1  # Evitar división por cero
mat_conf_normalizada = mat_conf.astype('float') / sumas_filas

# 🔹 **Matriz de Confusión (Absoluta)**
plt.figure(figsize=(8, 6))
sns.heatmap(mat_conf, annot=True, fmt="d", cmap="Blues", xticklabels=categorias_presentes, yticklabels=categorias_presentes)
plt.title("Matriz de Confusión")
plt.xlabel("Predicciones")
plt.ylabel("Reales")
plt.show()

# 🔹 **Matriz de Confusión Normalizada**
plt.figure(figsize=(8, 6))
sns.heatmap(mat_conf_normalizada, annot=True, fmt=".2f", cmap="Blues", xticklabels=categorias_presentes, yticklabels=categorias_presentes)
plt.title("Matriz de Confusión Normalizada")
plt.xlabel("Predicciones")
plt.ylabel("Reales")
plt.show()

# 3️⃣ **Cálculo de métricas por categoría**
resultados = []
for i, categoria in enumerate(categorias_presentes):
    a = mat_conf[i, i]  # Verdaderos positivos
    b = mat_conf[:, i].sum() - a  # Falsos positivos
    c = mat_conf[i, :].sum() - a  # Falsos negativos
    d = mat_conf.sum() - (a + b + c)  # Verdaderos negativos
    N = mat_conf.sum()  # Total de observaciones

    resultados.append({
        "Categoría": categoria,
        "iA (Índice de acuerdo ratio)": indice_acuerdo_ratio(a, d, N),
        "S (Sensibilidad)": sensibilidad(a, c),
        "fn (Índice de falsos negativos)": c / (a + c) if (a + c) != 0 else 0,
        "E (Especificidad)": especificidad(d, b),
        "fp (Índice de falsos positivos)": b / (b + d) if (b + d) != 0 else 0,
        "p+ (Índice predictivo positivo)": valor_predictivo_positivo(a, b),
        "p- (Índice predictivo negativo)": valor_predictivo_negativo(d, c),
        "ROC (Área bajo la curva ROC)": area_roc(sensibilidad(a, c), especificidad(d, b))
    })

# 🔹 **Guardar resultados en CSV**
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("mlp_af_rval.csv", index=False)