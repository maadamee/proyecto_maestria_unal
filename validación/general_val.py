import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score

# Datos de prueba
y_real = ["lu", "bfb", "adw", "pc", "tpf", "tbf", "fddf", "twdf", "tdf", "tdf", 
          "sbut", "sf", "sf", "adw", "pc", "bff", "bcf", "bfb", "bcf", "bff", 
          "bpf", "bpf", "sbut", "tpf", "adw", "bff", "tbf", "bff", "bfb", "tff"]
y_predicho = ["ur", "bcf", "adw", "adw", "lu", "tbf", "fddf", "twdf", "tdf", "tdf",
              "sbut", "sf", "sf", "ur", "abw", "bff", "bcf", "fdff", "bcf", "bcf",
              "bpf", "bpf", "lu", "tpf", "sf", "tdf", "tbf", "bcf", "bff", "bff"]

categorias_esperadas = ["bfb", "tbf", "tdf", "twdf", "bff", "tff", "tffss", "sccub", 
                        "tscc", "bcf", "tcf", "fddf", "fdff", "bpf", "tpf", "sbub", 
                        "sbut", "dkws", "abw", "adw", "sf", "lu", "ur", "pc"]

# Obtener categorías presentes en los datos
categorias_presentes = sorted(set(y_real + y_predicho))

# 1️⃣ **Crear y normalizar la matriz de confusión**
mat_conf = confusion_matrix(y_real, y_predicho, labels=categorias_presentes)
sumas_filas = mat_conf.sum(axis=1, keepdims=True)
sumas_filas[sumas_filas == 0] = 1  # Evita división por cero
mat_conf_normalizada = mat_conf.astype('float') / sumas_filas

# 2️⃣ **Índice de acuerdo (iA)**
iA = np.trace(mat_conf_normalizada) / np.sum(mat_conf_normalizada)

# 3️⃣ **Cálculo de Kappa y Kappa ponderado**
kappa = cohen_kappa_score(y_real, y_predicho)
kappa_w = cohen_kappa_score(y_real, y_predicho, weights="quadratic")

# 4️⃣ **Convertir a variables binarias**
y_real_bin = pd.get_dummies(y_real, dtype=int)
y_pred_bin = pd.get_dummies(y_predicho, dtype=int)

# Asegurar que ambas matrices tengan las mismas columnas
y_real_bin, y_pred_bin = y_real_bin.align(y_pred_bin, join="outer", axis=1, fill_value=0)

# Filtrar clases válidas (aquellas con más de un valor distinto)
clases_validas = y_real_bin.columns[(y_real_bin.nunique() > 1)]
y_real_bin = y_real_bin[clases_validas]
y_pred_bin = y_pred_bin[clases_validas]

def indice_acuerdo(a, d, N):
    return (a + d) / N if N != 0 else 0

def indice_acuerdo_ratio(a, b, c, d, N):
    return (a + d) / (a + b + c + d) if N != 0 else 0

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

# 5️⃣ **Cálculo de ROC-AUC**
if not y_real_bin.empty:
    roc_auc = roc_auc_score(y_real_bin, y_pred_bin, average="macro", multi_class="ovr")
else:
    roc_auc = np.nan  # No se puede calcular ROC si no hay clases válidas

# 6️⃣ **Cálculo de métricas por categoría**
resultados = []
for i, categoria in enumerate(categorias_presentes):
    a = mat_conf_normalizada[i, i]  # Verdaderos positivos
    b = mat_conf_normalizada[:, i].sum() - a  # Falsos positivos
    c = mat_conf_normalizada[i, :].sum() - a  # Falsos negativos
    d = np.sum(mat_conf_normalizada) - (a + b + c)  # Verdaderos negativos
    N = np.sum(mat_conf)  # Total de casos
    
    resultados.append({
        "Categoría": categoria,
        "iAp": indice_acuerdo(a, d, N),
        "iAr": indice_acuerdo_ratio (a, b, c, d, N),
        "S": sensibilidad(a, c),
        "fn": c / (a + c + 1e-8),
        "E": especificidad(d, b),
        "fp": b / (b + d + 1e-8),
        "p+": valor_predictivo_positivo(a, b),
        "p-": valor_predictivo_negativo(d, c),
        "ROC": area_roc(sensibilidad(a, c), especificidad(d, b))
    })

# 7️⃣ **Guardar resultados en CSV**
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("lstm_av_rval.csv", index=False)

# 8️⃣ **Visualización de la matriz de confusión**
plt.figure(figsize=(8, 6))
sns.heatmap(mat_conf, annot=True, fmt="d", cmap="Blues", xticklabels=categorias_presentes, yticklabels=categorias_presentes)
plt.title("Matriz de Confusión")
plt.xlabel("Predicciones inferencia LSTM_av")
plt.ylabel("Validación humana")
plt.show()

# 9️⃣ **Mostrar métricas generales**                                                                                                                                       
print(f"Índice de Acuerdo Ratio (iA): {iA:.4f}")
print(f"Área bajo la curva ROC: {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC no se puede calcular")

# 🔹 1️⃣ Promediar las métricas globales
iAp_promedio = df_resultados["iAp"].mean()
iAr_promedio = df_resultados["iAr"].mean()
sensibilidad_promedio = df_resultados["S"].mean()
especificidad_promedio = df_resultados["E"].mean()
valor_predictivo_positivo_promedio = df_resultados["p+"].mean()
valor_predictivo_negativo_promedio = df_resultados["p-"].mean()
fn_promedio = df_resultados["fn"].mean()
fp_promedio = df_resultados["fp"].mean()
roc_promedio = df_resultados["ROC"].mean()

# 🔹 2️⃣ Mostrar métricas globales promediadas
print("\n🔹 **Métricas Globales Promediadas del Modelo**")
print(f"{iAp_promedio:.4f}")
print(f"{kappa:.4f}")
print(f"{kappa_w:.4f}")
print(f"{iAr_promedio:.4f}")
print(f"{sensibilidad_promedio:.4f}")
print(f"{fn_promedio:.4f}")
print(f"{especificidad_promedio:.4f}")
print(f"{fp_promedio:.4f}")
print(f"{valor_predictivo_positivo_promedio:.4f}")
print(f"{valor_predictivo_negativo_promedio:.4f}")
print(f"{roc_promedio:.4f}")
