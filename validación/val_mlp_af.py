import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

# Definir etiquetas reales y predichas
y_real = ["lu", "bfb", "adw", "pc", "tpf", "tbf", "fddf", "twdf", "tdf", "tdf", "sbut", "sf", "sf", "adw", "pc", "bff", "bcf", "bfb", "bcf", "bff", "bpf", "bpf", "sbut", "tpf", "adw", "bff", "tbf", "bff", "bfb", "tff"]
y_predicho = ["pc", "bcf", "ur", "abw", "lu", "tbf", "fddf", "twdf", "tdf", "tdf", "sbut", "sf", "sf", "pc", "abw", "bff", "bcf", "bfb", "bcf", "bff", "bpf", "bpf", "lu", "tpf", "sf", "bff", "tscc", "bff", "bff", "bff"]

# Definir las 24 categorías de salida esperadas en el modelo
categorias_esperadas = ["bfb", "tbf", "tdf", "twdf", "bff", "tff", "tffss", "sccub", "tscc", "bcf", "tcf", "fddf", "fdff", "bpf", "tpf", "sbub", "sbut", "dkws", "abw", "adw", "sf", "lu", "ur", "pc"]

label_encoder = LabelEncoder()
label_encoder.fit(categorias_esperadas)
y_real_numerico = label_encoder.transform(y_real)
y_predicho_numerico = label_encoder.transform(y_predicho)

matriz_confusion = confusion_matrix(y_real_numerico, y_predicho_numerico, labels=range(len(categorias_esperadas)))

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=categorias_esperadas, yticklabels=categorias_esperadas)
plt.xlabel("Predicción motor de inferencia")
plt.ylabel("Experto humano")
plt.title("Matriz de Confusión - MLP_af")
plt.show()

# Matriz de pesos personalizada (Ejemplo basado en la tabla C-3)
W = np.ones((len(categorias_esperadas), len(categorias_esperadas)))
np.fill_diagonal(W, 0)  # Concordancias tienen peso 0

# Definir matriz de pesos personalizada
num_classes = len(categorias_esperadas)
W = np.zeros((num_classes, num_classes))
for i in range(num_classes):
    for j in range(num_classes):
        if i != j:
            W[i, j] = 2 if abs(i - j) == 1 else 4  # Penalización basada en la tabla C-3

# Calcular Kappa ponderado
kappa_w = cohen_kappa_score(y_real_numerico, y_predicho_numerico, weights=W)

# Funciones para las métricas
def indice_acuerdo(a, b, c, d):
    return (a + d) / (a + b + c + d) if (a + b + c + d) != 0 else 0

def calcular_kappa(a, b, c, d):
    N = a + b + c + d
    po = (a + d) / N if N != 0 else 0
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (N**2) if N != 0 else 0
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0

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
    return (sensibilidad + especificidad) / 2

# Calcular métricas a partir de la matriz de confusión
resultados_metricas = []
for i, categoria in enumerate(categorias_esperadas):
    a = matriz_confusion[i, i]  # Verdaderos positivos (TP)
    b = np.sum(matriz_confusion[:, i]) - a  # Falsos positivos (FP)
    c = np.sum(matriz_confusion[i, :]) - a  # Falsos negativos (FN)
    d = np.sum(matriz_confusion) - (a + b + c)  # Verdaderos negativos (TN)
    N = a + b + c + d

    resultados_metricas.append({
        "Categoría": categoria,
        "iAp (Índice de acuerdo de grupo)": indice_acuerdo(a, b, c, d),
        "Kappa (Índice kappa)": calcular_kappa(a, b, c, d),
        "Kappa Ponderado": kappa_w,
        "iA (Índice de acuerdo ratio)": indice_acuerdo_ratio(a, d, N),
        "S (Sensibilidad)": sensibilidad(a, c),
        "fn (Índice de falsos negativos)": c / (a + c) if (a + c) != 0 else 0,
        "E (Especificidad)": especificidad(d, b),
        "fp (Índice de falsos positivos)": b / (b + d) if (b + d) != 0 else 0,
        "p+ (Índice predictivo positivo)": valor_predictivo_positivo(a, b),
        "p- (Índice predictivo negativo)": valor_predictivo_negativo(d, c),
        "ROC (Área bajo la curva ROC)": area_roc(sensibilidad(a, c), especificidad(d, b))
    })

# Calcular métricas por categoría
resultados_metricas = []
for i, categoria in enumerate(categorias_esperadas):
    a = matriz_confusion[i, i]
    b = np.sum(matriz_confusion[:, i]) - a
    c = np.sum(matriz_confusion[i, :]) - a
    d = np.sum(matriz_confusion) - (a + b + c)
    N = a + b + c + d

    resultados_metricas.append({
        "Categoría": categoria,
        "iAp": indice_acuerdo(a, b, c, d),
        "Kappa": calcular_kappa(a, b, c, d),
        "Kappa_w": kappa_w,
        "iA": (a + d) / N if N != 0 else 0,
        "S": sensibilidad(a, c),
        "fn": c / (a + c) if (a + c) != 0 else 0,
        "E": especificidad(d, b),
        "fp": b / (b + d) if (b + d) != 0 else 0,
        "p+": valor_predictivo_positivo(a, b),
        "p-": valor_predictivo_negativo(d, c),
        "ROC": area_roc(sensibilidad(a, c), especificidad(d, b))
    })

df_metricas = pd.DataFrame(resultados_metricas)
print("\nMétricas calculadas por categoría:")
print(df_metricas)
df_metricas.to_csv("metricas_categorias_mlp_af.csv", index=False, encoding="utf-8")