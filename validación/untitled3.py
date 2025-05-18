import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Ejemplo de etiquetas reales y predichas (reemplázalo con tus datos)
y_real = ["lu", "bfb", "adw", "pc", "tpf", "tbf", "fddf", "twdf", "tdf", "tdf",
          "sbut", "sf", "sf", "adw", "pc", "bff", "bcf", "bfb", "bcf", "bff",
          "bpf", "bpf", "sbut", "tpf", "adw", "bff", "tbf", "bff", "bfb", "tff"]
y_predicho = ["bfp", "bff", "adw", "adw", "lu", "tbf", "fddf", "twdf", "fdff", "bfb",
              "bpf", "sf", "bpf", "bpf", "abw", "bff", "bff", "bfb", "bcf", "bff",
              "bpf", "bpf", "lu", "tpf", "sf", "bff", "tscc", "bff", "bff", "bff"]

# Obtener todas las categorías únicas
categorias_unicas = sorted(set(y_real) | set(y_predicho))

# Crear un codificador de etiquetas
label_encoder = LabelEncoder()
label_encoder.fit(categorias_unicas)

# Convertir las etiquetas de texto a valores numéricos
y_real_numerico = label_encoder.transform(y_real)
y_predicho_numerico = label_encoder.transform(y_predicho)

# Construir la matriz de confusión
matriz_confusion = confusion_matrix(y_real_numerico, y_predicho_numerico)

# Graficar la matriz de confusión con etiquetas de texto
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=categorias_unicas, yticklabels=categorias_unicas)
plt.xlabel("Predicción motor de inferencia")
plt.ylabel("Experto humano")
plt.title("Matriz de Confusión - DNN_af")
plt.show()