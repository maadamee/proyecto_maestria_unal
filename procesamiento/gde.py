import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

# =============================================================================
# 1. Generador para datos de entrenamiento
# =============================================================================

#Tomar la matriz de frecuencias de modos de falla
df = pd.read_csv("tf_entrenamiento.txt", sep="\t")

# Calcular el número total de muestras deseado por clase
max_samples_per_class = df["FrS"].max()  # Tomamos la clase más representada

# Generar datos sintéticos
synthetic_data = []
for index, row in df.iterrows():
    modo_falla = row["Mfalla"]
    total_casos_originales = row["FrS"]
    features = row.iloc[0:32]
    
    # Generar exactamente max_samples_per_class para cada clase
    for _ in range(max_samples_per_class):
        case = {}
        for feature_name, freq in features.items():
            prob = freq / total_casos_originales  # Probabilidad basada en los datos originales
            case[feature_name] = np.random.choice([1, 0], p=[prob, 1 - prob])  # Mantiene proporción real
        case["Mfalla"] = modo_falla
        synthetic_data.append(case)

# Crear DataFrame
synthetic_df = pd.DataFrame(synthetic_data)
# 🔀 Mezclar aleatoriamente los datos
synthetic_df = shuffle(synthetic_df, random_state=42).reset_index(drop=True)

# Guardar en un archivo .txt
synthetic_df.to_csv("synthetic_data.txt", sep="\t", index=False)

# Separar características y etiquetas
X = synthetic_df.drop("Mfalla", axis=1).values
y = synthetic_df["Mfalla"].values

# Codificar etiquetas
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

print("✅ Datos sintéticos guardados en 'synthetic_data.txt'")
