import numpy as np

def calcular_matriz_contingencia(y_real, y_pred):
    """Calcula la matriz de contingencia 2x2."""
    a = np.sum((y_real == 1) & (y_pred == 1))  # Verdaderos positivos
    b = np.sum((y_real == 0) & (y_pred == 1))  # Falsos positivos
    c = np.sum((y_real == 1) & (y_pred == 0))  # Falsos negativos
    d = np.sum((y_real == 0) & (y_pred == 0))  # Verdaderos negativos
    return a, b, c, d

def indice_acuerdo(a, b, c, d):
    return (a + d) / (a + b + c + d)

def sensibilidad(a, c):
    return a / (a + c) if (a + c) != 0 else 0

def especificidad(d, b):
    return d / (b + d) if (b + d) != 0 else 0

def valor_predictivo_positivo(a, b):
    return a / (a + b) if (a + b) != 0 else 0

def valor_predictivo_negativo(d, c):
    return d / (c + d) if (c + d) != 0 else 0

def area_roc(sens, esp):
    return (sens + esp) / 2

def calcular_kappa(a, b, c, d):
    N = a + b + c + d
    po = (a + d) / N
    pc = (((a + c) * (a + b)) + ((b + d) * (c + d))) / (N ** 2)
    return (po - pc) / (1 - pc) if (1 - pc) != 0 else 0

def calcular_kappa_ponderado(a, b, c, d):
    N = a + b + c + d
    qo = 1 - ((a + d) / N)
    qc = 1 - ((((a + c) * (a + b)) + ((b + d) * (c + d))) / (N ** 2))
    return 1 - (qo / qc) if qc != 0 else 0

def indice_acuerdo_ratio(a, d, N):
    return (a + d) / N if N != 0 else 0

def calcular_metricas(y_real, y_pred):
    a, b, c, d = calcular_matriz_contingencia(y_real, y_pred)
    N = a + b + c + d
    resultados = {
        "iAp (Índice de acuerdo de grupo)": indice_acuerdo(a, b, c, d),
        "Kappa (Índice kappa)": calcular_kappa(a, b, c, d),
        "Kappa Ponderado": calcular_kappa_ponderado(a, b, c, d),
        "iA (Índice de acuerdo ratio)": indice_acuerdo_ratio(a, d, N),
        "S (Sensibilidad)": sensibilidad(a, c),
        "fn (Índice de falsos negativos)": c / (a + c) if (a + c) != 0 else 0,
        "E (Especificidad)": especificidad(d, b),
        "fp (Índice de falsos positivos)": b / (b + d) if (b + d) != 0 else 0,
        "p+ (Índice predictivo positivo)": valor_predictivo_positivo(a, b),
        "p- (Índice predictivo negativo)": valor_predictivo_negativo(d, c),
        "ROC (Área bajo la curva ROC)": area_roc(sensibilidad(a, c), especificidad(d, b))
    }
    return resultados

# Ejemplo de uso
probabilidades = np.array([0.0, 0.0, 0.4959, 0.0, 0.0, 0.9999, 0.9938, 0.987, 0.9952, 0.9262,
                           0.3326, 0.9876, 0.7079, 0.0, 0.0, 0.9856, 0.9801, 0.0, 0.9979, 0.0,
                           0.9996, 0.9993, 0.0, 0.8756, 0.0, 0.0, 0.6592, 0.0, 0.0, 0.0])
threshold = 0.5
y_pred = (probabilidades >= threshold).astype(int)
y_real = np.ones(len(probabilidades))  # Valores reales todos en 1

metricas = calcular_metricas(y_real, y_pred)
for nombre, valor in metricas.items():
    print(f"{nombre}: {valor:.4f}")

