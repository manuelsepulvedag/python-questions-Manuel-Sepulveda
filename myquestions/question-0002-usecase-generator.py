import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import random

def generar_caso_de_uso_umbral_optimo():
    """
    Genera un caso de uso aleatorio para la optimización de umbrales de decisión
    basado en una matriz de costos (Falsos Negativos vs Falsos Positivos).
    Incluye la implementación interna para generar el output esperado.
    """
    
    # --- 1. COMPONENTE ALEATORIO (Generación de Datos Sintéticos) ---
    n_muestras = random.randint(500, 1000)
    
    # Generamos etiquetas reales (y_true) con cierto desequilibrio
    y_true = np.random.choice([0, 1], size=n_muestras, p=[0.7, 0.3])
    
    # Generamos probabilidades (y_probs) simulando un modelo ruidoso
    # Los positivos tienden a tener valores más altos, pero con solapamiento
    y_probs = np.where(y_true == 1, 
                       np.random.beta(5, 2, n_muestras), 
                       np.random.beta(2, 5, n_muestras))
    
    # Costos aleatorios: el Falso Negativo suele ser mucho más caro
    costo_fn = float(random.randint(50, 200))
    costo_fp = float(random.randint(5, 30))
    
    input_dict = {
        "y_true": y_true,
        "y_probs": y_probs,
        "costo_fn": costo_fn,
        "costo_fp": costo_fp
    }

    # --- 2. SOLUCIÓN (Lógica interna para generar el Output) ---
    def preparar_datos_umbral(y_true, y_probs, c_fn, c_fp):
        # Creamos 100 umbrales candidatos entre 0 y 1
        umbrales = np.linspace(0, 1, 100)
        costos = []
        
        for u in umbrales:
            # Convertimos probabilidades a predicciones binarias según el umbral
            y_pred = (y_probs >= u).astype(int)
            
            # Obtenemos la matriz de confusión: [[TN, FP], [FN, TP]]
            # Si el modelo solo predice una clase, controlamos las dimensiones
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fp = cm[0, 1]
            fn = cm[1, 0]
            
            # Cálculo del costo total para este umbral
            costo_total = (fn * c_fn) + (fp * c_fp)
            costos.append(costo_total)
            
        # Encontrar el índice del costo mínimo
        idx_min = np.argmin(costos)
        
        resultado = {
            "umbral_optimo": round(float(umbrales[idx_min]), 4),
            "costo_minimo": float(costos[idx_min]),
            "n_muestras": int(len(y_true))
        }
        
        return resultado

    # --- 3. GENERACIÓN DEL OUTPUT ---
    output_dict = preparar_datos_umbral(
        input_dict["y_true"], 
        input_dict["y_probs"], 
        input_dict["costo_fn"], 
        input_dict["costo_fp"]
    )
    
    return input_dict, output_dict

# --- Ejemplo de ejecución ---
# input_data, output_data = generar_caso_de_uso_umbral_optimo()
# print(f"Costo FN: {input_data['costo_fn']} | Costo FP: {input_data['costo_fp']}")
# print(f"Resultado Óptimo: {output_data}")