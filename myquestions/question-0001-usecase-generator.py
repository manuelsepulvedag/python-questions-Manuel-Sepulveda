import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random

def generar_caso_de_uso_atractor():
    """
    Genera un caso de uso aleatorio para la reconstrucción de atractores.
    Incluye la implementación interna (solución) para generar el output esperado.
    """
    
    # --- 1. COMPONENTE ALEATORIO (Generación de Datos Sintéticos) ---
    longitud = random.randint(150, 300)
    t = np.linspace(0, 100, longitud)
    # Creamos una señal compleja: combinación de dos senos y ruido
    freq1 = random.uniform(0.05, 0.2)
    freq2 = random.uniform(0.3, 0.6)
    senal_base = np.sin(freq1 * t) + 0.5 * np.cos(freq2 * t)
    ruido = np.random.normal(0, 0.05, longitud)
    serie_input = pd.Series(senal_base + ruido)
    
    # Parámetros de la reconstrucción
    dim_input = random.randint(4, 12)
    retardo_input = random.randint(1, 4)
    
    # Definición del diccionario de entrada
    input_dict = {
        "serie": serie_input,
        "dimension": dim_input,
        "retardo": retardo_input
    }

    # --- 2. SOLUCIÓN (Lógica interna para generar el Output) ---
    def preparar_datos_atractor(serie, dimension, retardo):
        # Cálculo de la longitud efectiva tras el lag-embedding
        n = len(serie)
        n_final = n - (dimension - 1) * retardo
        
        if n_final <= 0:
            return pd.DataFrame(), 0.0
            
        # Creación de la matriz de lags con NumPy
        # Cada columna 'i' es la serie desplazada por i * retardo
        lags = []
        valores = serie.values
        for i in range(dimension):
            inicio = i * retardo
            fin = inicio + n_final
            lags.append(valores[inicio:fin])
        
        matriz_m = np.column_stack(lags)
        
        # Reducción de dimensionalidad a 3 componentes (Espacio de Fases)
        pca = PCA(n_components=3)
        componentes_transformadas = pca.fit_transform(matriz_m)
        varianza_explicada = float(np.sum(pca.explained_variance_ratio_))
        
        # Formateo a DataFrame de Pandas
        df_pca = pd.DataFrame(
            componentes_transformadas, 
            columns=['PC1', 'PC2', 'PC3']
        )
        
        return df_pca, varianza_explicada

    # --- 3. GENERACIÓN DEL OUTPUT ---
    df_resultado, varianza_resultado = preparar_datos_atractor(
        serie_input, dim_input, retardo_input
    )
    
    output_dict_o_tuple = (df_resultado, varianza_resultado)
    
    return input_dict, output_dict_o_tuple

# --- Ejemplo de ejecución ---
# mi_input, mi_output = generar_caso_de_uso_atractor()
# print("Input - Dimensión elegida:", mi_input['dimension'])
# print("Output - Primeras filas del atractor:\n", mi_output[0].head())
# print(f"Output - Varianza capturada: {mi_output[1]:.4%}")