import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import random

def generar_caso_de_uso_estabilidad_clusters():
    """
    Genera un caso de uso aleatorio para evaluar la estabilidad de clustering.
    Compara múltiples ejecuciones de KMeans sobre submuestras (bootstrap)
    para medir la consistencia de los grupos mediante el ARI.
    """
    
    # --- 1. COMPONENTE ALEATORIO (Generación de Datos de Clusters) ---
    n_muestras = random.randint(300, 600)
    n_features = 2
    
    # Creamos centros reales para simular "estabilidad" o "caos"
    # Si los centros están lejos, la estabilidad será alta (~1.0)
    # Si están muy cerca, será baja
    separacion = random.uniform(1.0, 5.0)
    centros = np.array([[0, 0], [separacion, separacion], [0, separacion]])
    
    X = np.vstack([
        np.random.normal(c, 0.5, (n_muestras // 3, n_features)) 
        for c in centros
    ])
    
    n_clusters_input = 3
    n_bootstrap_input = random.randint(8, 15)
    
    input_dict = {
        "X": X,
        "n_clusters": n_clusters_input,
        "n_bootstrap": n_bootstrap_input
    }

    # --- 2. SOLUCIÓN (Lógica interna para generar el Output) ---
    def preparar_datos_estabilidad(X_data, k, n_iter):
        lista_etiquetas = []
        n = X_data.shape[0]
        
        # Generar múltiples particiones mediante Bootstrap
        for _ in range(n_iter):
            # 1. Remuestreo con reemplazo usando NumPy
            indices = np.random.choice(n, n, replace=True)
            X_resampled = X_data[indices]
            
            # 2. Ajustar modelo KMeans
            # Usamos random_state variable para probar estabilidad real
            km = KMeans(n_clusters=k, n_init=10)
            km.fit(X_resampled)
            
            # 3. Predecir sobre el dataset ORIGINAL para poder comparar
            # (Es vital predecir sobre la misma base X para calcular el ARI)
            etiquetas_completas = km.predict(X_data)
            lista_etiquetas.append(etiquetas_completas)
            
        # 4. Calcular ARI entre todos los pares posibles de iteraciones
        scores_ari = []
        for i in range(len(lista_etiquetas)):
            for j in range(i + 1, len(lista_etiquetas)):
                score = adjusted_rand_score(lista_etiquetas[i], lista_etiquetas[j])
                scores_ari.append(score)
                
        # Calcular métricas de estabilidad
        ari_medio = float(np.mean(scores_ari))
        ari_std = float(np.std(scores_ari))
        
        resultado = {
            "estabilidad_media": round(ari_medio, 4),
            "desviacion_estabilidad": round(ari_std, 4),
            "n_comparaciones": len(scores_ari)
        }
        
        return resultado

    # --- 3. GENERACIÓN DEL OUTPUT ---
    output_dict = preparar_datos_estabilidad(
        input_dict["X"], 
        input_dict["n_clusters"], 
        input_dict["n_bootstrap"]
    )
    
    return input_dict, output_dict

# --- Ejemplo de ejecución ---
# inputs, outputs = generar_caso_de_uso_estabilidad_clusters()
# print(f"Evaluando {inputs['n_clusters']} clusters con {inputs['n_bootstrap']} bootstraps.")
# print(f"Resultado: {outputs}")