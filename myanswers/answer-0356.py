import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def predecir_resistencia_concreto(df: pd.DataFrame, target_col: str) -> RandomForestRegressor:
    """
    Separa las variables predictoras de la variable objetivo y entrena 
    un modelo RandomForestRegressor para predecir la resistencia del concreto.
    
    Parámetros:
    df (pd.DataFrame): El DataFrame que contiene los datos de las mezclas y resistencias.
    target_col (str): El nombre de la columna objetivo (ej. 'resistencia_28_dias').
    
    Devuelve:
    RandomForestRegressor: El modelo de Bosque Aleatorio ya entrenado.
    """
    # 1. Separar las variables predictoras (X) y la variable objetivo (y)
    # X contendrá todo excepto la columna que queremos predecir
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Inicializar el modelo RandomForestRegressor
    # Usamos random_state para que los resultados sean replicables
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 3. Entrenar el modelo con los datos
    modelo.fit(X, y)
    
    # 4. Devolver el modelo entrenado
    return modelo