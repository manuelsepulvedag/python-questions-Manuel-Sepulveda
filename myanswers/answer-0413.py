import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def analizar_prioridad_retencion(df: pd.DataFrame, target_col: str, clv_col: str, threshold: float = 0.7) -> pd.DataFrame:
    """
    Identifica clientes de alto valor con riesgo de abandono (churn).
    
    Parámetros:
    df (pd.DataFrame): DataFrame con los datos de los clientes.
    target_col (str): Nombre de la columna objetivo (1 = Churn, 0 = Se queda).
    clv_col (str): Nombre de la columna que contiene el Customer Lifetime Value.
    threshold (float): Umbral de probabilidad para considerar riesgo de churn.
    
    Devuelve:
    pd.DataFrame: Clientes de alta prioridad ordenados por probabilidad de churn descrita.
    """
    # Para evitar modificar el DataFrame original que entra como argumento
    df_trabajo = df.copy()
    
    # 1. Eliminar filas donde la columna objetivo sea nula
    df_trabajo = df_trabajo.dropna(subset=[target_col])
    
    # 2. Separar temporalmente el target y el CLV para no aplicarles One-Hot Encoding si son numéricos
    # (o asegurar que el encoding solo afecte a las variables categóricas de predicción)
    y = df_trabajo[target_col]
    clv_serie = df_trabajo[clv_col]
    
    # Obtenemos las características (X) eliminando la columna objetivo
    X = df_trabajo.drop(columns=[target_col])
    
    # Aplicar One-Hot Encoding a las variables categóricas en X
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 3. Entrenar el modelo RandomForestClassifier
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_encoded, y)
    
    # 4. Obtener las probabilidades de pertenecer a la clase 1 (Churn)
    # predict_proba devuelve [prob_clase_0, prob_clase_1], tomamos la posición [:, 1]
    probabilidades_churn = modelo.predict_proba(X_encoded)[:, 1]
    
    # 5. Agregar las nuevas columnas al DataFrame de trabajo
    df_trabajo['probabilidad_churn'] = probabilidades_churn
    
    # Calcular el percentil 80 del CLV
    percentil_80_clv = df_trabajo[clv_col].quantile(0.8)
    
    # Crear la columna booleana 'prioridad_alta' basada en las dos condiciones
    df_trabajo['prioridad_alta'] = (df_trabajo['probabilidad_churn'] >= threshold) & (df_trabajo[clv_col] > percentil_80_clv)
    
    # 6. Filtrar únicamente los clientes de alta prioridad
    clientes_prioritarios = df_trabajo[df_trabajo['prioridad_alta'] == True]
    
    # Ordenar de mayor a menor probabilidad_churn
    clientes_prioritarios = clientes_prioritarios.sort_values(by='probabilidad_churn', ascending=False)
    
    return clientes_prioritarios