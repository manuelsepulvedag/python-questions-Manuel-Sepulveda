import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def analizar_eficiencia_energetica(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Limpia, escala de forma robusta frente a outliers y reduce a 3 componentes 
    principales el conjunto de características de consumo energético de edificios.
    
    Parámetros:
    df (pd.DataFrame): DataFrame con los datos de sensores de los edificios.
    target_col (str): Nombre de la columna objetivo (ej. 'consumo_total_kwh') que se excluirá de las características.
    
    Devuelve:
    tuple: (X_pca, pca_entrenado) donde X_pca es un numpy array con las 3 componentes, 
           y pca_entrenado es el objeto PCA ajustado.
    """
    # 0. Separar las características (X) eliminando la columna objetivo
    X = df.drop(columns=[target_col])
    
    # 1. Identificar las columnas numéricas de las características
    X_num = X.select_dtypes(include=[np.number])
    
    # Rellenar los valores faltantes utilizando la mediana
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_num)
    
    # 2. Escalado Robusto para mitigar el impacto de los picos aislados (outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 3. Reducción de dimensionalidad a las 3 componentes más importantes
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # 4. Retornar la matriz transformada y el objeto PCA entrenado
    return X_pca, pca