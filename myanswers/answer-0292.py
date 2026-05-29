import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def segmentar_zonas_sismicas(df: pd.DataFrame, n_clusters: int, n_components_pca: int) -> tuple:
    """
    Segmenta eventos sísmicos en clusters homogéneos utilizando un pipeline robusto
    de Imputación -> Escalado -> PCA -> Clustering Jerárquico.
    
    Devuelve las etiquetas de los clusters y un DataFrame resumen con las medianas originales.
    """
    # 1. Seleccionar solo las columnas numéricas
    df_num = df.select_dtypes(include=[np.number])
    n_cols = df_num.shape[1]
    
    # 2. Imputar valores faltantes con la mediana
    imputer = SimpleImputer(strategy='median')
    # Convertimos de nuevo a DataFrame para mantener los nombres de las columnas
    datos_imputados = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
    
    # 3. Escalar los datos con RobustScaler (resistente a outliers)
    scaler = RobustScaler()
    datos_escalados = scaler.fit_transform(datos_imputados)
    
    # 4. Ajustar dinámicamente n_components_pca y aplicar PCA
    componentes_reales = min(n_components_pca, n_cols)
    pca = PCA(n_components=componentes_reales)
    datos_pca = pca.fit_transform(datos_escalados)
    
    # 5. Aplicar AgglomerativeClustering sobre los componentes de PCA
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(datos_pca)
    
    # 6. Calcular el resumen por cluster usando los datos imputados (antes de escalar)
    # Añadimos temporalmente las etiquetas al DataFrame imputado para agrupar
    datos_imputados['cluster'] = labels
    
    # Agrupamos por cluster y calculamos la mediana de todas las variables
    resumen_clusters = datos_imputados.groupby('cluster').median()
    
    # Calculamos el tamaño de cada cluster ('n_eventos') usando operaciones vectorizadas
    resumen_clusters['n_eventos'] = datos_imputados.groupby('cluster').size()
    
    # 7. Formatear el DataFrame de resumen
    # Ordenar por etiqueta de cluster ascendente y reiniciar el índice desde 0
    resumen_clusters = resumen_clusters.sort_index().reset_index()
    
    # Eliminamos la columna 'cluster' del DataFrame final para que queden solo las métricas y n_eventos
    resumen_clusters = resumen_clusters.drop(columns=['cluster'])
    
    return labels, resumen_clusters