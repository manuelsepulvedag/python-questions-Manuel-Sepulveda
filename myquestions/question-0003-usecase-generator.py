import numpy as np
import pandas as pd
import random

def generar_caso_de_uso_ema_adaptativa():
    """
    Genera un caso de uso aleatorio para un filtro de Media Móvil Exponencial Adaptativa.
    Ajusta el factor de suavizado alpha dinámicamente según la volatilidad.
    """
    
    # --- 1. COMPONENTE ALEATORIO (Generación de Datos de Sensores) ---
    n_puntos = random.randint(200, 400)
    t = np.arange(n_puntos)
    
    # Señal base (cambios de nivel escalonados + tendencia)
    senal_limpia = np.zeros(n_puntos)
    for i in range(0, n_puntos, 50):
        senal_limpia[i:i+50] = random.uniform(10, 50)
    
    # Añadimos ruido blanco y algunos "picos" (outliers) aleatorios
    ruido = np.random.normal(0, 2, n_puntos)
    picos = np.zeros(n_puntos)
    if random.random() > 0.5:
        idx_picos = np.random.choice(n_puntos, 5)
        picos[idx_picos] = random.choice([-20, 20])
        
    df_input = pd.DataFrame({
        "sensor_val": senal_limpia + ruido + picos
    })
    
    col_name = "sensor_val"
    ventana_vol = random.choice([10, 15, 20, 25])
    
    input_dict = {
        "df": df_input,
        "valor_col": col_name,
        "ventana_volatilidad": ventana_vol
    }

    # --- 2. SOLUCIÓN (Lógica interna para generar el Output) ---
    def preparar_datos_ema(df, valor_col, ventana):
        df_res = df.copy()
        serie = df_res[valor_col].values
        
        # Calcular volatilidad local (desviación estándar móvil) con Pandas
        volatilidad = df_res[valor_col].rolling(window=ventana, min_periods=1).std().fillna(0).values
        
        # Normalizar volatilidad para obtener alpha dinámico [0.1, 0.9]
        # Si la volatilidad es alta, alpha es alto (sigue a la señal)
        # Si la volatilidad es baja, alpha es bajo (suaviza el ruido)
        max_vol = np.max(volatilidad) if np.max(volatilidad) > 0 else 1
        alphas = 0.1 + 0.8 * (volatilidad / max_vol)
        
        # Aplicación del filtro EMA Adaptativo con NumPy
        ema_filtrada = np.zeros(len(serie))
        ema_filtrada[0] = serie[0] # Inicialización
        
        for i in range(1, len(serie)):
            a = alphas[i]
            ema_filtrada[i] = a * serie[i] + (1 - a) * ema_filtrada[i-1]
            
        df_res["senal_limpia"] = ema_filtrada
        df_res["alpha_dinamico"] = alphas
        
        return df_res

    # --- 3. GENERACIÓN DEL OUTPUT ---
    output_df = preparar_datos_ema(
        input_dict["df"], 
        input_dict["valor_col"], 
        input_dict["ventana_volatilidad"]
    )
    
    return input_dict, output_df

# --- Ejemplo de ejecución ---
# inp, out = generar_caso_de_uso_ema_adaptativa()
# print(f"Ventana de volatilidad: {inp['ventana_volatilidad']}")
# print("Primeras filas con filtro aplicado:\n", out[['sensor_val', 'senal_limpia', 'alpha_dinamico']].head(10))