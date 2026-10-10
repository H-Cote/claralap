import pandas as pd

def ejecutar_match_matematico(df: pd.DataFrame, reqs: dict):
    if df.empty: 
        return pd.DataFrame()
    
    df_filt = df.copy()

    # Filtros
    if reqs.get('presupuesto_max') and reqs['presupuesto_max'] < 999999:
        df_filt = df_filt[df_filt['Precio_MXN'] <= reqs['presupuesto_max']]
    if reqs.get('necesita_gpu_dedicada') and 'GPU_Dedicada' in df_filt.columns:
        df_filt = df_filt[df_filt['GPU_Dedicada'] == 1]
    if reqs.get('ram_minima_gb') and 'RAM_GB' in df_filt.columns:
        df_filt = df_filt[df_filt['RAM_GB'] >= reqs['ram_minima_gb']]

    # Gama Técnica
    mapa_gamas = {'Baja': ['Baja', 'Media', 'Alta'], 'Media': ['Media', 'Alta'], 'Alta': ['Alta']}
    gamas_permitidas = mapa_gamas.get(reqs.get('gama_tecnica_minima', 'Baja'), ['Baja', 'Media', 'Alta'])
    if 'Gama_Tecnica' in df_filt.columns:
        df_filt = df_filt[df_filt['Gama_Tecnica'].isin(gamas_permitidas)]

    if df_filt.empty:
        df_filt = df.sort_values(by="Precio_MXN", ascending=True)

    if 'Diferencia_MXN' in df_filt.columns:
        return df_filt.sort_values(by='Diferencia_MXN', ascending=True).head(3)
    
    return df_filt.sort_values(by='Precio_MXN', ascending=True).head(3)


def evaluar_oferta_usuario(df: pd.DataFrame, reqs: dict):
    """
    Evalúa si la laptop encontrada por el usuario está barata, justa o costosa.
    """
    if df.empty:
        return None

    precio_usuario = reqs.get("precio_encontrado", 0)
    df_similares = df.copy()

    # Filtrar por especificaciones similares para encontrar el precio justo de referencia
    if reqs.get('necesita_gpu_dedicada') and 'GPU_Dedicada' in df_similares.columns:
        df_similares = df_similares[df_similares['GPU_Dedicada'] == 1]
    if reqs.get('ram_minima_gb') and 'RAM_GB' in df_similares.columns:
        df_similares = df_similares[df_similares['RAM_GB'] == reqs['ram_minima_gb']]

    if df_similares.empty:
        df_similares = df

    # Promedio del precio justo según el modelo de regresión
    precio_justo_estimado = df_similares['Precio_Justo_Modelo'].mean() if 'Precio_Justo_Modelo' in df_similares.columns else precio_usuario

    diferencia = precio_usuario - precio_justo_estimado

    if diferencia < -1000:
        veredicto = "ganga"
        etiqueta = "🟢 ¡Excelente Oferta!"
    elif abs(diferencia) <= 1000:
        veredicto = "precio_justo"
        etiqueta = "🟡 Precio Justo de Mercado"
    else:
        veredicto = "sobreprecio"
        etiqueta = "🔴 Sobreprecio Detectado"

    return {
        "veredicto": veredicto,
        "etiqueta": etiqueta,
        "precio_usuario": precio_usuario,
        "precio_justo_estimado": round(precio_justo_estimado, 2),
        "diferencia_mxn": round(abs(diferencia), 2)
    }