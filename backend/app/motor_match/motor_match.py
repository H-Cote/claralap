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

    # Fallback si el filtro es muy estricto
    if df_filt.empty:
        df_filt = df.sort_values(by="Precio_MXN", ascending=True)

    # Ordenar por gangas
    if 'Diferencia_MXN' in df_filt.columns:
        return df_filt.sort_values(by='Diferencia_MXN', ascending=True).head(3)
    
    return df_filt.sort_values(by='Precio_MXN', ascending=True).head(3)