from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import os
from openai import OpenAI

# Importación directa desde la carpeta app
from app.motor_match.motor_match import ejecutar_match_matematico

app = FastAPI(title="ClaraLap API")

# Habilitar CORS para permitir peticiones desde Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cargar el catálogo (Se asume que está en claralap/backend/catalogo_enriquecido.csv)
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "catalogo_enriquecido.csv")
try:
    df_catalogo = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {CSV_PATH}")
    df_catalogo = pd.DataFrame()

# Reconstruir la columna 'Marca'
def extraer_marca(row):
    for col in row.index:
        if str(col).startswith("Marca_") and row[col] == 1:
            return col.replace("Marca_", "")
    nombre = str(row.get("Nombre", ""))
    return nombre.split()[0] if nombre else "Laptop"

if not df_catalogo.empty and "Marca" not in df_catalogo.columns:
    df_catalogo["Marca"] = df_catalogo.apply(extraer_marca, axis=1)

class ChatRequest(BaseModel):
    mensaje: str

def agente_extraer_requerimientos(prompt_usuario: str):
    system_prompt = """
    Eres un asistente experto en traducir necesidades de usuarios a requerimientos de hardware.
    Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
    {
      "buscar_laptops": bool (true solo si el usuario menciona presupuesto, uso, tareas, o características. false si es solo un saludo o charla),
      "presupuesto_max": float (si no se especifica, asigna 999999),
      "necesita_gpu_dedicada": bool,
      "gama_tecnica_minima": "Baja" | "Media" | "Alta",
      "ram_minima_gb": int (4, 8, 16 o 32)
    }
    Reglas:
    - AutoCAD, 3D, Gaming, Premiere o IA requieren "necesita_gpu_dedicada": true, "gama_tecnica_minima": "Alta", y "ram_minima_gb": 16.
    - Oficina, Programación, Universidad requieren "gama_tecnica_minima": "Media" y "ram_minima_gb": 8.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt_usuario}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error extrayendo requerimientos: {e}")
        return None

# --- ENDPOINT PRINCIPAL ---

@app.post("/api/chat")
def procesar_consulta(payload: ChatRequest):
    # 1. Extracción de requerimientos
    reqs = agente_extraer_requerimientos(payload.mensaje)
    if not reqs:
        raise HTTPException(status_code=500, detail="Error al interpretar la consulta.")

    # NUEVA LÓGICA: Si es un saludo o charla sin intención de búsqueda
    # Usamos get("buscar_laptops", True) por si el LLM olvida la variable, asumimos que busca.
    if not reqs.get("buscar_laptops", True):
        try:
            respuesta_charla = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Eres ClaraLap, una asesora experta en laptops. Responde de forma muy amigable, natural y servicial a este saludo o comentario. Pregunta cortésmente qué uso le darán a la laptop, qué programas usan o qué presupuesto tienen para poder recomendarles las mejores opciones."},
                    {"role": "user", "content": payload.mensaje}
                ],
                temperature=0.7
            )
            explicacion_charla = respuesta_charla.choices[0].message.content
        except:
            explicacion_charla = "¡Hola! Soy Clara. Cuéntame, ¿para qué usarás tu laptop o qué presupuesto tienes en mente?"

        return {
            "requerimientos": reqs,
            "recomendaciones": [],
            "explicacion": explicacion_charla
        }

    # 2. Match Matemático (usando la función importada)
    top_laptops_df = ejecutar_match_matematico(df_catalogo, reqs)
    
    if top_laptops_df.empty:
        return {"recomendaciones": [], "explicacion": "Lo siento, no encontré equipos en el catálogo actual que se ajusten a esos requisitos tan específicos. ¿Podrías ser un poco más flexible con el precio o las características?"}

    # Limpiar Dataframe y seleccionar columnas útiles
    columnas = ["Marca", "Nombre", "Precio_MXN", "Precio_Justo_Modelo", "Diferencia_MXN"]
    cols_existentes = [col for col in columnas if col in top_laptops_df.columns]
    
    top_laptops_df = top_laptops_df[cols_existentes].fillna("")
    laptops = top_laptops_df.to_dict(orient="records")

    # 3. Traducción Empática
    sys_prompt_empatico = "Eres ClaraLap. Explica amigablemente por qué estas laptops son ideales para el usuario y destaca el ahorro detectado. Sé directa y conversacional."
    try:
        resp_texto = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_prompt_empatico},
                      {"role": "user", "content": f"Requisitos: {reqs}\nLaptops recomendadas: {laptops}"}]
        )
        explicacion = resp_texto.choices[0].message.content
    except:
        explicacion = "¡Perfecto! Con esa información, encontré estas excelentes opciones que se ajustan muy bien a tus necesidades."

    return {
        "requerimientos": reqs,
        "recomendaciones": laptops,
        "explicacion": explicacion
    }