from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import os
from openai import OpenAI
from typing import List, Dict, Optional

# Importación directa desde la carpeta app
from app.motor_match.motor_match import ejecutar_match_matematico

app = FastAPI(title="ClaraLap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "catalogo_enriquecido.csv")
try:
    df_catalogo = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    df_catalogo = pd.DataFrame()

def extraer_marca(row):
    for col in row.index:
        if str(col).startswith("Marca_") and row[col] == 1:
            return col.replace("Marca_", "")
    nombre = str(row.get("Nombre", ""))
    return nombre.split()[0] if nombre else "Laptop"

if not df_catalogo.empty and "Marca" not in df_catalogo.columns:
    df_catalogo["Marca"] = df_catalogo.apply(extraer_marca, axis=1)

# AHORA ACEPTAMOS EL HISTORIAL DE LA CONVERSACIÓN
class ChatRequest(BaseModel):
    mensaje: str
    historial: Optional[List[Dict[str, str]]] = []

def agente_extraer_requerimientos(historial_mensajes: list):
    system_prompt = """
    Eres un asistente experto evaluando conversaciones para comprar una laptop.
    Analiza el historial de la conversación.
    Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
    {
      "accion": "saludar" | "preguntar_datos" | "buscar_laptops",
      "presupuesto_max": float (si no se menciona en absoluto, asigna 999999),
      "necesita_gpu_dedicada": bool,
      "gama_tecnica_minima": "Baja" | "Media" | "Alta",
      "ram_minima_gb": int (4, 8, 16 o 32)
    }
    Reglas para "accion":
    - "saludar": Si el usuario solo dice hola o charla sin intención clara.
    - "preguntar_datos": Si el usuario menciona el uso (ej. "jugar") pero NO el presupuesto, o viceversa.
    - "buscar_laptops": Si en la conversación ya se entiende el uso principal Y un presupuesto aproximado.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            # Enviamos el sistema + los últimos 6 mensajes de contexto
            messages=[{"role": "system", "content": system_prompt}] + historial_mensajes[-6:],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

@app.post("/api/chat")
def procesar_consulta(payload: ChatRequest):
    # Armamos el historial completo para que el LLM tenga contexto
    historial_completo = payload.historial + [{"role": "user", "content": payload.mensaje}]
    
    reqs = agente_extraer_requerimientos(historial_completo)
    if not reqs:
        raise HTTPException(status_code=500, detail="Error al interpretar la consulta.")

    accion = reqs.get("accion", "buscar_laptops")

    # Si falta información, Clara pregunta antes de buscar
    if accion == "saludar" or accion == "preguntar_datos":
        prompt_charla = "Eres ClaraLap. Falta información (presupuesto o uso). Haz una pregunta CORTA y amable (máximo 2 líneas) para obtener el dato que falta. REGLA ESTRICTA: NO uses formato markdown (CERO asteriscos)."
        try:
            respuesta_charla = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt_charla}] + historial_completo[-3:]
            )
            explicacion = respuesta_charla.choices[0].message.content
        except:
            explicacion = "¡Hola! Cuéntame, ¿para qué usarás tu laptop y qué presupuesto tienes en mente?"

        return {"requerimientos": reqs, "recomendaciones": [], "explicacion": explicacion}

    # Si ya tenemos los datos, buscamos:
    top_laptops_df = ejecutar_match_matematico(df_catalogo, reqs)
    
    if top_laptops_df.empty:
        return {"recomendaciones": [], "explicacion": "No encontré equipos con esos requisitos. ¿Podríamos ajustar un poco el presupuesto?"}

    columnas = ["Marca", "Nombre", "Precio_MXN", "Precio_Justo_Modelo", "Diferencia_MXN"]
    cols_existentes = [col for col in columnas if col in top_laptops_df.columns]
    laptops = top_laptops_df[cols_existentes].fillna("").to_dict(orient="records")

    # REGLAS ESTRICTAS PARA QUE NO ESCRIBA TEXTOS LARGOS NI ASTERISCOS
    sys_prompt_empatico = """Eres ClaraLap. Dile al usuario que encontraste opciones ideales.
    REGLAS ESTRICTAS E INQUEBRANTABLES:
    1. Sé EXTREMADAMENTE BREVE (máximo 2 oraciones).
    2. NO menciones nombres de laptops, procesadores ni RAM (el usuario ya lo está viendo en las tarjetas).
    3. NO uses Markdown bajo ninguna circunstancia (CERO asteriscos **, CERO negritas).
    4. Habla en texto plano y resalta de forma general que encontraste buen ahorro.
    """
    try:
        resp_texto = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_prompt_empatico},
                      {"role": "user", "content": "Genera el mensaje final de entrega de recomendaciones."}]
        )
        explicacion = resp_texto.choices[0].message.content
    except:
        explicacion = "¡Listo! Aquí tienes las mejores opciones según lo que me comentaste."

    return {
        "requerimientos": reqs,
        "recomendaciones": laptops,
        "explicacion": explicacion
    }