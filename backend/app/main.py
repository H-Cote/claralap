from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import os
from openai import OpenAI
from typing import List, Dict, Optional

# Importación de ambas funciones del motor
from app.motor_match.motor_match import ejecutar_match_matematico, evaluar_oferta_usuario

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

class ChatRequest(BaseModel):
    mensaje: str
    historial: Optional[List[Dict[str, str]]] = []

def agente_extraer_requerimientos(historial_mensajes: list):
    system_prompt = """
    Eres un asistente experto evaluando conversaciones sobre laptops.
    Debes responder ÚNICAMENTE con un objeto JSON válido con la siguiente estructura:
    {
      "accion": "saludar" | "preguntar_datos" | "buscar_laptops" | "evaluar_oferta",
      "precio_encontrado": float (si el usuario menciona el precio de una laptop que vio),
      "presupuesto_max": float (si no se menciona en absoluto, asigna 999999),
      "necesita_gpu_dedicada": bool,
      "gama_tecnica_minima": "Baja" | "Media" | "Alta",
      "ram_minima_gb": int (4, 8, 16 o 32)
    }
    Reglas para "accion":
    - "evaluar_oferta": Si el usuario dice que ENCONTRÓ una laptop y da su precio/specs para saber si vale la pena o está cara.
    - "saludar": Si es un saludo o charla genérica.
    - "preguntar_datos": Si busca recomendación pero falta presupuesto o uso.
    - "buscar_laptops": Si quiere que le recomendemos opciones con base en su presupuesto y uso.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + historial_mensajes[-6:],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

@app.post("/api/chat")
def procesar_consulta(payload: ChatRequest):
    historial_completo = payload.historial + [{"role": "user", "content": payload.mensaje}]
    
    reqs = agente_extraer_requerimientos(historial_completo)
    if not reqs:
        raise HTTPException(status_code=500, detail="Error al interpretar la consulta.")

    accion = reqs.get("accion", "buscar_laptops")

    # CASO 1: Evaluar una oferta encontrada por el usuario
    if accion == "evaluar_oferta":
        evaluacion = evaluar_oferta_usuario(df_catalogo, reqs)
        
        sys_prompt_evaluacion = """Eres ClaraLap. Explica amigablemente si la laptop que encontró el usuario vale la pena o no.
        REGLAS:
        1. Sé breve (máximo 3 oraciones).
        2. Menciona el precio justo estimado por el modelo y la diferencia.
        3. NO uses Markdown (CERO asteriscos **).
        """
        try:
            resp_texto = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt_evaluacion},
                    {"role": "user", "content": f"Resultado evaluación: {evaluacion}"}
                ]
            )
            explicacion_eval = resp_texto.choices[0].message.content
        except:
            explicacion_eval = f"Evalué la opción: el precio justo estimado es de ${evaluacion['precio_justo_estimado']} MXN."

        return {
            "requerimientos": reqs,
            "recomendaciones": [],
            "evaluacion_oferta": evaluacion,
            "explicacion": explicacion_eval
        }

    # CASO 2: Saludo o solicitud de más datos
    if accion == "saludar" or accion == "preguntar_datos":
        prompt_charla = "Eres ClaraLap. Falta información (presupuesto o uso). Haz una pregunta CORTA y amable (máximo 2 líneas). CERO asteriscos."
        try:
            respuesta_charla = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt_charla}] + historial_completo[-3:]
            )
            explicacion = respuesta_charla.choices[0].message.content
        except:
            explicacion = "¡Hola! Cuéntame, ¿para qué usarás tu laptop o qué oferta encontraste?"

        return {"requerimientos": reqs, "recomendaciones": [], "explicacion": explicacion}

    # CASO 3: Búsqueda normal de recomendación
    top_laptops_df = ejecutar_match_matematico(df_catalogo, reqs)
    
    if top_laptops_df.empty:
        return {"recomendaciones": [], "explicacion": "No encontré equipos con esos requisitos. ¿Podríamos ajustar un poco el presupuesto?"}

    columnas = ["Marca", "Nombre", "Precio_MXN", "Precio_Justo_Modelo", "Diferencia_MXN"]
    cols_existentes = [col for col in columnas if col in top_laptops_df.columns]
    laptops = top_laptops_df[cols_existentes].fillna("").to_dict(orient="records")

    sys_prompt_empatico = """Eres ClaraLap. Dile al usuario que encontraste opciones ideales.
    REGLAS: Máximo 2 oraciones. CERO asteriscos. Sin mencionar especificaciones técnicas detalladas.
    """
    try:
        resp_texto = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_prompt_empatico},
                      {"role": "user", "content": "Genera el mensaje final."}]
        )
        explicacion = resp_texto.choices[0].message.content
    except:
        explicacion = "¡Listo! Aquí tienes las mejores opciones según lo que me comentaste."

    return {
        "requerimientos": reqs,
        "recomendaciones": laptops,
        "explicacion": explicacion
    }