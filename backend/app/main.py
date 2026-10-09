from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import json
import os
from openai import OpenAI

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

# Cargar el catálogo al iniciar el servidor
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "catalogo_enriquecido.csv")
df_catalogo = pd.read_csv(CSV_PATH)

class ChatRequest(BaseModel):
    mensaje: str

@app.post("/api/chat")
def procesar_consulta(payload: ChatRequest):
    # 1. OpenAI extrae requerimientos
    sys_prompt = """Traduce la necesidad a JSON: {"presupuesto_max": int, "ram_minima_gb": int, "gama_tecnica_minima": "Baja"|"Media"|"Alta"}"""
    
    resp_json = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": payload.mensaje}],
        response_format={"type": "json_object"}
    )
    reqs = json.loads(resp_json.choices[0].message.content)

    # 2. Match Matemático usando tu columna de Diferencia_MXN
    df_filt = df_catalogo[df_catalogo["Precio_MXN"] <= reqs.get("presupuesto_max", 999999)]
    df_filt = df_filt[df_filt["RAM_GB"] >= reqs.get("ram_minima_gb", 4)]
    
    if df_filt.empty:
        return {"error": "No hay laptops para ese presupuesto."}

    # Ordenar por tu detector de gangas (Diferencia_MXN más negativa)
    top_3 = df_filt.sort_values(by="Diferencia_MXN", ascending=True).head(3)
    laptops = top_3[["Marca", "Nombre", "Precio_MXN", "Precio_Justo_Modelo", "Diferencia_MXN"]].to_dict(orient="records")

    # 3. OpenAI redacta la explicación
    sys_prompt_empatico = "Explica amigablemente por qué estas laptops son ideales y menciona el ahorro detectado."
    resp_texto = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys_prompt_empatico}, 
                  {"role": "user", "content": f"Requisitos: {reqs}\nLaptops: {laptops}"}]
    )

    return {
        "requerimientos": reqs,
        "recomendaciones": laptops,
        "explicacion": resp_texto.choices[0].message.content  # Mismo nombre que espera tu HTML
    }