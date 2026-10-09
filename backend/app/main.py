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

# Cargar el catálogo
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "catalogo_enriquecido.csv")
df_catalogo = pd.read_csv(CSV_PATH)

# Reconstruir la columna 'Marca' a partir de las columnas One-Hot Encoding o del 'Nombre'
def extraer_marca(row):
    for col in row.index:
        if str(col).startswith("Marca_") and row[col] == 1:
            return col.replace("Marca_", "")
    nombre = str(row.get("Nombre", ""))
    return nombre.split()[0] if nombre else "Laptop"

if "Marca" not in df_catalogo.columns:
    df_catalogo["Marca"] = df_catalogo.apply(extraer_marca, axis=1)

class ChatRequest(BaseModel):
    mensaje: str

@app.post("/api/chat")
def procesar_consulta(payload: ChatRequest):
    # 1. OpenAI extrae requerimientos
    sys_prompt = """Traduce la necesidad del usuario a un JSON estricto:
    {"presupuesto_max": int, "ram_minima_gb": int, "gama_tecnica_minima": "Baja"|"Media"|"Alta"}"""

    resp_json = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": payload.mensaje}],
        response_format={"type": "json_object"}
    )
    reqs = json.loads(resp_json.choices[0].message.content)

    # 2. Filtrado por presupuesto y RAM
    presupuesto = reqs.get("presupuesto_max", 999999)
    ram = reqs.get("ram_minima_gb", 4)

    df_filt = df_catalogo[
        (df_catalogo["Precio_MXN"] <= presupuesto) &
        (df_catalogo["RAM_GB"] >= ram)
    ]

    # Si no se encuentra nada con ese filtro, traer las 3 más económicas
    if df_filt.empty:
        df_filt = df_catalogo.sort_values(by="Precio_MXN", ascending=True).head(3)

    # Ordenar por gangas (Diferencia_MXN más negativa representa mayor ahorro)
    top_3 = df_filt.sort_values(by="Diferencia_MXN", ascending=True).head(3)

    laptops = top_3[["Marca", "Nombre", "Precio_MXN", "Precio_Justo_Modelo", "Diferencia_MXN"]].to_dict(orient="records")

    # 3. OpenAI redacta la explicación
    sys_prompt_empatico = "Explica amigablemente por qué estas laptops son ideales para el usuario y destaca el ahorro detectado."
    resp_texto = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys_prompt_empatico},
                  {"role": "user", "content": f"Requisitos: {reqs}\nLaptops recomendadas: {laptops}"}]
    )

    return {
        "requerimientos": reqs,
        "recomendaciones": laptops,
        "explicacion": resp_texto.choices[0].message.content
    }
