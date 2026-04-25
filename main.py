from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, pandas as pd, numpy as np, os

app = FastAPI(title="AUTORA API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = os.getenv("MODEL_PATH", "modelo_autora_fasecolda.pkl")
with open(MODEL_PATH, "rb") as f:
    artefacto = pickle.load(f)

modelo = artefacto["modelo"]
encoders = artefacto["encoders"]
FEATURES = artefacto["features"]
MARCAS = sorted(artefacto.get("marcas", []))
REFS = sorted(artefacto.get("referencias", []))

class ValuacionRequest(BaseModel):
    marca: str
    referencia: str
    anio: int
    combustible: str
    traccion: str
    transmision: str
    cilindraje: float

class ValuacionResponse(BaseModel):
    precio_estimado: int
    precio_minimo: int
    precio_maximo: int
    confianza: float
    marca: str
    referencia: str
    anio: int

@app.get("/")
def root():
    return {"status": "ok", "motor": "AUTORA v1.0"}

@app.get("/marcas")
def get_marcas():
    return {"marcas": MARCAS}

@app.get("/referencias/{marca}")
def get_referencias(marca: str):
    refs = [r for r in REFS if marca.upper() in r.upper()]
    return {"marca": marca, "referencias": sorted(refs)}

def resolver_marca(nombre: str) -> str:
    """Resuelve el nombre de marca al valor exacto del encoder (case-insensitive, strip)."""
    clases = list(encoders["marca"].classes_)
    nombre_up = nombre.upper().strip()
    if nombre_up in clases:
        return nombre_up
    # Coincidencia parcial: busca la primera clase que contenga el nombre o viceversa
    candidatos = [c for c in clases if c.startswith(nombre_up) or nombre_up in c]
    if candidatos:
        return candidatos[0]
    raise HTTPException(422, detail=f"Marca '{nombre}' no disponible. Sugerencias: {clases[:3]}")


def resolver_referencia(nombre: str) -> str:
    """Resuelve el nombre de referencia al valor exacto del encoder con fuzzy match."""
    clases = list(encoders["referencia"].classes_)
    nombre_up = nombre.upper().strip()
    # 1. Coincidencia exacta
    if nombre_up in clases:
        return nombre_up
    # 2. Empieza con el nombre seguido de " " o "["
    candidatos = [c for c in clases if c.startswith(nombre_up + ' ') or c.startswith(nombre_up + '[')]
    if candidatos:
        return candidatos[0]
    # 3. Coincidencia parcial para sugerencias en el error
    sugerencias = [c for c in clases if nombre_up in c][:3]
    raise HTTPException(422, detail=f"Referencia '{nombre}' no disponible. Sugerencias: {sugerencias}")


@app.post("/valuar", response_model=ValuacionResponse)
def valuar(req: ValuacionRequest):
    try:
        marca_resuelta = resolver_marca(req.marca)
        referencia_resuelta = resolver_referencia(req.referencia)
        fila = {
            "marca_enc": encoders["marca"].transform([marca_resuelta])[0],
            "referencia_enc": encoders["referencia"].transform([referencia_resuelta])[0],
            "anio": req.anio, "antiguedad": 2025 - req.anio, "cilindraje": req.cilindraje,
            "combustible_enc": encoders["combustible"].transform([req.combustible.upper()])[0],
            "traccion_enc": encoders["traccion"].transform([req.traccion.upper()])[0],
            "transmision_enc": encoders["transmision"].transform([req.transmision.upper()])[0],
        }
        precio = int(modelo.predict(pd.DataFrame([fila]))[0])
        return ValuacionResponse(precio_estimado=precio, precio_minimo=int(precio*0.90),
            precio_maximo=int(precio*1.10), confianza=0.92,
            marca=marca_resuelta, referencia=referencia_resuelta, anio=req.anio)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
