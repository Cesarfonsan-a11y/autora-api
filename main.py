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

@app.post("/valuar", response_model=ValuacionResponse)
def valuar(req: ValuacionRequest):
    try:
        fila = {
            "marca_enc": encoders["marca"].transform([req.marca.upper()])[0],
            "referencia_enc": encoders["referencia"].transform([req.referencia.upper()])[0],
            "anio": req.anio, "antiguedad": 2025 - req.anio, "cilindraje": req.cilindraje,
            "combustible_enc": encoders["combustible"].transform([req.combustible.upper()])[0],
            "traccion_enc": encoders["traccion"].transform([req.traccion.upper()])[0],
            "transmision_enc": encoders["transmision"].transform([req.transmision.upper()])[0],
        }
        precio = int(modelo.predict(pd.DataFrame([fila]))[0])
        return ValuacionResponse(precio_estimado=precio, precio_minimo=int(precio*0.90),
            precio_maximo=int(precio*1.10), confianza=0.92,
            marca=req.marca, referencia=req.referencia, anio=req.anio)
    except Exception as e:
        raise HTTPException(500, str(e))
