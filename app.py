# ==============================================================================
# FASE II: SERVICIO DE INFERENCIA Y TELEMETRÍA (PASOS 4 Y 5)
# Archivo: app.py
# ==============================================================================

# Importamos las librerías necesarias
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import csv
from datetime import datetime
import os

# ------------------------------------------------------------------------------
# 1. CARGA DEL MODELO EN MEMORIA
# ------------------------------------------------------------------------------
# Definimos la ruta donde empaquetamos nuestro modelo campeón en la Fase 2.
MODEL_PATH = "models_flotacion/modelo_flotacion.joblib"

# Validación de seguridad: verificamos que el modelo exista antes de levantar la API.
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}.")

# Cargamos el pipeline completo (incluye preprocesamiento + HistGradientBoosting)
modelo = joblib.load(MODEL_PATH)

# Inicializamos la aplicación FastAPI
app = FastAPI(
    title="API Predictiva - Planta de Flotación Minera",
    description="Servicio local para predecir el % de Sílice en el concentrado final.",
    version="1.0.0"
)

# ------------------------------------------------------------------------------
# 2. CONTRATO DE DATOS (ESQUEMA DE ENTRADA)
# ------------------------------------------------------------------------------
# Usamos Pydantic para definir cómo deben ser los datos que envían los sensores.
# Esto actúa como un escudo protector: si un sensor envía un texto en lugar de un 
# número para el flujo de aire, la API rechazará el dato antes de que rompa el modelo.
class DatosSensores(BaseModel):
    hour: int
    dow: str
    pct_iron_feed: float
    pct_silica_feed: float
    starch_flow: float
    amina_flow: float
    ore_pulp_flow: float
    ore_pulp_ph: float
    ore_pulp_density: float
    flotation_column_01_level: float
    flotation_column_02_level: float
    flotation_column_03_level: float
    flotation_column_01_air_flow: float
    flotation_column_02_air_flow: float
    flotation_column_03_air_flow: float

# ------------------------------------------------------------------------------
# 3. MÓDULO DE TELEMETRÍA (PASO 5)
# ------------------------------------------------------------------------------
def registrar_telemetria(datos_dict: dict, prediccion: float):
    """
    Guarda cada predicción y sus inputs en un archivo CSV local.
    Evita depender de bases de datos productivas pesadas.
    """
    archivo_csv = "telemetria.csv"
    
    # Añadimos la marca de tiempo exacta de cuándo ocurrió la predicción
    datos_dict['timestamp_inferencia'] = datetime.now().isoformat()
    # Añadimos el resultado que entregó nuestro modelo predictivo
    datos_dict['prediccion_silica_pct'] = prediccion
    
    # Comprobamos si el archivo ya existe para saber si debemos escribir los encabezados
    file_exists = os.path.isfile(archivo_csv)
    
    # Abrimos el archivo en modo "append" ("a") para agregar filas al final sin borrar nada
    with open(archivo_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=datos_dict.keys())
        
        # Si el archivo es nuevo, escribimos los nombres de las columnas en la primera fila
        if not file_exists:
            writer.writeheader()
        
        # Guardamos el registro completo
        writer.writerow(datos_dict)

# ------------------------------------------------------------------------------
# 4. ENDPOINT DE PREDICCIÓN (PASO 4)
# ------------------------------------------------------------------------------
# Este es el punto de acceso al que se enviarán los datos simulados mediante el método POST.
@app.post("/predict")
def predecir_impureza(datos: DatosSensores):
    try:
        # Paso A: Convertimos los datos validados a formato diccionario
        datos_dict = datos.dict()
        
        # Paso B: Transformamos el diccionario a un DataFrame de 1 sola fila 
        # (Scikit-Learn / joblib requieren formato tabular bidimensional).
        df_input = pd.DataFrame([datos_dict])
        
        # Paso C: Ejecutamos la predicción. Extraemos el elemento [0] porque 
        # .predict() devuelve un array y solo necesitamos el valor numérico.
        prediccion = modelo.predict(df_input)[0]
        
        # Paso D: Guardamos silenciosamente la telemetría para su futuro monitoreo.
        registrar_telemetria(datos_dict, float(prediccion))
        
        # Paso E: Respondemos al usuario (o al sistema de simulación)
        return {
            "status": "success",
            "prediccion_silica_pct": round(float(prediccion), 3)
        }
        
    except Exception as e:
        # Si algo falla (ej. error interno de Pandas), devolvemos un error 500 claro
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {str(e)}")

# Endpoint de prueba para verificar que el servidor levantó correctamente
@app.get("/")
def health_check():
    return {"mensaje": "API de la Planta Minera Activa. Todo en orden."}