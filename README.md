# Sistema MLOps — Monitoreo de Data Drift en Planta de Flotación Minera

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-API-009688) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E)

> **Dataset:** [Quality Prediction in a Mining Process (Kaggle)](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process). No se incluye en el repo por tamaño (~184 MB): descárgalo y guárdalo como `data_flotacion/dataset_flotacion.csv`.

Predice el `% de Sílice` (impureza)
en el concentrado final de una planta de flotación minera a partir de sensores de
proceso (flujos de reactivos, niveles de aire, pH, densidad), y monitorea en
producción si la distribución de esos sensores — y el desempeño real del modelo —
se degrada respecto a los datos de entrenamiento (**Data Drift**), usando **PSI,
Test de Kolmogorov-Smirnov y Chi²**. Cuando el drift es severo, dispara un
**plan de acción semi-automático** de reentrenamiento (Fase 5).

Todo el stack corre **local, sin Docker ni servicios cloud**: FastAPI + Uvicorn
(API de inferencia), Streamlit (dashboard), y archivos CSV locales como telemetría.

## Arquitectura

```
 dataset_flotacion.csv (histórico, 6 meses cada 20s)
            │
            │  (simulacion.py: limpia, resamplea a 1h, reproduce
            │   cronológicamente el flujo de sensores)
            ▼
   POST /predict  ──────────────►  app.py (FastAPI + modelo .joblib)
                                        │
                                        ├─► telemetria.csv        (cada predicción)
                                        │
   PSI + KS + Chi² (ventana vs.        │
   referencia) + MAE real ◄────────────┘
            │
            ▼
   drift_log.csv + performance_log.csv ──► dashboard.py (Streamlit)
            │                                      │
            ▼                                      │
   plan_de_accion.py (Fase 5) ◄────────────────────┘  (botón "Ejecutar plan de acción")
      │
      └─► si drift severo: reentrena con datos recientes →
          models_flotacion/modelo_flotacion_candidato.joblib (no reemplaza el de producción)
```

## Estructura del proyecto

```
proyecto_mlops_flotacion/
├── notebooks/
│   ├── 01_eda_flotation_plant.ipynb                                       # Fase 1: EDA, fuga de datos, evidencia preliminar de drift
│   └── 02_predictive_modeling.ipynb                                    # Fase 2: pipeline, split temporal, baselines, empaquetado
├── app.py                          # API FastAPI: carga el modelo, expone /predict, registra telemetría
├── simulacion.py                   # Fase 4: reproduce el histórico cronológicamente, calcula PSI/KS/Chi²/MAE
├── plan_de_accion.py               # Fase 5: gatillo de alerta / reentrenamiento semi-automático ante drift
├── dashboard.py                    # Dashboard Streamlit (lee telemetria.csv, drift_log.csv, performance_log.csv)
├── requirements.txt
├── models_flotacion/
│   ├── modelo_flotacion.joblib             # Pipeline en producción (preprocesamiento + HistGradientBoosting)
│   ├── train_metrics.json                  # Métricas y features del entrenamiento (Fase 2)
│   ├── modelo_flotacion_candidato.joblib   # Generado por plan_de_accion.py si hay drift severo (no viene en el repo)
│   └── train_metrics_candidato.json        # Métricas del modelo candidato (no viene en el repo)
├── data_flotacion/
│   ├── dataset_flotacion.csv       # Dataset crudo (Kaggle: Quality Prediction in a Mining Process)
│   └── serie_horaria.csv           # Cache de la limpieza/resampling (se genera solo, no viene en el repo)
├── telemetria.csv                  # Se genera en tiempo de ejecución (no viene en el repo)
├── drift_log.csv                   # Se genera en tiempo de ejecución (no viene en el repo)
├── performance_log.csv             # Se genera en tiempo de ejecución (no viene en el repo)
└── ALERTA_DRIFT.log                # Se genera en tiempo de ejecución (no viene en el repo)
```

## Requisitos previos

- Python 3.11 (recomendado; el proyecto se validó con Python 3.11 vía Anaconda).
- El archivo `data_flotacion/dataset_flotacion.csv` debe existir (dataset crudo de
  Kaggle *"Quality Prediction in a Mining Process"*). Si no lo tienes, descárgalo y
  cópialo a esa ruta con ese nombre exacto antes de correr `simulacion.py`.

> ⚠️ **Nota sobre versiones:** el modelo `.joblib` fue serializado con
> `scikit-learn==1.6.1`. Versiones más nuevas (ej. 1.9.x) cambiaron una clase
> interna de `ColumnTransformer` y **no pueden deserializar el archivo**
> (`AttributeError: Can't get attribute '_RemainderColsList'`). Por eso
> `requirements.txt` fija esa versión exacta — no la actualices sin volver a
> entrenar y re-guardar el modelo.

## Instalación y ejecución (≈ 5-8 minutos)

### Paso 0 — Crear el entorno (≈ 2 min)

Con conda (recomendado):

```bash
conda create -n mlops_flotacion python=3.11 -y
conda activate mlops_flotacion
pip install -r requirements.txt
```

O con `venv` si no usas conda:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

En VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → elige el entorno
recién creado (`mlops_flotacion` o `.venv`), para que Pylance deje de marcar
`fastapi`/`pandas`/`joblib` como no encontrados.

### Paso 1 — Levantar la API de inferencia (≈ 30 seg)

En una terminal (con el entorno activado):

```bash
uvicorn app:app --reload
```

Verifica que responde: abre `http://127.0.0.1:8000/` — deberías ver
`{"mensaje": "API de la Planta Minera Activa. Todo en orden."}`.
La documentación interactiva está en `http://127.0.0.1:8000/docs`.

### Paso 2 — Simular producción y detectar drift (≈ 2-4 min)

En **otra terminal** (mismo entorno activado, deja la API corriendo):

```bash
python simulacion.py
```

Esto:
1. Procesa el CSV crudo (~737.000 filas cada 20s) y lo agrupa a nivel horario
   — la primera vez tarda unos segundos y deja un cache en
   `data_flotacion/serie_horaria.csv` (las corridas siguientes son instantáneas).
2. Toma como **referencia** la misma ventana con la que se entrenó el modelo
   (marzo, días 1-24).
3. Envía en orden cronológico, hora por hora, todo lo que viene después
   (fin de marzo → septiembre) al endpoint `/predict`.
4. Cada **ventana de producción** (por defecto 1 semana = 168 horas), calcula:
   - **PSI** (Population Stability Index) para las 13 variables numéricas y para
     `dow` (categórica) — es el criterio que decide el nivel de alerta.
   - **KS-test** (numéricas) y **Chi²** (`dow`) como evidencia estadística
     complementaria (p-value), guardados junto al PSI en `drift_log.csv`.
   - **MAE/RMSE real** de esa ventana (drift del *target*/performance, no solo
     de los inputs) en `performance_log.csv` — sabemos el valor real porque es
     dato histórico.

**¿Por qué ventana de 1 semana y no de 1 día?** `dow` es una de las variables
monitoreadas; con una ventana de 24h sería una categoría constante (sin
variación posible para comparar). Con 168h hay hasta 7 días distintos.

**Umbrales de alerta (justificación):** se usa **PSI** como criterio principal
porque, a diferencia de un p-value, no depende del tamaño de muestra (con
ventanas grandes un p-value cae a ~0 aunque el cambio práctico sea mínimo). Se
usa la convención estándar de industria (scoring de riesgo crediticio):

| PSI | Interpretación | Acción |
|---|---|---|
| < 0.10 | Sin drift relevante | Ninguna |
| 0.10 – 0.25 | Drift moderado | Alertar, revisión humana |
| ≥ 0.25 | Drift severo | Alertar + gatillar reentrenamiento (Fase 5) |

KS y Chi² usan `alpha = 0.05` (95% de confianza), el nivel de significancia
convencional en estadística — se muestran como evidencia adicional, no
reemplazan al PSI en la decisión de alerta.

Parámetros útiles (todos opcionales):

```bash
python simulacion.py --max-horas 200 --delay 0 --ventana-horas 168 --alpha 0.05
```

- `--max-horas`: limita cuántas horas simular (útil para una demo rápida).
- `--delay`: segundos de espera entre cada request (0 = lo más rápido posible).
- `--ventana-horas`: tamaño de la ventana de producción para PSI/KS/Chi².
- `--alpha`: nivel de significancia para KS/Chi² (solo informativo).

> **Nota honesta:** en las corridas de prueba, el PSI de `dow` suele quedar en
> nivel "moderado" de forma casi constante — es un efecto de tamaño de muestra
> (una ventana de 168h no reparte los 7 días exactamente igual que la
> referencia), no necesariamente un cambio de comportamiento real de la planta.
> Las variables de proceso (`pct_iron_feed`, `flotation_column_*`, etc.) sí
> muestran PSI severo de forma sostenida y consistente con lo visto en el EDA.

### Paso 3 — Levantar el dashboard (≈ 30 seg)

En una **tercera terminal**:

```bash
streamlit run dashboard.py
```

Se abre en `http://localhost:8501`. Muestra:
- KPIs: predicciones registradas, última predicción, MAE real de la última
  ventana (vs. la primera), y el nivel de alerta actual.
- Una caja de **plan de acción** arriba de todo (verde/amarillo/rojo según el
  nivel), con un botón para ejecutar `plan_de_accion.py` directamente si el
  drift es severo.
- Evolución temporal de la predicción de % Sílice.
- **Performance real (MAE) por ventana** — la señal de drift del *target*, no
  solo de los inputs.
- Línea de tiempo del nivel de alerta (PSI) ventana por ventana.
- Detalle de PSI + KS/Chi² por variable, con selector, líneas de umbral (0.10 /
  0.25) y tabla de la última ventana evaluada.
- Ranking histórico de qué sensores presentan drift con más frecuencia.

Tiene un botón **"Actualizar datos"** para releer los CSV más recientes mientras
`simulacion.py` sigue corriendo en la otra terminal.

### Paso 4 — Plan de acción ante drift (Fase 5)

Cuando `simulacion.py` deja la última ventana en nivel **severo**, corre:

```bash
python plan_de_accion.py
```

(o usa el botón equivalente en el dashboard). El script:
1. Lee `drift_log.csv`, determina el nivel de alerta de la última ventana.
2. Si es **moderado**: registra la alerta en `ALERTA_DRIFT.log` y recomienda
   revisión humana — no reentrena solo (podría ser ruido normal del proceso).
3. Si es **severo**: además de la alerta, **reentrena automáticamente** el
   mismo pipeline (misma arquitectura: `ColumnTransformer` + `HistGradientBoostingRegressor`)
   usando los **últimos 30 días de producción simulada** (no marzo), con un
   nuevo split temporal 80/20 dentro de esa ventana reciente. Guarda el
   resultado como `models_flotacion/modelo_flotacion_candidato.joblib` +
   `train_metrics_candidato.json`, **sin reemplazar** el modelo en producción.

Es **semi-automático a propósito**: promover un modelo sin revisión humana es
riesgoso (podría entrenarse sobre una ventana corta y ruidosa, o con un
problema de sensor puntual). El gatillo está implementado de punta a punta; la
decisión de promover el candidato a producción queda documentada pero manual
(comparar `train_metrics.json` vs `train_metrics_candidato.json` y, si
corresponde, renombrar el candidato como `modelo_flotacion.joblib`).

## Decisiones clave del modelo (resumen de Fases 1-2)

- **Fuga de datos:** se eliminó `pct_iron_concentrate` porque se mide en el mismo
  análisis de laboratorio que el target — no existe en tiempo real.
- **Resampling:** los sensores (cada 20s) se agrupan por promedio a 1 hora, para
  alinear con la frecuencia de las pruebas de laboratorio.
- **Split temporal:** el modelo se entrenó solo con marzo 2017 (día 1-24 train,
  25-31 test), sin mezclar aleatoriamente — así se simula honestamente cómo
  se comporta el modelo prediciendo el futuro a partir del pasado.
- **Modelo campeón:** `HistGradientBoostingRegressor` sobre un
  `ColumnTransformer` (StandardScaler + OneHotEncoder). Superó a la regresión
  lineal, pero su `test_mae` (1.31%) quedó por sobre el baseline global (0.97%)
  — evidencia de que con solo ~150 horas de entrenamiento el modelo es frágil,
  lo cual hace *aún más relevante* el monitoreo de drift de este proyecto.

## Troubleshooting

| Síntoma | Causa | Solución |
|---|---|---|
| Pylance marca `fastapi`/`pandas`/`joblib` como no encontrados | VS Code apunta a un intérprete distinto al que tiene las librerías instaladas | `Ctrl+Shift+P` → *Python: Select Interpreter* → elegir el entorno correcto |
| `AttributeError: Can't get attribute '_RemainderColsList'` al cargar el `.joblib` | Versión de scikit-learn distinta a la usada para entrenar | `pip install scikit-learn==1.6.1` en el entorno activo |
| `RuntimeError: No se encontró el modelo en models_flotacion/...` | El `.joblib`/`.json` no están dentro de `models_flotacion/` | Verifica que ambos archivos estén en esa carpeta, no en la raíz |
| `UnicodeEncodeError: 'charmap' codec can't encode...` al correr `simulacion.py` | La consola de Windows usa `cp1252` por defecto y no imprime emojis | Ya viene resuelto en el script (`sys.stdout.reconfigure(encoding="utf-8")`); si aparece en otro script, aplica el mismo fix |
| `simulacion.py` no logra conectarse a la API | `uvicorn` no está corriendo, o corre en otro puerto | Levanta `uvicorn app:app --reload` primero, o usa `--api-url` para apuntar al puerto correcto |
| `dashboard.py` muestra "Todavía no hay datos" | No se ha corrido `simulacion.py` (o ninguna petición manual a `/predict`) todavía | Corre `python simulacion.py` con la API arriba |
| `KeyError: 'dow'` (o 'hour') al analizar una ventana en `simulacion.py` | El cache `data_flotacion/serie_horaria.csv` quedó de una versión anterior del script, sin esas columnas | Borra `data_flotacion/serie_horaria.csv` y vuelve a correr `python simulacion.py` (se regenera solo) |
| `plan_de_accion.py` dice "No hay suficientes datos recientes para reentrenar" | La simulación se cortó con `--max-horas` antes de acumular 30 días de producción | Corre `simulacion.py` sin `--max-horas` (o con un valor mayor a ~720) antes de pedir el reentrenamiento |
