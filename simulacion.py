# ==============================================================================
# FASE 4: SIMULACIÓN DE PRODUCCIÓN + DETECCIÓN DE DATA DRIFT
# Archivo: simulacion.py
# ==============================================================================
#
# Qué hace este script:
# 1. Carga el dataset histórico crudo (6 meses, marzo-septiembre 2017, cada 20s)
#    y lo transforma en una serie horaria, EXACTAMENTE con las mismas reglas de
#    limpieza/resampling que se usaron en la Fase 1 (EDA) y Fase 2 (Modelado).
# 2. Define la "ventana de referencia" = los mismos datos con los que se entrenó
#    el modelo (marzo, días 1-24). Esa es la distribución que el modelo "conoce".
# 3. Recorre en orden cronológico TODO lo que viene después (día 25 de marzo en
#    adelante) simulando el flujo real de sensores hacia la API en producción,
#    enviando cada hora como una petición POST a /predict.
# 4. Cada ventana de producción (por defecto 1 semana = 168 horas), calcula tres
#    familias de métricas de drift:
#      - PSI (Population Stability Index) para TODAS las variables (numéricas
#        y categóricas) — es el criterio de alerta principal.
#      - KS-test (numéricas) y Chi² (categóricas, "dow") como evidencia
#        estadística complementaria.
#      - MAE/RMSE real de esa ventana (tenemos el valor real porque es
#        histórico) — el drift del TARGET/performance, no solo de los inputs.
# 5. Registra todo en 'drift_log.csv' (drift por variable) y
#    'performance_log.csv' (performance real por ventana), para que
#    dashboard.py y plan_de_accion.py los lean.
#
# ¿Por qué ventana de 1 semana y no de 1 día? "dow" (día de la semana) es una
# de las variables monitoreadas. Con una ventana de 24 horas, "dow" sería una
# sola categoría constante (no hay comparación posible). Con 168 horas hay
# hasta 7 días distintos, lo que permite un PSI/Chi² de "dow" con sentido.
# ------------------------------------------------------------------------------

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2_contingency, ks_2samp

# La consola de Windows suele usar cp1252 por defecto, que no sabe imprimir
# emojis (✅, 🚨, ❌). Forzamos UTF-8 para que los prints no hagan crashear el script.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ------------------------------------------------------------------------------
# 1. CONSTANTES Y RUTAS (deben calzar con lo definido en la Fase 2 - Modelado)
# ------------------------------------------------------------------------------
DATA_DIR = pathlib.Path("data_flotacion")
RAW_CSV = DATA_DIR / "dataset_flotacion.csv"          # CSV crudo (cada 20s)
SERIE_HORARIA_CACHE = DATA_DIR / "serie_horaria.csv"  # Cache ya limpio (1 fila = 1 hora)
DRIFT_LOG = pathlib.Path("drift_log.csv")
PERFORMANCE_LOG = pathlib.Path("performance_log.csv")

# Mismas features que usó el modelo (ver models_flotacion/train_metrics.json)
NUM_FEATURES = [
    "hour",
    "pct_iron_feed", "pct_silica_feed",
    "starch_flow", "amina_flow",
    "ore_pulp_flow", "ore_pulp_ph", "ore_pulp_density",
    "flotation_column_01_level", "flotation_column_02_level", "flotation_column_03_level",
    "flotation_column_01_air_flow", "flotation_column_02_air_flow", "flotation_column_03_air_flow",
]
CAT_FEATURES = ["dow"]

# "hour" es determinística (0-23 siempre) y no aporta como señal de drift real
# de proceso, así que se excluye del análisis (pero sí se envía a la API,
# porque el modelo la necesita como feature).
FEATURES_NUMERICAS_DRIFT = [c for c in NUM_FEATURES if c != "hour"]

TARGET = "pct_silica_concentrate"  # Se usa para medir performance real, no se envía a la API

# ------------------------------------------------------------------------------
# UMBRALES DE DRIFT (documentados y justificados)
# ------------------------------------------------------------------------------
# PSI (Population Stability Index): convención estándar de industria (scoring
# de riesgo crediticio):
#     PSI < 0.10          -> sin drift relevante
#     0.10 <= PSI < 0.25  -> drift moderado (monitorear, no accionar solo)
#     PSI >= 0.25         -> drift severo (evidencia fuerte de cambio real)
#
# Se usa PSI como criterio de ALERTA (en vez del p-value de KS/Chi²) porque el
# p-value depende del tamaño de muestra: con ventanas grandes, un p-value cae
# a ~0 aunque el cambio práctico sea mínimo. PSI mide la MAGNITUD del cambio,
# no solo si es "estadísticamente distinguible". KS y Chi² igual se calculan y
# se muestran como evidencia estadística complementaria, con el nivel de
# significancia convencional (alpha=0.05, 95% de confianza).
PSI_SIN_DRIFT = 0.10
PSI_SEVERO = 0.25
ALPHA = 0.05
NIVELES_ORDEN = ["sin_drift", "moderado", "severo"]


# ------------------------------------------------------------------------------
# 2. LIMPIEZA Y RESAMPLING (idéntico a la función `preprocesar_planta` de la
#    Fase 2 - Modelado, para garantizar que la "producción" simulada tenga
#    exactamente la misma forma que los datos con los que se entrenó el modelo)
# ------------------------------------------------------------------------------
def preprocesar_planta(df: pd.DataFrame) -> pd.DataFrame:
    df_temp = df.copy()

    # Normalizamos nombres de columnas (ej. "% Iron Feed" -> "pct_iron_feed")
    df_temp.columns = (
        df_temp.columns.str.strip().str.lower().str.replace(" ", "_")
        .str.replace("%", "pct").str.replace(",", "")
        .str.replace(r"\(|\)", "", regex=True)
    )

    # Arreglamos decimales con coma (formato europeo del CSV original).
    # Nota: no basta con chequear dtype == "object": desde pandas 3.0 las
    # columnas de texto llegan como "string[pyarrow]" por defecto, así que
    # usamos is_numeric_dtype para detectar cualquier columna aún no numérica.
    for col in df_temp.columns:
        if col == "date" or pd.api.types.is_numeric_dtype(df_temp[col]):
            continue
        df_temp[col] = df_temp[col].astype(str).str.replace(",", ".").astype(float)

    # Filtros físicos básicos (mismas reglas de negocio de la Fase 1/2)
    mask = (df_temp["amina_flow"] >= 0) & (df_temp["ore_pulp_flow"] > 0)
    df_temp = df_temp[mask].copy()

    # Orden cronológico y resampling a 1 hora (promedio)
    # Nota: se usa 'h' (minúscula) porque el alias 'H' fue removido en pandas >= 2.2
    df_temp["date"] = pd.to_datetime(df_temp["date"], errors="coerce")
    df_temp = df_temp.dropna(subset=["date"]).sort_values("date")
    df_temp.set_index("date", inplace=True)

    serie = df_temp.resample("1h").mean().dropna()

    # Ingeniería de features temporales (mismo criterio que la Fase 2)
    serie["hour"] = serie.index.hour
    serie["dow"] = serie.index.day_name()

    return serie


def cargar_serie_horaria() -> pd.DataFrame:
    """
    Carga la serie horaria limpia. Si ya existe un cache en disco (generado en
    una corrida anterior), lo reutiliza para no tener que reprocesar ~737.000
    filas crudas cada vez que se ejecuta la simulación.
    """
    if SERIE_HORARIA_CACHE.exists():
        print(f"Usando cache existente: {SERIE_HORARIA_CACHE}")
        return pd.read_csv(SERIE_HORARIA_CACHE, index_col="date", parse_dates=["date"])

    if not RAW_CSV.exists():
        print(f"❌ No se encontró el dataset crudo en '{RAW_CSV}'.")
        print("   Copia 'MiningProcess_Flotation_Plant_Database.csv' a esa ruta y vuelve a intentar.")
        sys.exit(1)

    print("Procesando dataset crudo (~737.000 filas cada 20s). Esto puede tardar unos segundos...")
    df_raw = pd.read_csv(RAW_CSV, low_memory=False)
    serie = preprocesar_planta(df_raw)

    # Guardamos el cache para que la próxima ejecución sea instantánea
    serie.to_csv(SERIE_HORARIA_CACHE)
    print(f"✅ Serie horaria calculada y cacheada en '{SERIE_HORARIA_CACHE}' ({len(serie)} horas válidas)")
    return serie


# ------------------------------------------------------------------------------
# 3. PARTICIÓN: REFERENCIA (entrenamiento) VS. STREAM DE PRODUCCIÓN
# ------------------------------------------------------------------------------
def construir_referencia(serie: pd.DataFrame) -> pd.DataFrame:
    """Ventana de referencia = exactamente el set de entrenamiento (marzo, día 1-24)."""
    return serie[(serie.index.month == 3) & (serie.index.day <= 24)]


def construir_stream_produccion(serie: pd.DataFrame) -> pd.DataFrame:
    """Todo lo que el modelo NO vio en entrenamiento, en orden cronológico
    (incluye el holdout de fin de marzo y los 5 meses siguientes con drift)."""
    referencia = (serie.index.month == 3) & (serie.index.day <= 24)
    return serie[~referencia].sort_index()


# ------------------------------------------------------------------------------
# 4. MÉTRICAS DE DRIFT: PSI, KS-TEST Y CHI²
# ------------------------------------------------------------------------------
def psi_numeric(ref_s: pd.Series, cur_s: pd.Series, buckets: int = 10) -> float:
    """PSI para variables numéricas: se arman `buckets` intervalos a partir de
    los cuantiles de la REFERENCIA, y se compara qué proporción de la ventana
    actual cae en cada intervalo."""
    edges = np.unique(np.quantile(ref_s.dropna(), np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    rc, _ = np.histogram(ref_s.dropna(), bins=edges)
    cc, _ = np.histogram(cur_s.dropna(), bins=edges)
    rp = np.where(rc == 0, 1e-4, rc / rc.sum())
    cp = np.where(cc == 0, 1e-4, cc / cc.sum())
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def psi_categorical(ref_s: pd.Series, cur_s: pd.Series) -> float:
    """PSI para variables categóricas: mismo principio, pero las 'categorías'
    son las categorías observadas en vez de intervalos numéricos."""
    cats = sorted(set(ref_s.dropna().unique()) | set(cur_s.dropna().unique()))
    rp = ref_s.value_counts(normalize=True).reindex(cats, fill_value=1e-4).replace(0, 1e-4).values
    cp = cur_s.value_counts(normalize=True).reindex(cats, fill_value=1e-4).replace(0, 1e-4).values
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def clasificar_psi(psi: float) -> str:
    if psi >= PSI_SEVERO:
        return "severo"
    if psi >= PSI_SIN_DRIFT:
        return "moderado"
    return "sin_drift"


def chi2_categorico(ref_s: pd.Series, cur_s: pd.Series):
    """Test Chi² de independencia: ¿la proporción de cada categoría (día de la
    semana) es distinta entre referencia y ventana actual?"""
    categorias = sorted(set(ref_s.dropna().unique()) | set(cur_s.dropna().unique()))
    conteo_ref = ref_s.value_counts().reindex(categorias, fill_value=0)
    conteo_cur = cur_s.value_counts().reindex(categorias, fill_value=0)
    tabla = np.array([conteo_ref.values, conteo_cur.values])
    chi2_stat, p_value, _, _ = chi2_contingency(tabla)
    return float(chi2_stat), float(p_value)


def analizar_ventana(referencia: pd.DataFrame, ventana: pd.DataFrame) -> list:
    """Corre PSI (todas las features) + KS (numéricas) + Chi² (dow) de una
    ventana de producción contra la referencia de entrenamiento."""
    resultados = []

    for col in FEATURES_NUMERICAS_DRIFT:
        ref_col, cur_col = referencia[col].dropna(), ventana[col].dropna()
        psi = psi_numeric(ref_col, cur_col)
        ks_stat, ks_p = ks_2samp(ref_col, cur_col)
        resultados.append({
            "feature": col,
            "tipo": "numerica",
            "psi": round(psi, 4),
            "psi_nivel": clasificar_psi(psi),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 5),
            "chi2_stat": None,
            "chi2_pvalue": None,
        })

    # Variable categórica ("dow"): PSI + Chi² (KS no aplica a categóricas)
    psi_dow = psi_categorical(referencia["dow"], ventana["dow"])
    chi2_stat, chi2_p = chi2_categorico(referencia["dow"], ventana["dow"])
    resultados.append({
        "feature": "dow",
        "tipo": "categorica",
        "psi": round(psi_dow, 4),
        "psi_nivel": clasificar_psi(psi_dow),
        "ks_stat": None,
        "ks_pvalue": None,
        "chi2_stat": round(float(chi2_stat), 4),
        "chi2_pvalue": round(float(chi2_p), 5),
    })

    for r in resultados:
        # PSI manda en la clasificación de drift (ver justificación de umbrales arriba).
        r["drift_detectado"] = r["psi_nivel"] != "sin_drift"

    return resultados


def registrar_drift(timestamp_ventana, resultados: list) -> None:
    """Guarda el resultado del análisis de esta ventana en drift_log.csv (una fila por feature)."""
    filas = [{"timestamp_ventana": timestamp_ventana.isoformat(), **r} for r in resultados]
    pd.DataFrame(filas).to_csv(DRIFT_LOG, mode="a", header=not DRIFT_LOG.exists(), index=False)


def registrar_performance(timestamp_ventana, mae: float, rmse: float, n_muestras: int) -> None:
    """Guarda el MAE/RMSE REAL de la ventana (drift del target/performance, no solo de inputs)."""
    fila = pd.DataFrame([{
        "timestamp_ventana": timestamp_ventana.isoformat(),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_muestras": n_muestras,
    }])
    fila.to_csv(PERFORMANCE_LOG, mode="a", header=not PERFORMANCE_LOG.exists(), index=False)


# ------------------------------------------------------------------------------
# 5. ENVÍO DE CADA HORA SIMULADA A LA API
# ------------------------------------------------------------------------------
def fila_a_payload(fila: pd.Series) -> dict:
    return {
        "hour": int(fila["hour"]),
        "dow": fila["dow"],  # ej. "Monday" (mismo formato que usó el entrenamiento)
        "pct_iron_feed": float(fila["pct_iron_feed"]),
        "pct_silica_feed": float(fila["pct_silica_feed"]),
        "starch_flow": float(fila["starch_flow"]),
        "amina_flow": float(fila["amina_flow"]),
        "ore_pulp_flow": float(fila["ore_pulp_flow"]),
        "ore_pulp_ph": float(fila["ore_pulp_ph"]),
        "ore_pulp_density": float(fila["ore_pulp_density"]),
        "flotation_column_01_level": float(fila["flotation_column_01_level"]),
        "flotation_column_02_level": float(fila["flotation_column_02_level"]),
        "flotation_column_03_level": float(fila["flotation_column_03_level"]),
        "flotation_column_01_air_flow": float(fila["flotation_column_01_air_flow"]),
        "flotation_column_02_air_flow": float(fila["flotation_column_02_air_flow"]),
        "flotation_column_03_air_flow": float(fila["flotation_column_03_air_flow"]),
    }


# ------------------------------------------------------------------------------
# 6. LOOP PRINCIPAL DE SIMULACIÓN
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Simula el flujo cronológico de sensores de la planta hacia la API y detecta Data Drift."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="URL base de la API FastAPI")
    parser.add_argument("--delay", type=float, default=0.02, help="Segundos de espera entre cada hora simulada")
    parser.add_argument("--ventana-horas", type=int, default=168,
                         help="Tamaño de la ventana de producción (horas) para PSI/KS/Chi². Default 168 = 1 semana.")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                         help="Umbral de significancia para KS/Chi² (solo informativo; la alerta la define el PSI)")
    parser.add_argument("--max-horas", type=int, default=None, help="Límite de horas a simular (útil para demos rápidas)")
    args = parser.parse_args()

    # Verificamos que la API esté arriba ANTES de empezar a mandar tráfico
    try:
        requests.get(args.api_url, timeout=3).raise_for_status()
    except requests.exceptions.RequestException:
        print(f"❌ No se pudo conectar a la API en '{args.api_url}'.")
        print("   Levántala primero con: uvicorn app:app --reload")
        sys.exit(1)

    serie = cargar_serie_horaria()
    referencia = construir_referencia(serie)
    produccion = construir_stream_produccion(serie)
    if args.max_horas:
        produccion = produccion.iloc[: args.max_horas]

    print(f"Ventana de referencia (entrenamiento, marzo 1-24): {len(referencia)} horas")
    print(f"Stream de producción a simular: {len(produccion)} horas")
    print(f"Chequeo de drift cada {args.ventana_horas} horas "
          f"(PSI: <{ PSI_SIN_DRIFT}=sin_drift, <{PSI_SEVERO}=moderado, >=lo anterior=severo)\n")

    buffer_ventana, buffer_pred, buffer_real = [], [], []
    for i, (ts, fila) in enumerate(produccion.iterrows(), start=1):
        payload = fila_a_payload(fila)
        try:
            resp = requests.post(f"{args.api_url}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            pred = resp.json()["prediccion_silica_pct"]
            print(f"[{i}/{len(produccion)}] {ts} | pred={pred:.2f}%  real={fila[TARGET]:.2f}%")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error llamando a la API en la fila {ts}: {e}")
            continue

        buffer_ventana.append(fila)
        buffer_pred.append(pred)
        buffer_real.append(float(fila[TARGET]))

        # Cada vez que se acumula una ventana completa, corremos el análisis de drift + performance
        if len(buffer_ventana) >= args.ventana_horas:
            ventana_df = pd.DataFrame(buffer_ventana)
            resultados = analizar_ventana(referencia, ventana_df)
            registrar_drift(ts, resultados)

            pred_arr, real_arr = np.array(buffer_pred), np.array(buffer_real)
            mae = float(np.mean(np.abs(pred_arr - real_arr)))
            rmse = float(np.sqrt(np.mean((pred_arr - real_arr) ** 2)))
            registrar_performance(ts, mae, rmse, len(buffer_ventana))

            peor_nivel = max((r["psi_nivel"] for r in resultados), key=NIVELES_ORDEN.index)
            features_con_drift = [r["feature"] for r in resultados if r["drift_detectado"]]
            emoji = {"sin_drift": "✅", "moderado": "⚠️", "severo": "🚨"}[peor_nivel]
            print(f"   {emoji} Ventana hasta {ts.date()} | Nivel: {peor_nivel.upper()} | "
                  f"MAE real: {mae:.2f}% | variables con drift: {features_con_drift}")

            buffer_ventana, buffer_pred, buffer_real = [], [], []

        time.sleep(args.delay)

    print("\nSimulación finalizada. Revisa 'telemetria.csv' (predicciones), 'drift_log.csv' (drift) "
          "y 'performance_log.csv' (performance real) para el dashboard.")


if __name__ == "__main__":
    main()
