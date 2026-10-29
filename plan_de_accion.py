# ==============================================================================
# FASE 5: PLAN DE ACCIÓN ANTE DRIFT
# Archivo: plan_de_accion.py
# ==============================================================================
#
# Qué hace:
# 1. Lee 'drift_log.csv' y determina el nivel de alerta de la ÚLTIMA ventana
#    evaluada por simulacion.py (el peor psi_nivel entre todas las variables).
# 2. Según ese nivel, actúa:
#      - sin_drift  -> no hace nada, informa que todo está en orden.
#      - moderado   -> deja una alerta registrada (ALERTA_DRIFT.log) pidiendo
#                      revisión humana. No reentrena solo: un cambio moderado
#                      puede ser ruido normal del proceso.
#      - severo     -> dispara un REENTRENAMIENTO SEMI-AUTOMÁTICO: usa los
#                      datos MÁS RECIENTES de producción (no marzo) para
#                      entrenar un modelo candidato con la misma arquitectura,
#                      lo evalúa con un nuevo split temporal, y lo guarda SIN
#                      reemplazar el modelo en producción.
#
# Por qué semi-automático y no 100% automático: promover un modelo a
# producción sin revisión humana es riesgoso (podría entrenarse con una
# ventana corta y ruidosa, o sobre datos con un problema de sensor puntual).
# El gatillo SÍ está implementado de punta a punta; la decisión final de
# reemplazar el modelo productivo queda en manos humanas (ver el mensaje final
# que imprime este script).
# ------------------------------------------------------------------------------

import json
import pathlib
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DRIFT_LOG = pathlib.Path("drift_log.csv")
SERIE_HORARIA = pathlib.Path("data_flotacion/serie_horaria.csv")
MODELS_DIR = pathlib.Path("models_flotacion")
TRAIN_METRICS_PATH = MODELS_DIR / "train_metrics.json"
ALERTA_LOG = pathlib.Path("ALERTA_DRIFT.log")

NIVELES_ORDEN = ["sin_drift", "moderado", "severo"]

# Misma arquitectura EXACTA que la Fase 2 (Modelado), para que el modelo
# candidato sea directamente comparable con el que está en producción.
NUM_FEATURES = [
    "hour", "pct_iron_feed", "pct_silica_feed", "starch_flow", "amina_flow",
    "ore_pulp_flow", "ore_pulp_ph", "ore_pulp_density",
    "flotation_column_01_level", "flotation_column_02_level", "flotation_column_03_level",
    "flotation_column_01_air_flow", "flotation_column_02_air_flow", "flotation_column_03_air_flow",
]
CAT_FEATURES = ["dow"]
TARGET = "pct_silica_concentrate"

DIAS_VENTANA_REENTRENO = 30  # usar el último mes de producción disponible como nuevo set de entrenamiento


def leer_nivel_alerta_actual():
    if not DRIFT_LOG.exists():
        print("No existe 'drift_log.csv' todavía. Corre primero 'python simulacion.py'.")
        sys.exit(1)

    drift = pd.read_csv(DRIFT_LOG, parse_dates=["timestamp_ventana"])
    ultima_ts = drift["timestamp_ventana"].max()
    ultima_ventana = drift[drift["timestamp_ventana"] == ultima_ts]
    peor_nivel = max(ultima_ventana["psi_nivel"], key=NIVELES_ORDEN.index)
    return peor_nivel, ultima_ts, ultima_ventana


def registrar_alerta(nivel: str, ts_ventana, detalle: pd.DataFrame) -> None:
    momento = datetime.now(timezone.utc).isoformat()
    variables_afectadas = detalle.loc[detalle["psi_nivel"] != "sin_drift", "feature"].tolist()
    with open(ALERTA_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"[{momento}] Ventana simulada={ts_ventana} | nivel={nivel} | "
            f"variables afectadas={variables_afectadas}\n"
        )


def construir_pipeline() -> Pipeline:
    preprocesador = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), CAT_FEATURES),
    ])
    return Pipeline([
        ("pre", preprocesador),
        ("clf", HistGradientBoostingRegressor(random_state=42, max_depth=6)),
    ])


def reentrenar_con_datos_recientes() -> dict:
    if not SERIE_HORARIA.exists():
        print(f"No se encontró '{SERIE_HORARIA}'. Corre primero 'python simulacion.py' (genera el cache).")
        sys.exit(1)

    serie = pd.read_csv(SERIE_HORARIA, index_col="date", parse_dates=["date"]).sort_index()

    corte = serie.index.max() - pd.Timedelta(days=DIAS_VENTANA_REENTRENO)
    reciente = serie[serie.index > corte]

    # Split temporal dentro de la ventana reciente (80% train / 20% test) —
    # misma lógica de honestidad que la Fase 2: nunca mezclar aleatoriamente.
    corte_train = reciente.index.min() + (reciente.index.max() - reciente.index.min()) * 0.8
    train = reciente[reciente.index <= corte_train]
    test = reciente[reciente.index > corte_train]

    if len(train) < 20 or len(test) < 5:
        print(f"No hay suficientes datos recientes para reentrenar de forma confiable "
              f"(train={len(train)}, test={len(test)}). Se aborta el reentrenamiento automático.")
        sys.exit(1)

    X_train, y_train = train[NUM_FEATURES + CAT_FEATURES], train[TARGET]
    X_test, y_test = test[NUM_FEATURES + CAT_FEATURES], test[TARGET]

    candidato = construir_pipeline()
    candidato.fit(X_train, y_train)
    pred = candidato.predict(X_test)

    metricas_candidato = {
        "ventana_entrenamiento": f"últimos {DIAS_VENTANA_REENTRENO} días de producción simulada",
        "rango_fechas": [str(reciente.index.min()), str(reciente.index.max())],
        "n_train": len(train),
        "n_test": len(test),
        "test_mae": round(mean_absolute_error(y_test, pred), 3),
        "test_rmse": round(mean_squared_error(y_test, pred) ** 0.5, 3),
        "test_r2": round(r2_score(y_test, pred), 3),
        "generado_en": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(candidato, MODELS_DIR / "modelo_flotacion_candidato.joblib")
    with open(MODELS_DIR / "train_metrics_candidato.json", "w", encoding="utf-8") as f:
        json.dump(metricas_candidato, f, indent=2, ensure_ascii=False)

    return metricas_candidato


def main():
    nivel, ts_ventana, detalle = leer_nivel_alerta_actual()
    print(f"Última ventana evaluada: {ts_ventana} | Nivel de alerta: {nivel.upper()}")

    if nivel == "sin_drift":
        print("✅ Sin drift relevante. No se requiere ninguna acción.")
        return

    registrar_alerta(nivel, ts_ventana, detalle)

    if nivel == "moderado":
        print("⚠️ Drift MODERADO detectado. Se registró la alerta en 'ALERTA_DRIFT.log'.")
        print("   Acción recomendada: revisión humana de las variables afectadas.")
        print("   No se gatilla reentrenamiento automático todavía (podría ser ruido normal del proceso).")
        return

    # nivel == "severo"
    print("🚨 Drift SEVERO detectado. Registrada la alerta en 'ALERTA_DRIFT.log'.")
    print("Gatillando reentrenamiento semi-automático con los datos más recientes...\n")
    metricas_candidato = reentrenar_con_datos_recientes()

    with open(TRAIN_METRICS_PATH, encoding="utf-8") as f:
        metricas_originales = json.load(f)
    mae_original = metricas_originales["resultados"][metricas_originales["modelo_campeon"]]["test_mae"]

    print("Modelo candidato entrenado con datos recientes:")
    print(f"  Ventana de reentrenamiento: {metricas_candidato['rango_fechas'][0]} -> {metricas_candidato['rango_fechas'][1]}")
    print(f"  MAE modelo ORIGINAL (marzo):   {mae_original}%")
    print(f"  MAE modelo CANDIDATO (reciente): {metricas_candidato['test_mae']}%")
    print(f"  Guardado en: {MODELS_DIR / 'modelo_flotacion_candidato.joblib'}")
    print()
    print("Este modelo candidato NO reemplaza automáticamente al de producción.")
    print("Para promoverlo: revisa 'models_flotacion/train_metrics_candidato.json' y, si el")
    print("desempeño es mejor, renómbralo/cópialo como 'modelo_flotacion.joblib'.")


if __name__ == "__main__":
    main()
