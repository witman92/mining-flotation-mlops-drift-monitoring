# ==============================================================================
# FASE 5: DASHBOARD DE MONITOREO (STREAMLIT)
# Archivo: dashboard.py
# ==============================================================================
#
# Este dashboard NO llama al modelo ni ejecuta inferencia: solo lee los
# archivos locales que los demás componentes van generando:
#   - telemetria.csv      -> lo escribe app.py en cada petición a /predict
#   - drift_log.csv       -> lo escribe simulacion.py (PSI + KS + Chi² por ventana)
#   - performance_log.csv -> lo escribe simulacion.py (MAE/RMSE real por ventana)
#
# Al ser Streamlit, el script se vuelve a ejecutar completo cada vez que el
# usuario interactúa con un widget (o aprieta "Actualizar"), así que basta con
# leer los CSV de nuevo en cada corrida para tener datos frescos.
#
# Paleta de color (ver skill dataviz / references/palette.md):
#   - Azul (#2a78d6): serie secuencial única (predicciones, MAE).
#   - Verde/Ámbar/Rojo fijos (estado): reservados para sin_drift/moderado/severo,
#     nunca se reutilizan para otra serie.
# ------------------------------------------------------------------------------

import pathlib
import subprocess
import sys

import altair as alt
import pandas as pd
import streamlit as st

COLOR_PRIMARY = "#2a78d6"
COLOR_GOOD = "#0ca30c"
COLOR_WARNING = "#fab219"
COLOR_CRITICAL = "#d03b3b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

NIVEL_COLOR = {"sin_drift": COLOR_GOOD, "moderado": COLOR_WARNING, "severo": COLOR_CRITICAL}
NIVEL_ORDEN = ["sin_drift", "moderado", "severo"]
NIVEL_LABEL = {"sin_drift": "Sin drift", "moderado": "Drift moderado", "severo": "Drift severo"}

# Mismos umbrales que simulacion.py (ver justificación en ese archivo)
PSI_SIN_DRIFT = 0.10
PSI_SEVERO = 0.25

TELEMETRIA_PATH = pathlib.Path("telemetria.csv")
DRIFT_LOG_PATH = pathlib.Path("drift_log.csv")
PERFORMANCE_LOG_PATH = pathlib.Path("performance_log.csv")

st.set_page_config(page_title="Monitoreo MLOps - Planta de Flotación", layout="wide")

st.title("Monitoreo de Producción y Data Drift — Planta de Flotación")
st.caption(
    "Lee 'telemetria.csv' (predicciones), 'drift_log.csv' (PSI + KS + Chi²) y "
    "'performance_log.csv' (MAE/RMSE real), todos generados por app.py y simulacion.py."
)

if st.button("Actualizar datos"):
    st.rerun()

# ------------------------------------------------------------------------------
# 1. TELEMETRÍA (predicciones registradas por la API)
# ------------------------------------------------------------------------------
if not TELEMETRIA_PATH.exists():
    st.warning(
        "Todavía no hay datos en 'telemetria.csv'. Levanta la API con "
        "`uvicorn app:app --reload` y corre `python simulacion.py` para generar tráfico."
    )
    st.stop()

telemetria = pd.read_csv(TELEMETRIA_PATH, parse_dates=["timestamp_inferencia"])
if telemetria.empty:
    st.info("'telemetria.csv' existe pero todavía no tiene filas. Envía al menos una predicción.")
    st.stop()

drift = pd.read_csv(DRIFT_LOG_PATH, parse_dates=["timestamp_ventana"]) if DRIFT_LOG_PATH.exists() else None
performance = pd.read_csv(PERFORMANCE_LOG_PATH, parse_dates=["timestamp_ventana"]) if PERFORMANCE_LOG_PATH.exists() else None

# --- Nivel de alerta actual (peor psi_nivel de la última ventana evaluada) ---
nivel_actual, ts_ultima_ventana = None, None
if drift is not None and not drift.empty:
    ts_ultima_ventana = drift["timestamp_ventana"].max()
    ultima_ventana = drift[drift["timestamp_ventana"] == ts_ultima_ventana]
    nivel_actual = max(ultima_ventana["psi_nivel"], key=NIVEL_ORDEN.index)

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicciones registradas", f"{len(telemetria):,}")
col2.metric("Última predicción", f"{telemetria['prediccion_silica_pct'].iloc[-1]:.2f}%")

if performance is not None and not performance.empty:
    mae_ultima = performance["mae"].iloc[-1]
    mae_referencia = performance["mae"].iloc[0]
    col3.metric("MAE real (última ventana)", f"{mae_ultima:.2f}%", delta=f"{mae_ultima - mae_referencia:+.2f}% vs. 1ra ventana", delta_color="inverse")
else:
    col3.metric("MAE real (última ventana)", "sin datos aún")

if nivel_actual is not None:
    col4.metric("Nivel de alerta actual", NIVEL_LABEL[nivel_actual].upper())
else:
    col4.metric("Nivel de alerta actual", "sin datos aún")

st.divider()

# ------------------------------------------------------------------------------
# 2. PLAN DE ACCIÓN (Fase 5) — visible arriba de todo, es lo más urgente
# ------------------------------------------------------------------------------
if nivel_actual == "severo":
    st.error(
        f"🚨 **Nivel de alerta SEVERO** en la ventana que termina el {ts_ultima_ventana.date()}. "
        "Se recomienda ejecutar el plan de acción (reentrenamiento semi-automático)."
    )
    if st.button("▶ Ejecutar plan de acción ahora (plan_de_accion.py)"):
        with st.spinner("Ejecutando plan_de_accion.py..."):
            resultado = subprocess.run(
                [sys.executable, "plan_de_accion.py"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        st.code((resultado.stdout or "") + (resultado.stderr or ""), language="text")
        st.info("El modelo candidato (si se generó) queda en 'models_flotacion/modelo_flotacion_candidato.joblib', "
                "sin reemplazar el modelo en producción. Revisa 'train_metrics_candidato.json' antes de promoverlo.")
elif nivel_actual == "moderado":
    st.warning(
        f"⚠️ **Drift moderado** en la ventana que termina el {ts_ultima_ventana.date()}. "
        "Se recomienda revisión humana; corre `python plan_de_accion.py` para dejar la alerta registrada."
    )
elif nivel_actual == "sin_drift":
    st.success(f"✅ Sin drift relevante en la ventana que termina el {ts_ultima_ventana.date()}.")

st.divider()

# ------------------------------------------------------------------------------
# 3. EVOLUCIÓN DE LAS PREDICCIONES
# ------------------------------------------------------------------------------
st.subheader("Evolución de la predicción de % Sílice")
chart_pred = (
    alt.Chart(telemetria)
    .mark_line(color=COLOR_PRIMARY, strokeWidth=2)
    .encode(
        x=alt.X("timestamp_inferencia:T", title="Tiempo"),
        y=alt.Y("prediccion_silica_pct:Q", title="% Sílice predicho"),
        tooltip=["timestamp_inferencia:T", "prediccion_silica_pct:Q"],
    )
    .properties(height=280)
    .configure_axis(gridColor=GRIDLINE, labelColor=INK_MUTED, titleColor=INK_SECONDARY)
)
st.altair_chart(chart_pred, use_container_width=True)

with st.expander("Ver últimas filas de telemetria.csv"):
    st.dataframe(telemetria.tail(20), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# 4. PERFORMANCE REAL EN EL TIEMPO (drift del target, no solo de los inputs)
# ------------------------------------------------------------------------------
st.divider()
st.subheader("Performance real del modelo (MAE por ventana)")
st.caption(
    "A diferencia del drift de inputs, esto mide si el modelo efectivamente se equivoca más "
    "con el tiempo. Un modelo puede tener inputs con drift y aun así seguir funcionando bien "
    "(o viceversa) — por eso esta señal es complementaria, no un reemplazo del PSI."
)

if performance is None or performance.empty:
    st.info("Todavía no existe 'performance_log.csv'. Corre `python simulacion.py` para generarlo.")
else:
    chart_mae = (
        alt.Chart(performance)
        .mark_line(color=COLOR_PRIMARY, strokeWidth=2, point=True)
        .encode(
            x=alt.X("timestamp_ventana:T", title="Ventana de producción"),
            y=alt.Y("mae:Q", title="MAE real (% Sílice)"),
            tooltip=["timestamp_ventana:T", "mae:Q", "rmse:Q", "n_muestras:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart_mae, use_container_width=True)

st.divider()

# ------------------------------------------------------------------------------
# 5. DRIFT DE VARIABLES: PSI (principal) + KS/Chi² (evidencia complementaria)
# ------------------------------------------------------------------------------
st.subheader("Drift de variables (PSI, KS-test y Chi²)")
st.caption(
    "Umbral de alerta basado en PSI (Population Stability Index): "
    "< 0.10 sin drift · 0.10–0.25 moderado · ≥ 0.25 severo. "
    "KS (numéricas) y Chi² (día de la semana) se muestran como evidencia estadística "
    "adicional (p-value < 0.05 = diferencia estadísticamente significativa)."
)

if drift is None or drift.empty:
    st.info("Todavía no existe 'drift_log.csv'. Corre `python simulacion.py` para generarlo.")
    st.stop()

# --- Nivel de alerta por ventana (a lo largo del tiempo) ---
st.markdown("**Nivel de alerta por ventana (a lo largo del tiempo)**")
nivel_por_ventana = (
    drift.groupby("timestamp_ventana")["psi_nivel"]
    .apply(lambda s: max(s, key=NIVEL_ORDEN.index))
    .reset_index()
)
nivel_por_ventana["orden"] = nivel_por_ventana["psi_nivel"].map(NIVEL_ORDEN.index)
chart_alertas = (
    alt.Chart(nivel_por_ventana)
    .mark_bar()
    .encode(
        x=alt.X("timestamp_ventana:T", title="Ventana de producción"),
        y=alt.Y("orden:Q", title="Nivel de alerta", scale=alt.Scale(domain=[0, 2]),
                axis=alt.Axis(values=[0, 1, 2], labelExpr="['Sin drift','Moderado','Severo'][datum.value]")),
        color=alt.Color(
            "psi_nivel:N",
            scale=alt.Scale(domain=NIVEL_ORDEN, range=[COLOR_GOOD, COLOR_WARNING, COLOR_CRITICAL]),
            legend=alt.Legend(title="Nivel"),
        ),
        tooltip=["timestamp_ventana:T", "psi_nivel:N"],
    )
    .properties(height=220)
)
st.altair_chart(chart_alertas, use_container_width=True)

# --- Variable a inspeccionar en detalle ---
features_disponibles = sorted(drift["feature"].unique())
feature_sel = st.selectbox("Variable a inspeccionar en detalle:", features_disponibles)
drift_feature = drift[drift["feature"] == feature_sel].copy()

chart_psi = (
    alt.Chart(drift_feature)
    .mark_line(point=alt.OverlayMarkDef(size=80))
    .encode(
        x=alt.X("timestamp_ventana:T", title="Ventana de producción"),
        y=alt.Y("psi:Q", title="PSI"),
        color=alt.Color(
            "psi_nivel:N",
            scale=alt.Scale(domain=NIVEL_ORDEN, range=[COLOR_GOOD, COLOR_WARNING, COLOR_CRITICAL]),
            legend=alt.Legend(title="Nivel"),
        ),
        tooltip=["timestamp_ventana:T", "psi:Q", "psi_nivel:N", "ks_pvalue:Q", "chi2_pvalue:Q"],
    )
    .properties(height=280)
)
lineas_umbral = pd.DataFrame({"y": [PSI_SIN_DRIFT, PSI_SEVERO]})
regla_umbrales = alt.Chart(lineas_umbral).mark_rule(strokeDash=[4, 4], color=INK_MUTED).encode(y="y:Q")
st.altair_chart(chart_psi + regla_umbrales, use_container_width=True)

# --- Snapshot de la última ventana evaluada ---
ultima_ts = drift["timestamp_ventana"].max()
st.markdown(f"**Detalle de la última ventana evaluada: {ultima_ts}**")
ultima_ventana = drift[drift["timestamp_ventana"] == ultima_ts].copy()
ultima_ventana["Estado"] = ultima_ventana["psi_nivel"].map(
    {"sin_drift": "🟢 Sin drift", "moderado": "🟡 Moderado", "severo": "🔴 Severo"}
)
st.dataframe(
    ultima_ventana[["feature", "tipo", "psi", "Estado", "ks_pvalue", "chi2_pvalue"]]
    .sort_values("psi", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# --- Ranking histórico: qué variables se salen de rango con más frecuencia ---
st.subheader("Variables más inestables (histórico completo)")
ranking = (
    drift.groupby("feature")["drift_detectado"]
    .mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"drift_detectado": "pct_ventanas_con_drift"})
)
chart_ranking = (
    alt.Chart(ranking)
    .mark_bar(color=COLOR_PRIMARY)
    .encode(
        x=alt.X("pct_ventanas_con_drift:Q", title="% de ventanas con drift (PSI >= 0.10)"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", "pct_ventanas_con_drift"],
    )
    .properties(height=350)
)
st.altair_chart(chart_ranking, use_container_width=True)
