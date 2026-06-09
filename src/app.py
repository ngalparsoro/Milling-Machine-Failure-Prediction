import base64
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Predictor de Parada de Máquina",
    page_icon="⚙️",
    layout="wide",
)

# ── CSS industrial ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg:     #0A0C0F;
  --panel:  #111518;
  --border: #1E2530;
  --rust:   #C0390A;
  --fire:   #E84A0A;
  --amber:  #F07820;
  --green:  #52B788;
  --red:    #E63946;
  --dim:    #60737F;
  --text:   #D8E4EC;
}
.stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--border); }

.block-container { padding-top: 1rem !important; }

/* Hero */
.hero {
  position: relative; width: 100%; height: 220px; overflow: hidden;
  border-bottom: 3px solid var(--rust); margin-bottom: 28px;
}
.hero img { width: 100%; height: 100%; object-fit: cover; object-position: center 40%; filter: brightness(0.28) saturate(0.6); display: block; }
.hero-overlay {
  position: absolute; inset: 0; display: flex; flex-direction: column;
  justify-content: flex-end; padding: 28px 36px 28px 4px;
  background: linear-gradient(to top, rgba(10,12,15,.85) 0%, transparent 60%);
}
.hero-sub   { font-size: 11px; font-weight: 900; letter-spacing: 4px; text-transform: uppercase; color: var(--fire); margin-bottom: 8px; }
.hero-title { font-size: 52px; font-weight: 900; text-transform: uppercase; letter-spacing: -1px; color: #fff; line-height: 1; }
.hero-title span { color: var(--fire); }

.ind-title { font-size: 26px; font-weight: 900; text-transform: uppercase; letter-spacing: 2px; color: #fff; line-height: 1.15; margin-bottom: 4px; }
.ind-title span { color: var(--fire); }
.ind-sub { font-size: 10px; font-weight: 900; letter-spacing: 3px; text-transform: uppercase; color: var(--dim); }

.sec-label {
  font-size: 10px; font-weight: 900; letter-spacing: 3px; text-transform: uppercase;
  color: var(--rust); display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
}
.sec-label::before { content:''; display:inline-block; width:20px; height:2px; background:var(--rust); }

/* Tarjetas de sensor con color dinámico */
.sensor-card { background: var(--panel); border: 1px solid var(--border); padding: 14px 18px; text-align: center; }
.sensor-value        { font-size: 22px; font-weight: 900; letter-spacing: -1px; color: var(--text); }
.sensor-value.ok     { color: var(--green); }
.sensor-value.warn   { color: var(--amber); }
.sensor-value.danger { color: var(--red); }
.sensor-label { font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--dim); margin-top: 2px; }
.sensor-dot        { display:inline-block; width:6px; height:6px; border-radius:50%; vertical-align:middle; margin-right:4px; background:var(--dim); }
.sensor-dot.ok     { background: var(--green); }
.sensor-dot.warn   { background: var(--amber); }
.sensor-dot.danger { background: var(--red); }

/* Badge de resultado de capa */
.layer-badge { padding: 16px 20px; display: flex; flex-direction: column; gap: 4px; }
.layer-badge.ok   { background: rgba(82,183,136,.10); border: 1px solid var(--green); }
.layer-badge.warn { background: rgba(230,57,70,.10);  border: 1px solid var(--red); }
.badge-score { font-size: 36px; font-weight: 900; letter-spacing: -2px; line-height: 1; }
.badge-lbl   { font-size: 10px; font-weight: 900; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 2px; }
.badge-sub   { font-size: 10px; color: var(--dim); letter-spacing: 1px; }

/* Cabeceras de capa */
.layer-hdr { display:flex; align-items:center; gap:14px; padding:14px 20px; margin-bottom:16px; }
.layer-hdr.c1 { background:rgba(74,158,191,.07);  border-left:4px solid #4A9EBF; }
.layer-hdr.c2 { background:rgba(244,162,97,.07);  border-left:4px solid #F4A261; }
.layer-hdr.c3 { background:rgba(45,212,191,.06);   border-left:4px solid #2DD4BF; }
.layer-num  { font-size:28px; font-weight:900; letter-spacing:-2px; opacity:.35; }
.layer-name { font-size:20px; font-weight:900; letter-spacing:3px; text-transform:uppercase; }
.layer-sub  { font-size:10px; color:var(--dim); letter-spacing:1.5px; text-transform:uppercase; margin-top:2px; }

/* Escalada */
.escalate {
  display:flex; align-items:center; justify-content:center; gap:18px;
  padding:18px 28px; margin:0;
  font-size:15px; font-weight:900; letter-spacing:3px; text-transform:uppercase;
}
.escalate-icon { font-size:22px; line-height:1; }

/* Parada */
.stop-banner {
  padding:18px 24px; margin:0;
  background:rgba(82,183,136,.07); border:1px solid rgba(82,183,136,.25); border-left:4px solid var(--green);
  font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:var(--green);
}

/* Sugerencia */
.hint-box   { background: rgba(232,74,10,.07); border: 1px solid var(--fire); padding: 12px 18px; margin-bottom: 6px; }
.hint-title { font-size: 10px; font-weight: 900; letter-spacing: 3px; text-transform: uppercase; color: var(--fire); margin-bottom: 6px; }
.hint-item  { font-size: 13px; color: var(--text); line-height: 1.8; }

/* Señal operativa */
.alarm-banner { padding: 20px 24px; margin-bottom: 10px; }
.alarm-banner.critical { background: rgba(230,57,70,.10); border: 2px solid var(--red);   border-left: 6px solid var(--red); }
.alarm-banner.nominal  { background: rgba(82,183,136,.08); border: 2px solid var(--green); border-left: 6px solid var(--green); }
.alarm-level  { font-size: 10px; font-weight: 900; letter-spacing: 4px; text-transform: uppercase; margin-bottom: 6px; }
.alarm-action { font-size: 20px; font-weight: 900; text-transform: uppercase; letter-spacing: .3px; line-height: 1.2; }
.alarm-prob   { font-size: 38px; font-weight: 900; letter-spacing: -2px; line-height: 1; }
.action-row  { display:flex; gap:12px; align-items:flex-start; padding:9px 0; border-bottom:1px solid var(--border); }
.action-row:last-child { border-bottom:none; }
.action-code { font-size:13px; font-weight:900; min-width:42px; letter-spacing:1px; }
.action-text { font-size:13px; color:var(--text); line-height:1.4; }
.action-sub  { font-size:11px; color:var(--dim); font-family:monospace; margin-top:2px; }

/* Tarjetas de fallo — estilo presentación */
.fallo { background: var(--panel); border: 1px solid var(--border); overflow: hidden; margin-bottom: 5px; display: flex; align-items: stretch; }
.fallo.active { border-color: var(--red); }
.fallo-bar  { width: 4px; flex-shrink: 0; }
.fallo-inner{ display: flex; align-items: center; flex: 1; padding: 11px 16px; gap: 14px; }
.fallo-code { font-size: 16px; font-weight: 900; min-width: 44px; letter-spacing: 1px; }
.fallo-info { flex: 1; }
.fallo-name { font-size: 12px; font-weight: 800; color: #fff; text-transform: uppercase; letter-spacing: .4px; margin-bottom: 2px; }
.fallo-name.dim { color: var(--dim); }
.fallo-desc { font-size: 11.5px; color: var(--dim); line-height: 1.4; }
.fallo-status { font-size: 18px; font-weight: 900; min-width: 28px; text-align: right; }

[data-testid="stSlider"]    label { font-size: 10px !important; font-weight: 900 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--dim) !important; }
[data-testid="stSelectbox"] label { font-size: 10px !important; font-weight: 900 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--dim) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Modelo ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open(BASE / "models/final_model.pkl", "rb") as f:
        return pickle.load(f)

bundle    = load_bundle()
model     = bundle["model"]
threshold = bundle["threshold"]
scaler    = bundle["scaler"]
metrics   = bundle.get("metrics", {})

TWF_MIN = {"L — Low quality": 200, "M — Medium quality": 220, "H — High quality": 240}

# ── Helpers de color ───────────────────────────────────────────────────────────
def clr(value, ok_range=None, warn_range=None):
    """Devuelve 'ok', 'warn' o 'danger' según rangos (low, high) inclusivos."""
    if ok_range and ok_range[0] <= value <= ok_range[1]:
        return "ok"
    if warn_range and warn_range[0] <= value <= warn_range[1]:
        return "warn"
    return "danger"

def sensor_card(value_str, label, css_class):
    return f'<div class="sensor-card" style="flex:1"><div class="sensor-value {css_class}"><span class="sensor-dot {css_class}"></span>{value_str}</div><div class="sensor-label">{label}</div></div>'

def sensor_row(*cards):
    inner = "".join(cards)
    return f'<div style="display:flex;gap:8px">{inner}</div>'

# ══════════════════════════════════════════════════════════════════════════════
# DATOS DE ESCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

ELEC_SCENARIOS = {
    "✅ Operación eléctrica normal": {
        "desc": "Todos los sensores en rango nominal. No se detecta anomalía.",
        "P_total": 487.3, "I_media": 2.4, "THD": 4.8, "PF": 0.93,
        "anomaly_score": 0.22, "is_anomaly": False,
        "cnc_sugeridos": [],
    },
    "⚡ Sobrecarga — potencia alta": {
        "desc": "Consumo muy elevado. Posible sobresfuerzo, desgaste o problemas de disipación térmica.",
        "P_total": 1240.8, "I_media": 5.9, "THD": 9.2, "PF": 0.86,
        "anomaly_score": -0.08, "is_anomaly": True,
        "cnc_sugeridos": ["⚙️ Defecto de pieza", "🔧 Desgaste de herramienta", "📳 Vibración / Chatter"],
    },
    "📉 Señal débil — potencia baja": {
        "desc": "Consumo muy reducido. Posible fallo de potencia por baja carga.",
        "P_total": 98.5, "I_media": 0.7, "THD": 3.1, "PF": 0.97,
        "anomaly_score": -0.12, "is_anomaly": True,
        "cnc_sugeridos": ["⚠️ Anomalía de proceso", "✅ Sin fallo mecánico"],
    },
    "🌡️ Alta distorsión armónica": {
        "desc": "THD y desequilibrio de fases elevados. Indicativo de problemas térmicos o desgaste.",
        "P_total": 631.0, "I_media": 3.1, "THD": 27.6, "PF": 0.71,
        "anomaly_score": -0.09, "is_anomaly": True,
        "cnc_sugeridos": ["🔧 Desgaste de herramienta", "📳 Vibración / Chatter"],
    },
    "⚠️ Fluctuación irregular de carga": {
        "desc": "Picos y caídas inestables de corriente. Posible desgaste, sobresfuerzo o vibración.",
        "P_total": 725.4, "I_media": 3.6, "THD": 11.3, "PF": 0.81,
        "anomaly_score": -0.07, "is_anomaly": True,
        "cnc_sugeridos": ["🔧 Desgaste de herramienta", "⚙️ Defecto de pieza", "📳 Vibración / Chatter"],
    },
}

CNC_SCENARIOS = {
    "✅ Sin fallo mecánico": {
        "desc": "Torque y vibración normales. El controlador no detecta irregularidad.",
        "torque_max": 38.2, "vib_rms": 0.12, "wear_pct": 18.0, "p_fault": 0.04,
        "fault_type": "Normal",
        "ai4i_sugeridos": [],
    },
    "🔧 Desgaste de herramienta": {
        "desc": "Torque creciente y vibración moderada. Herramienta próxima al límite de vida.",
        "torque_max": 67.5, "vib_rms": 0.38, "wear_pct": 84.0, "p_fault": 0.91,
        "fault_type": "Tool Wear",
        "ai4i_sugeridos": ["🔴 TWF — Desgaste herramienta", "🔴 OSF — Sobresfuerzo"],
    },
    "📳 Vibración / Chatter": {
        "desc": "Vibración alta y frecuencia de resonancia detectada. Posible fallo de disipación.",
        "torque_max": 52.1, "vib_rms": 0.74, "wear_pct": 45.0, "p_fault": 0.83,
        "fault_type": "Chatter",
        "ai4i_sugeridos": ["🔴 HDF — Disipación de calor", "🔴 PWF — Fallo de potencia"],
    },
    "⚙️ Defecto de pieza": {
        "desc": "Torque muy elevado y vibración moderada. Resistencia irregular del material.",
        "torque_max": 71.8, "vib_rms": 0.29, "wear_pct": 72.0, "p_fault": 0.87,
        "fault_type": "Workpiece Defect",
        "ai4i_sugeridos": ["🔴 OSF — Sobresfuerzo", "🔴 TWF — Desgaste herramienta"],
    },
    "⚠️ Anomalía de proceso": {
        "desc": "Señales erráticas en múltiples ejes. Anomalía de proceso generalizada.",
        "torque_max": 44.3, "vib_rms": 0.51, "wear_pct": 55.0, "p_fault": 0.76,
        "fault_type": "Process Anomaly",
        "ai4i_sugeridos": ["🔴 HDF — Disipación de calor", "🔴 PWF — Fallo de potencia", "🔴 OSF — Sobresfuerzo"],
    },
}

PROCESS_SCENARIOS = {
    "🔴 TWF — Desgaste herramienta": dict(machine_type="M — Medium quality", air_temp=301.2, proc_temp=311.6, rot_speed=1380, torque=68.0, tool_wear=221),
    "🔴 HDF — Disipación de calor":  dict(machine_type="L — Low quality",    air_temp=302.0, proc_temp=309.5, rot_speed=1200, torque=55.0, tool_wear=110),
    "🔴 PWF — Fallo de potencia":    dict(machine_type="H — High quality",    air_temp=298.0, proc_temp=308.0, rot_speed=2886, torque=3.8,  tool_wear=40),
    "🔴 OSF — Sobresfuerzo":         dict(machine_type="L — Low quality",    air_temp=299.5, proc_temp=310.0, rot_speed=1350, torque=70.0, tool_wear=200),
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ind-sub">Sistema de monitorización</div>', unsafe_allow_html=True)
    st.markdown('<div class="ind-title">PARADA DE<br><span>MÁQUINA</span></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sec-label">Métricas del modelo</div>', unsafe_allow_html=True)
    m = metrics
    st.markdown(f"""
<div style="font-size:12px;line-height:2.1;color:var(--dim)">
F1-Score &nbsp;&nbsp;&nbsp;<span style="color:var(--amber);font-weight:900">{m.get('f1',0.913):.3f}</span><br>
Recall &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:var(--amber);font-weight:900">{m.get('recall',0.853):.3f}</span><br>
Precision &nbsp;<span style="color:var(--amber);font-weight:900">{m.get('precision',0.983):.3f}</span><br>
ROC-AUC &nbsp;&nbsp;&nbsp;<span style="color:var(--amber);font-weight:900">{m.get('roc_auc',0.975):.3f}</span><br>
Umbral &nbsp;&nbsp;&nbsp;&nbsp;<code style="color:var(--amber)">{threshold:.4f}</code>
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:10px;color:var(--dim);letter-spacing:1px">Stacking · LGBM + RF<br>Dataset AI4I 2020 (UCI)<br>10 features · test 2 000 muestras</div>', unsafe_allow_html=True)

# ── Título ─────────────────────────────────────────────────────────────────────
_hero_path = BASE / "hero-machine.jpg"
_hero_b64  = base64.b64encode(_hero_path.read_bytes()).decode()
st.markdown(f"""
<div class="hero">
  <img src="data:image/jpeg;base64,{_hero_b64}" alt="máquina CNC">
  <div class="hero-overlay">
    <div class="hero-sub">⚙️ Demo — Pipeline de predicción por capas</div>
    <div class="hero-title">PREDICTOR DE PARADA<br>DE <span>MÁQUINA</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CAPA 1 — SEÑALES ELÉCTRICAS (SPARK)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="layer-hdr c1">
  <span class="layer-num" style="color:#4A9EBF">01</span>
  <div>
    <div class="layer-name" style="color:#4A9EBF">Sensores eléctricos</div>
    <div class="layer-sub">SPARK TEC &nbsp;·&nbsp; Isolation Forest</div>
  </div>
</div>""", unsafe_allow_html=True)

elec_col, score_col1 = st.columns([3, 1], gap="large")

with elec_col:
    elec_name = st.selectbox("Escenario eléctrico", options=list(ELEC_SCENARIOS.keys()), label_visibility="collapsed")
    elec = ELEC_SCENARIOS[elec_name]
    st.markdown(f'<div style="font-size:13px;color:var(--dim);margin-bottom:12px">{elec["desc"]}</div>', unsafe_allow_html=True)

    # Rangos normales: P 200-1000W, I 1-4.5A, THD <10%, PF >0.88
    c_p   = clr(elec["P_total"], ok_range=(200, 1000), warn_range=(100, 1200))
    c_i   = clr(elec["I_media"], ok_range=(1.0, 4.5),  warn_range=(0.5, 5.5))
    c_thd = clr(elec["THD"],     ok_range=(0, 10),      warn_range=(10, 20))
    c_pf  = "ok" if elec["PF"] >= 0.88 else ("warn" if elec["PF"] >= 0.78 else "danger")

    st.markdown(sensor_row(
        sensor_card(f'{elec["P_total"]:.1f} W',  "Potencia total",      c_p),
        sensor_card(f'{elec["I_media"]:.1f} A',  "Corriente media",     c_i),
        sensor_card(f'{elec["THD"]:.1f} %',      "THD corriente",       c_thd),
        sensor_card(f'{elec["PF"]:.2f}',         "Factor de potencia",  c_pf),
    ), unsafe_allow_html=True)

with score_col1:
    is_elec_anom = elec["is_anomaly"]
    cls1 = "warn" if is_elec_anom else "ok"
    col1 = "#E63946" if is_elec_anom else "#52B788"
    lbl1 = "⚠ ANOMALÍA" if is_elec_anom else "✔ NORMAL"
    st.markdown(f"""
<div class="layer-badge {cls1}">
  <div class="badge-lbl" style="color:{col1}">{lbl1}</div>
  <div class="badge-score" style="color:{col1}">{elec['anomaly_score']:+.3f}</div>
  <div class="badge-sub">ANOMALY SCORE (IF)</div>
</div>""", unsafe_allow_html=True)

# ── ¿Hay anomalía eléctrica? ───────────────────────────────────────────────────
if not is_elec_anom:
    st.markdown("""
<div style="height:30px"></div>
<div class="stop-banner">
  ✔ &nbsp; Sin anomalía eléctrica — sistema estable · no se requiere análisis adicional
</div>
<div style="height:30px"></div>""", unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div style="height:30px"></div>
<div class="escalate" style="color:#4A9EBF;background:rgba(74,158,191,.08);border-top:2px solid rgba(74,158,191,.30);border-bottom:2px solid rgba(74,158,191,.30)">
  <span class="escalate-icon">↓</span>
  <span>ANOMALÍA ELÉCTRICA DETECTADA — ESCALANDO A CAPA 2</span>
  <span class="escalate-icon">↓</span>
</div>
<div style="height:30px"></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CAPA 2 — SEÑALES CNC (INDUSTRIAL KIT)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="layer-hdr c2">
  <span class="layer-num" style="color:#F4A261">02</span>
  <div>
    <div class="layer-name" style="color:#F4A261">Controlador CNC</div>
    <div class="layer-sub">KIT Industrial &nbsp;·&nbsp; Random Forest</div>
  </div>
</div>""", unsafe_allow_html=True)

cnc_col, score_col2 = st.columns([3, 1], gap="large")

cnc_sugeridos = elec["cnc_sugeridos"]

with cnc_col:
    st.markdown(
        f'<div style="font-size:13px;color:var(--dim);margin-bottom:10px">'
        f'La anomalía eléctrica detectada es compatible con estos patrones de fallo mecánico. '
        f'Selecciona uno para ver sus lecturas CNC — puedes explorar cada opción.</div>',
        unsafe_allow_html=True,
    )
    _cnc_default = next(
        (i for i, o in enumerate(cnc_sugeridos) if "Sin fallo" in o), 0
    )
    cnc_name = st.selectbox(
        "Escenario CNC",
        options=cnc_sugeridos,
        index=_cnc_default,
        label_visibility="collapsed",
    )
    cnc = CNC_SCENARIOS[cnc_name]
    st.markdown(f'<div style="font-size:13px;color:var(--dim);margin-top:8px;margin-bottom:12px">{cnc["desc"]}</div>', unsafe_allow_html=True)

    # Rangos normales CNC: torque <55Nm, vib_rms <0.25, wear_pct <50%
    c_tq   = clr(cnc["torque_max"], ok_range=(0, 55),   warn_range=(55, 65))
    c_vib  = clr(cnc["vib_rms"],   ok_range=(0, 0.25),  warn_range=(0.25, 0.5))
    c_wear = clr(cnc["wear_pct"],  ok_range=(0, 50),    warn_range=(50, 75))

    st.markdown(sensor_row(
        sensor_card(f'{cnc["torque_max"]:.1f} Nm', "Torque máximo",     c_tq),
        sensor_card(f'{cnc["vib_rms"]:.2f}',       "Vibración RMS",     c_vib),
        sensor_card(f'{cnc["wear_pct"]:.0f} %',    "Desgaste estimado", c_wear),
        sensor_card(cnc["fault_type"],             "Tipo detectado",    "warn" if cnc["p_fault"] >= 0.5 else "ok"),
    ), unsafe_allow_html=True)

with score_col2:
    is_cnc_fault = cnc["p_fault"] >= 0.5
    cls2 = "warn" if is_cnc_fault else "ok"
    col2 = "#E63946" if is_cnc_fault else "#52B788"
    lbl2 = "⚠ FALLO" if is_cnc_fault else "✔ NORMAL"
    st.markdown(f"""
<div class="layer-badge {cls2}">
  <div class="badge-lbl" style="color:{col2}">{lbl2}</div>
  <div class="badge-score" style="color:{col2}">{cnc['p_fault']:.0%}</div>
  <div class="badge-sub">P(FALLO MECÁNICO)</div>
</div>""", unsafe_allow_html=True)

# ── ¿Hay fallo mecánico? ───────────────────────────────────────────────────────
if not is_cnc_fault:
    st.markdown("""
<div style="height:30px"></div>
<div class="stop-banner">
  ✔ &nbsp; Sin fallo mecánico confirmado — pipeline detenido
</div>
<div style="height:30px"></div>""", unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div style="height:30px"></div>
<div class="escalate" style="color:#F4A261;background:rgba(244,162,97,.08);border-top:2px solid rgba(244,162,97,.30);border-bottom:2px solid rgba(244,162,97,.30)">
  <span class="escalate-icon">↓</span>
  <span>FALLO MECÁNICO CONFIRMADO — ESCALANDO A CAPA 3</span>
  <span class="escalate-icon">↓</span>
</div>
<div style="height:30px"></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CAPA 3 — PARÁMETROS DE PROCESO (AI4I)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="layer-hdr c3">
  <span class="layer-num" style="color:#2DD4BF">03</span>
  <div>
    <div class="layer-name" style="color:#2DD4BF">Parámetros de proceso</div>
    <div class="layer-sub">AI4I &nbsp;·&nbsp; Stacking LGBM + RF</div>
  </div>
</div>""", unsafe_allow_html=True)

ai4i_sugeridos = cnc["ai4i_sugeridos"]
# Usa el primer escenario sugerido como punto de partida para los sliders
preset    = PROCESS_SCENARIOS[ai4i_sugeridos[0]]
proc_name = ai4i_sugeridos[0]

st.markdown(
    f'<div style="font-size:13px;color:var(--dim);margin-bottom:14px">'
    f'Diagnóstico CNC: <strong style="color:var(--text)">{cnc["fault_type"]}</strong>. '
    f'Los sliders están pre-cargados con valores representativos de ese fallo. '
    f'Ajústalos para explorar cómo responde el modelo.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
col_out, col_in = st.columns([3, 2], gap="large")

with col_in:
    st.markdown('<div class="sec-label">Tipo de máquina</div>', unsafe_allow_html=True)
    machine_type = st.selectbox(
        "Tipo de máquina",
        options=["L — Low quality", "M — Medium quality", "H — High quality"],
        index=["L — Low quality", "M — Medium quality", "H — High quality"].index(preset["machine_type"]),
        label_visibility="collapsed",
    )
    type_enc = {"L — Low quality": 0, "M — Medium quality": 1, "H — High quality": 2}[machine_type]

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Parámetros del proceso</div>', unsafe_allow_html=True)
    air_temp  = st.slider("Temperatura aire (K)",       295.0, 305.0, float(preset["air_temp"]),  0.1)
    proc_temp = st.slider("Temperatura proceso (K)",    306.0, 314.0, float(preset["proc_temp"]), 0.1)
    rot_speed = st.slider("Velocidad rotación (rpm)",   1168,  2886,  int(preset["rot_speed"]),   1)
    torque    = st.slider("Par motor (Nm)",             3.8,   76.6,  float(preset["torque"]),    0.1)
    tool_wear = st.slider("Desgaste herramienta (min)", 0,     253,   int(preset["tool_wear"]),   1)

    # Features de ingeniería
    power       = torque * rot_speed * 2 * np.pi / 60
    temp_diff   = proc_temp - air_temp
    wear_torque = tool_wear * torque
    wear_ratio  = tool_wear / TWF_MIN[machine_type]

    # Colores según umbrales del dataset AI4I
    c_pow  = clr(power,       ok_range=(3500, 9000), warn_range=(2500, 10000))
    c_tdif = "ok" if temp_diff >= 8.6 else ("warn" if temp_diff >= 7.0 else "danger")
    c_wt   = clr(wear_torque, ok_range=(0, 9000),    warn_range=(9000, 11000))
    c_wr   = clr(wear_ratio,  ok_range=(0, 0.70),    warn_range=(0.70, 0.85))

    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Features calculadas automáticamente</div>', unsafe_allow_html=True)
    st.markdown(
        sensor_row(
            sensor_card(f'{power:.0f} W',     "Potencia",    c_pow),
            sensor_card(f'{temp_diff:.1f} K', "Δ Temp",      c_tdif),
        ) + '<div style="height:5px"></div>' +
        sensor_row(
            sensor_card(f'{wear_torque:.0f}', "Wear×Torque", c_wt),
            sensor_card(f'{wear_ratio:.2f}',  "Wear ratio",  c_wr),
        ),
        unsafe_allow_html=True,
    )

    checks = [
        ("TWF", "Desgaste herramienta",
         tool_wear >= TWF_MIN[machine_type] * 0.85 and torque > 60,
         f"Desgaste {tool_wear} min / umbral {TWF_MIN[machine_type]} min — Torque {torque:.1f} Nm"),
        ("HDF", "Disipación de calor",
         temp_diff < 8.6 and rot_speed < 1380,
         f"ΔT = {temp_diff:.1f} K (umbral 8.6 K) — RPM {rot_speed}"),
        ("PWF", "Fallo de potencia",
         power < 3500 or power > 9000,
         f"Potencia {power:.0f} W — rango seguro [3 500–9 000 W]"),
        ("OSF", "Sobresfuerzo",
         wear_torque > 11000,
         f"Wear×Torque = {wear_torque:.0f} (umbral 11 000)"),
    ]
    active_faults = [(code, name, detail) for code, name, at_risk, detail in checks if at_risk]

# ── Predicción ─────────────────────────────────────────────────────────────────
X          = np.array([[type_enc, air_temp, proc_temp, rot_speed,
                         torque, tool_wear, power, temp_diff, wear_torque, wear_ratio]])
prob       = model.predict_proba(scaler.transform(X))[0, 1]
prediction = int(prob >= threshold)

FAULT_ACTIONS = {
    "TWF": ("Cambiar herramienta antes del próximo ciclo",          "#00B4D8"),
    "HDF": ("Revisar sistema de refrigeración — aumentar caudal",   "#F4A261"),
    "PWF": ("Verificar carga eléctrica y variador de frecuencia",   "#E63946"),
    "OSF": ("Reducir profundidad de corte — verificar fijación",    "#52B788"),
}

with col_out:
    st.markdown('<div class="sec-label">Señal del sistema</div>', unsafe_allow_html=True)

    # ── Banner principal ───────────────────────────────────────────────────────
    if prediction == 1:
        banner_cls  = "critical"
        lvl_color   = "#E63946"
        lvl_text    = "ALARMA CRÍTICA"
        action_text = "PARADA PREVENTIVA RECOMENDADA"
    else:
        banner_cls  = "nominal"
        lvl_color   = "#52B788"
        lvl_text    = "SISTEMA NOMINAL"
        action_text = "SIN ACCIÓN REQUERIDA"

    st.markdown(f"""
<div class="alarm-banner {banner_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
    <div>
      <div class="alarm-level"  style="color:{lvl_color}">{lvl_text}</div>
      <div class="alarm-action" style="color:{lvl_color}">{action_text}</div>
    </div>
    <div class="alarm-prob" style="color:{lvl_color}">{prob:.1%}</div>
  </div>
  <div style="font-size:11px;color:var(--dim);margin-top:8px">
    P(fallo) · umbral de producción {threshold:.4f}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Acciones requeridas (si hay fallos) ────────────────────────────────────
    if prediction == 1 and active_faults:
        rows = "".join(
            f'<div class="action-row">'
            f'<div class="action-code" style="color:{FAULT_ACTIONS.get(code, ("", "#D8E4EC"))[1]}">{code}</div>'
            f'<div>'
            f'<div class="action-text">{FAULT_ACTIONS.get(code, (name, ""))[0]}</div>'
            f'<div class="action-sub">{detail}</div>'
            f'</div></div>'
            for code, name, detail in active_faults
        )
        st.markdown(f"""
<div style="background:var(--panel);border:1px solid var(--border);
            padding:14px 18px;margin-bottom:10px">
  <div style="font-size:10px;font-weight:900;letter-spacing:3px;text-transform:uppercase;
              color:var(--dim);margin-bottom:10px">Acciones requeridas</div>
  {rows}
</div>""", unsafe_allow_html=True)

    # ── Gauge semicircular con escala exponencial ──────────────────────────────
    def _p2a(p, k=3):
        return np.pi * (1 - (1 - np.clip(p, 0, 0.9999)) ** (1 / k))

    GAUGE_ZONES = [
        (0.00, 0.85,      "#52B788"),
        (0.85, threshold, "#F07820"),
        (threshold, 1.00, "#E63946"),
    ]
    TICKS = [0.0, 0.5, 0.80, 0.90, 0.95, 0.99]

    fig, ax = plt.subplots(figsize=(4, 1.7))
    fig.patch.set_facecolor("#111518")
    ax.set_facecolor("#111518")
    R, LW = 0.72, 16

    for p0, p1, col in GAUGE_ZONES:
        ts = np.linspace(_p2a(p0), _p2a(p1), 120)
        ax.plot(R * np.cos(ts), R * np.sin(ts),
                color=col, lw=LW, alpha=0.18, solid_capstyle="butt")

    for p0, p1, col in GAUGE_ZONES:
        lo, hi = max(p0, 0.0), min(p1, prob)
        if hi <= lo:
            continue
        ts = np.linspace(_p2a(lo), _p2a(hi), 120)
        ax.plot(R * np.cos(ts), R * np.sin(ts),
                color=col, lw=LW, solid_capstyle="butt")

    t_thr = _p2a(threshold)
    ax.plot([0.57 * np.cos(t_thr), 0.87 * np.cos(t_thr)],
            [0.57 * np.sin(t_thr), 0.87 * np.sin(t_thr)],
            color="#F07820", lw=1.8)
    ax.text(0.96 * np.cos(t_thr), 0.96 * np.sin(t_thr),
            f"{threshold:.2f}", ha="center", va="center",
            fontsize=5.5, color="#F07820", fontfamily="monospace")

    for pv in TICKS:
        t = _p2a(pv)
        ax.plot([0.64 * np.cos(t), 0.72 * np.cos(t)],
                [0.64 * np.sin(t), 0.72 * np.sin(t)],
                color="#2A3540", lw=1.5)
        ax.text(0.51 * np.cos(t), 0.51 * np.sin(t),
                f"{pv:.0%}", ha="center", va="center",
                fontsize=6, color="#60737F", fontfamily="monospace")

    t_n = _p2a(prob)
    ax.annotate("", xy=(0.60 * np.cos(t_n), 0.60 * np.sin(t_n)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#D8E4EC", lw=2))
    ax.plot(0, 0, "o", ms=6, color="#D8E4EC")

    g_col = "#E63946" if prob >= threshold else ("#F07820" if prob >= 0.85 else "#52B788")
    ax.text(0, -0.15, f"{prob:.1%}", ha="center", va="center",
            fontsize=15, fontweight="bold", color=g_col)
    ax.text(0, -0.30, "P(FALLO)", ha="center", va="center",
            fontsize=7, color="#60737F", fontweight="bold")

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.22, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.markdown('<div style="font-size:10px;font-weight:900;letter-spacing:2px;text-transform:uppercase;color:var(--dim);margin-top:12px;margin-bottom:2px">Probabilidad de fallo</div>', unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Referencia ─────────────────────────────────────────────────────────────────
st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)
with st.expander("Tipos de fallo — referencia"):
    st.markdown("""
| Código | Nombre | Condición típica |
|---|---|---|
| **TWF** | Tool Wear Failure | Desgaste 200–240 min con torque elevado |
| **HDF** | Heat Dissipation Failure | ΔT < 8.6 K y velocidad < 1380 rpm |
| **PWF** | Power Failure | Potencia fuera de [3 500–9 000] W |
| **OSF** | Overstrain Failure | Wear × Torque > 11 000 |
| **RNF** | Random Failure | Probabilidad aleatoria independiente |
""")
