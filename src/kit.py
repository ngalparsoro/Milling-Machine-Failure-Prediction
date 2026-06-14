"""
kit.py
──────
Constantes, carga y feature engineering del dataset KIT Karlsruhe
(CNC industrial, 33 experimentos a 500 Hz).

Única fuente de verdad para las etiquetas de los experimentos: si se corrige
una etiqueta aquí, todos los notebooks (01_03, 02_03, 03_03, 04_03) la heredan.

Uso en notebooks:
    import sys; sys.path.insert(0, '../../src')
    from kit import FAULT_TYPE, FAULT_NAMES, FAULT_COLORS, load_kit, time_features
"""

import os

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

from paths import RAW

KIT_DIR = RAW / "kit_extracted"
FS = 500  # Hz de muestreo

KEY_SIGNALS = [
    "TORQUE|1", "TORQUE|2", "TORQUE|3", "TORQUE|6",
    "CURRENT|1", "CURRENT|2", "CURRENT|3", "CURRENT|6",
    "POWER|1", "POWER|2", "POWER|3", "POWER|6",
    "LOAD|1", "LOAD|2", "LOAD|3", "LOAD|6",
]

# Etiquetas multi-clase — 33 experimentos completos
FAULT_TYPE = {
    # 0 — Normal
    "IM-01F": 0, "IM-01R": 0,
    "IMP-01": 0, "IMP-02": 0, "IMP-03": 0, "IMP-04": 0,
    "IMP-05": 0, "IMP-06": 0, "IMP-07": 0, "IMP-08": 0,
    "IMP-09": 0, "IMP-10": 0, "IMP-11": 0, "IMP-12": 0, "IMP-BASE": 0,
    "TF-01": 0, "TF-02": 0, "TF-03": 0,
    # 1 — Tool Wear
    "IM-01R-A01": 1, "IM-01R-A02": 1, "IM-01R-A03": 1,
    "IM-01R-A04": 1, "IM-01R-A05": 1,
    # 2 — Chatter
    "IM-02F-A01": 2,
    # 3 — Process Anomaly (overload, feedrate, built-up edge, stock)
    "IM-01F-A01": 3, "TF-01-A01": 3, "TF-02-A01": 3,
    "TF-03-A01": 3, "TF-03-A02": 3,
    # 4 — Workpiece Defect (cavity, crack, chipped edge)
    "IMP-01-A01": 4, "IMP-01-A02": 4, "IMP-01-A03": 4, "IMP-01-A04": 4,
}

FAULT_NAMES = {0: "Normal", 1: "Tool Wear", 2: "Chatter",
               3: "Process Anomaly", 4: "Workpiece Defect"}
FAULT_COLORS = {0: "#3498db", 1: "#e74c3c", 2: "#e67e22",
                3: "#9b59b6", 4: "#2ecc71"}
BINARY = {t: int(l > 0) for t, l in FAULT_TYPE.items()}
TRIALS = sorted(FAULT_TYPE.keys())


def load_kit(trial):
    """Carga el hfdata de un experimento y añade columnas de etiqueta.

    Devuelve None si el CSV no existe.
    """
    path = os.path.join(KIT_DIR, f"{trial}_hfdata.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["trial"] = trial
    df["fault_type"] = FAULT_TYPE.get(trial, -1)
    df["anomaly"] = BINARY.get(trial, -1)
    return df


def time_features(vals):
    """10 estadísticos en el dominio temporal sobre un array 1D."""
    vals = vals[~np.isnan(vals)]
    if len(vals) < 10:
        return {k: np.nan for k in ["mean", "std", "rms", "max", "p2p",
                                    "kurt", "skew", "crest", "zcr", "mad"]}
    rms = np.sqrt(np.mean(vals ** 2))
    mx = np.max(np.abs(vals))
    return {
        "mean": np.mean(vals),
        "std": np.std(vals),
        "rms": rms,
        "max": mx,
        "p2p": mx - np.min(np.abs(vals)),
        "kurt": kurtosis(vals),
        "skew": skew(vals),
        "crest": mx / rms if rms > 1e-10 else 0,
        "zcr": np.mean(np.diff(np.sign(vals - vals.mean())) != 0),
        "mad": np.median(np.abs(vals - np.median(vals))),
    }


def freq_features(vals):
    """Energía relativa en 3 bandas espectrales + frecuencia dominante."""
    vals = vals[~np.isnan(vals)]
    if len(vals) < 100:
        return {"e_low": np.nan, "e_mid": np.nan, "e_high": np.nan, "f_dom": np.nan}
    seg = vals - vals.mean()
    yf = np.abs(rfft(seg))
    xf = rfftfreq(len(seg), 1 / FS)
    total = yf.sum() + 1e-10
    return {
        "e_low": yf[(xf < 25)].sum() / total,
        "e_mid": yf[(xf >= 25) & (xf < 100)].sum() / total,
        "e_high": yf[(xf >= 100)].sum() / total,
        "f_dom": xf[np.argmax(yf)],
    }
