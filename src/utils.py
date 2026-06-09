"""
utils.py
────────
Constantes y funciones compartidas por todos los notebooks del proyecto.

Uso en notebooks:
    import sys; sys.path.insert(0, '../src')
    from utils import eval_threshold, load_ai4i, get_splits, FEATURE_COLS
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Constantes ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Type_enc",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Power [W]",
    "Temp_diff [K]",
    "Wear_torque",
    "wear_ratio",
]

FEAT_NAMES = [
    "Type", "Air Temp", "Process Temp", "Rot Speed", "Torque",
    "Tool Wear", "Power", "Temp Diff", "Wear×Torque", "wear_ratio",
]

TWF_MIN = {"L": 200, "M": 220, "H": 240}


# ── Funciones ─────────────────────────────────────────────────────────────────

def eval_threshold(y_true, y_prob, min_recall=0.85):
    """
    Encuentra el umbral que maximiza F1 con Recall >= min_recall.
    Devuelve dict con Precision, Recall, F1, ROC-AUC y Umbral.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1   = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    mask = rec[:-1] >= min_recall
    idx  = np.argmax(f1 * mask) if mask.any() else np.argmax(f1)
    thr_opt = thr[idx]
    y_pred  = (y_prob >= thr_opt).astype(int)
    return {
        "Precision": round(float(precision_score(y_true, y_pred)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred)), 4),
        "F1":        round(float(f1_score(y_true, y_pred)), 4),
        "ROC-AUC":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "Umbral":    round(float(thr_opt), 4),
    }


def load_ai4i(raw_path="../data/raw/ai4i2020.csv"):
    """
    Carga AI4I y aplica el pipeline completo de feature engineering.
    Devuelve el DataFrame con las columnas originales + features nuevas.
    """
    df = pd.read_csv(raw_path)
    df["Type_enc"]      = LabelEncoder().fit_transform(df["Type"])
    df["Power [W]"]     = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Wear_torque"]   = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["wear_ratio"]    = df.apply(lambda r: r["Tool wear [min]"] / TWF_MIN[r["Type"]], axis=1)
    return df


def get_splits(df, test_size=0.2, random_state=42):
    """
    Aplica StandardScaler y split estratificado sobre un DataFrame de AI4I
    con features de ingeniería ya calculadas.

    Devuelve: X_train, X_test, y_train, y_test, idx_train, idx_test, scaler
    """
    X = df[FEATURE_COLS].values
    y = df["Machine failure"].values
    idx = np.arange(len(df))

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(
        X_scaled, y, idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, i_train, i_test, scaler
