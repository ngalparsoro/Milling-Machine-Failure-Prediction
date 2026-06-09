"""
data_processing.py
──────────────────
Lee los datos brutos (AI4I + CNC), aplica el pipeline de preprocesado completo
y guarda los artefactos procesados en data/processed/.

Uso:
    python src/data_processing.py

Salidas:
    data/processed/ai4i_featured.csv   — AI4I con features de ingeniería
    data/processed/cnc_aggregated.csv  — CNC agregado por experimento
    data/processed/splits.pkl          — splits reproducibles + scaler
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_AI4I     = os.path.join(ROOT, "data", "raw", "ai4i2020.csv")
RAW_CNC_DIR  = os.path.join(ROOT, "data", "raw", "cnc_extracted")
OUT_DIR      = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Configuración ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
TWF_MIN      = {"L": 200, "M": 220, "H": 240}   # umbrales de desgaste por tipo

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


# ── 1. AI4I ───────────────────────────────────────────────────────────────────
def process_ai4i():
    print("[AI4I] Cargando datos brutos...")
    df = pd.read_csv(RAW_AI4I)
    print(f"       Shape: {df.shape} | Nulos: {df.isnull().sum().sum()}")

    # Encoding categórico
    df["Type_enc"] = LabelEncoder().fit_transform(df["Type"])

    # Feature engineering (cada feature captura la condición física de un modo de fallo)
    df["Power [W]"]     = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    df["Temp_diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Wear_torque"]   = df["Tool wear [min]"] * df["Torque [Nm]"]
    df["wear_ratio"]    = df.apply(
        lambda r: r["Tool wear [min]"] / TWF_MIN[r["Type"]], axis=1
    )

    X = df[FEATURE_COLS].values
    y = df["Machine failure"].values

    # Escalado — necesario para SVM y para el meta-modelo del stacking
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split estratificado — garantiza misma tasa de fallo en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"       Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"       Failure rate — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    # Guardar CSV con features
    df.to_csv(os.path.join(OUT_DIR, "ai4i_featured.csv"), index=False)
    print("       ai4i_featured.csv guardado.")

    return X_train, X_test, y_train, y_test, scaler


# ── 2. CNC Mill ───────────────────────────────────────────────────────────────
def process_cnc():
    print("[CNC]  Cargando datos brutos...")
    train_meta = pd.read_csv(os.path.join(RAW_CNC_DIR, "train.csv")).rename(
        columns={"No": "experiment"}
    )

    dfs = []
    for i in range(1, 19):
        path = os.path.join(RAW_CNC_DIR, f"experiment_{i:02d}.csv")
        df_exp = pd.read_csv(path)
        df_exp["experiment"] = i
        dfs.append(df_exp)
    cnc_raw = pd.concat(dfs, ignore_index=True)
    cnc_raw = cnc_raw.merge(train_meta, on="experiment", how="left")
    print(f"       Shape raw: {cnc_raw.shape}")

    # Agregar por experimento: 18 filas × estadísticos por señal
    exclude = {
        "experiment", "Machining_Process", "M1_CURRENT_PROGRAM_NUMBER",
        "M1_sequence_number", "material", "feedrate", "clamp_pressure",
        "tool_condition", "machining_finalized", "passed_visual_inspection",
    }
    sensor_cols = [c for c in cnc_raw.columns if c not in exclude]
    agg_dict = {c: ["mean", "std", "max", "min"] for c in sensor_cols}
    cnc_agg = cnc_raw.groupby("experiment").agg(agg_dict)
    cnc_agg.columns = ["_".join(c) for c in cnc_agg.columns]
    cnc_agg = cnc_agg.reset_index()
    cnc_agg = cnc_agg.merge(
        train_meta[["experiment", "material", "feedrate", "clamp_pressure", "tool_condition"]],
        on="experiment",
    )
    cnc_agg["material_enc"] = LabelEncoder().fit_transform(cnc_agg["material"])
    cnc_agg["target"] = (cnc_agg["tool_condition"] == "worn").astype(int)
    print(f"       Agregado: {cnc_agg.shape} | worn: {cnc_agg['target'].sum()} | unworn: {(cnc_agg['target']==0).sum()}")

    # Features y split CNC
    feat_cnc = [c for c in cnc_agg.columns if c not in
                ["experiment", "material", "tool_condition", "target"]]
    X_cnc = cnc_agg[feat_cnc].fillna(0).values
    y_cnc = cnc_agg["target"].values

    scaler_cnc = StandardScaler()
    X_cnc_scaled = scaler_cnc.fit_transform(X_cnc)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cnc_scaled, y_cnc, test_size=0.2, random_state=RANDOM_STATE
    )

    cnc_agg.to_csv(os.path.join(OUT_DIR, "cnc_aggregated.csv"), index=False)
    print("       cnc_aggregated.csv guardado.")

    return X_train_c, X_test_c, y_train_c, y_test_c, X_cnc_scaled, y_cnc


# ── 3. Guardar splits ─────────────────────────────────────────────────────────
def save_splits(ai4i_splits, cnc_splits, scaler):
    X_train, X_test, y_train, y_test = ai4i_splits
    X_train_c, X_test_c, y_train_c, y_test_c, X_cnc_scaled, y_cnc = cnc_splits

    splits = {
        "ai4i":    (X_train, X_test, y_train, y_test),
        "cnc":     (X_train_c, X_test_c, y_train_c, y_test_c),
        "cnc_loo": (X_cnc_scaled, y_cnc),
        "scaler":  scaler,
        "feature_cols": FEATURE_COLS,
    }
    path = os.path.join(OUT_DIR, "splits.pkl")
    with open(path, "wb") as f:
        pickle.dump(splits, f)
    print(f"[OK]   splits.pkl guardado en {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  data_processing.py — Preprocesado AI4I + CNC")
    print("=" * 55)

    X_train, X_test, y_train, y_test, scaler = process_ai4i()
    cnc_splits = process_cnc()
    save_splits((X_train, X_test, y_train, y_test), cnc_splits, scaler)

    print("\n✓ Preprocesado completado.")
    print(f"  Artefactos guardados en: {OUT_DIR}")
