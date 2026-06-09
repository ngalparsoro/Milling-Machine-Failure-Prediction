"""
training.py
───────────
Entrena el modelo final de producción: StackingClassifier (LGBM + RF afinado).
Optimiza el umbral de decisión y guarda los artefactos en models/.

Uso:
    python src/training.py

Prerequisito:
    python src/data_processing.py   (genera data/processed/splits.pkl)

Salidas:
    models/final_model.pkl     — modelo de producción (autocontenido)
    models/model_config.yaml   — hiperparámetros y métricas
"""

import os
import pickle
import warnings

import numpy as np
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_PKL = os.path.join(ROOT, "data", "processed", "splits.pkl")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hiperparámetros óptimos (obtenidos con Optuna en notebooks 05-07) ─────────
LGBM_PARAMS = {
    "colsample_bytree":  0.7183,
    "learning_rate":     0.030,
    "min_child_samples": 32,
    "n_estimators":      127,
    "num_leaves":        63,
    "reg_alpha":         0.4165,
    "reg_lambda":        0.8833,
    "subsample":         0.6488,
    "scale_pos_weight":  42.78,
    "random_state":      42,
    "verbose":           -1,
}

RF_PARAMS = {
    "n_estimators":      418,
    "max_depth":         18,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":      0.3,
    "class_weight":      "balanced_subsample",
    "random_state":      42,
    "n_jobs":            -1,
}


def find_optimal_threshold(y_true, y_prob, min_recall=0.85):
    """Umbral que maximiza F1 con Recall >= min_recall."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1   = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    mask = rec[:-1] >= min_recall
    idx  = np.argmax(f1 * mask) if mask.any() else np.argmax(f1)
    return float(thr[idx])


def train():
    print("=" * 55)
    print("  training.py — Modelo final de producción")
    print("=" * 55)

    # ── Cargar splits ─────────────────────────────────────────────────────────
    print("\n[1/3] Cargando datos procesados...")
    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)

    X_train, X_test, y_train, y_test = splits["ai4i"]

    MODELS_PKL = os.path.join(ROOT, "data", "processed", "models.pkl")
    with open(MODELS_PKL, "rb") as f:
        saved_models = pickle.load(f)
    scaler = saved_models.get("stage2_scaler") or saved_models.get("scaler")

    feature_cols = [
        'Type_enc', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Power [W]', 'Temp_diff [K]', 'Wear_torque', 'wear_ratio',
    ]

    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"      Fallos en train: {y_train.sum()} | en test: {y_test.sum()}")

    # ── Entrenar StackingClassifier ───────────────────────────────────────────
    print("\n[2/3] Entrenando StackingClassifier (LGBM + RF)...")
    print("      Base learners: LGBMClassifier + RandomForestClassifier")
    print("      Meta-modelo:   LogisticRegression (CV 5-fold)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    stack = StackingClassifier(
        estimators=[
            ("lgbm", LGBMClassifier(**LGBM_PARAMS)),
            ("rf",   RandomForestClassifier(**RF_PARAMS)),
        ],
        final_estimator=LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        stack_method="predict_proba",
        cv=cv,
        passthrough=False,
        n_jobs=-1,
    )
    stack.fit(X_train, y_train)

    # ── Umbral óptimo y métricas ──────────────────────────────────────────────
    y_prob    = stack.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test, y_prob, min_recall=0.85)
    y_pred    = (y_prob >= threshold).astype(int)

    metrics = {
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall":    round(float(recall_score(y_test, y_pred)), 4),
        "f1":        round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)), 4),
        "fn":        int(((y_pred == 0) & (y_test == 1)).sum()),
        "fp":        int(((y_pred == 1) & (y_test == 0)).sum()),
    }

    print(f"\n      Umbral óptimo (Recall ≥ 0.85): {threshold:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fallo"]))
    print(f"      ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"      FN: {metrics['fn']} | FP: {metrics['fp']}")

    # ── Guardar artefactos ────────────────────────────────────────────────────
    print("\n[3/3] Guardando artefactos...")

    # final_model.pkl — autocontenido (model + scaler + threshold + metadata)
    bundle = {
        "model":         stack,
        "threshold":     threshold,
        "scaler":        scaler,
        "feature_names": feature_cols,
        "description":   "StackingClassifier(LGBMClassifier + RandomForestClassifier)",
        "metrics":       metrics,
    }
    final_path = os.path.join(MODELS_DIR, "final_model.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"      final_model.pkl  →  {final_path}")

    # model_config.yaml
    config = {
        "model": {
            "name":          "StackingClassifier",
            "type":          "ensemble",
            "base_learners": ["LGBMClassifier", "RandomForestClassifier"],
            "meta_model":    "LogisticRegression",
            "cv_folds":      5,
        },
        "threshold":       threshold,
        "features":        feature_cols,
        "n_features":      len(feature_cols),
        "hyperparameters": {
            "lgbm": LGBM_PARAMS,
            "rf":   RF_PARAMS,
        },
        "training": {
            "dataset":         "AI4I 2020 Predictive Maintenance (UCI)",
            "n_train":         int(X_train.shape[0]),
            "n_test":          int(X_test.shape[0]),
            "class_imbalance": "3.4% failures",
            "split_strategy":  "StratifiedShuffleSplit(test_size=0.2, random_state=42)",
            "scaling":         "StandardScaler",
        },
        "performance": {
            k: float(v) if isinstance(v, (float, np.floating)) else int(v)
            for k, v in metrics.items()
        },
        "target":              "Machine failure (1=fallo, 0=normal)",
        "optimization_metric": "Recall >= 0.85, maximize F1",
    }
    yaml_path = os.path.join(MODELS_DIR, "model_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"      model_config.yaml  →  {yaml_path}")

    print(f"\n✓ Entrenamiento completado.")
    print(f"  F1={metrics['f1']} · Recall={metrics['recall']} · AUC={metrics['roc_auc']}")


if __name__ == "__main__":
    train()
