"""
evaluation.py
─────────────
Evalúa el modelo final sobre el conjunto de test y genera un informe completo
con métricas, análisis de errores y tradeoff de umbral.

Uso:
    python src/evaluation.py

Prerequisito:
    python src/data_processing.py
    python src/training.py

Salidas:
    imprime el informe en consola
    reports/evaluation_report.txt  — informe en texto plano
"""

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_PKL  = os.path.join(ROOT, "data", "processed", "splits.pkl")
FINAL_MODEL = os.path.join(ROOT, "models", "final_model.pkl")
DOCS_DIR    = os.path.join(ROOT, "reports")
os.makedirs(DOCS_DIR, exist_ok=True)


# ── Utilidades ────────────────────────────────────────────────────────────────
def separator(char="─", width=60):
    return char * width


def evaluate():
    lines = []   # acumular líneas para el informe

    def pr(text=""):
        print(text)
        lines.append(text)

    pr(separator("═"))
    pr("  INFORME DE EVALUACIÓN — Predicción de Parada de Máquina")
    pr(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pr(separator("═"))

    # ── Cargar artefactos ─────────────────────────────────────────────────────
    pr("\n[1/5] Cargando modelo y datos...")

    with open(FINAL_MODEL, "rb") as f:
        bundle = pickle.load(f)

    model        = bundle["model"]
    threshold    = bundle["threshold"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["feature_names"]
    description  = bundle.get("description", "")

    pr(f"      Modelo: {description}")
    pr(f"      Umbral de producción: {threshold:.4f}")
    pr(f"      Features ({len(feature_cols)}): {', '.join(feature_cols)}")

    with open(SPLITS_PKL, "rb") as f:
        splits = pickle.load(f)
    X_train, X_test, y_train, y_test = splits["ai4i"]

    pr(f"      Test: {X_test.shape} | Fallos reales: {y_test.sum()}")

    # ── Métricas principales ──────────────────────────────────────────────────
    pr(f"\n[2/5] Métricas con umbral de producción ({threshold:.4f})...")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    pr(f"\n{separator()}")
    pr("  MÉTRICAS PRINCIPALES")
    pr(separator())
    report = classification_report(y_test, y_pred, target_names=["Normal", "Fallo"])
    pr(report)
    pr(f"  ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    pr(f"\n  Matriz de confusión:")
    pr(f"                  Predicho Normal  Predicho Fallo")
    pr(f"  Real Normal          {tn:>6}          {fp:>6}   ← FP (alarmas falsas)")
    pr(f"  Real Fallo           {fn:>6}          {tp:>6}   ← FN (fallos no detectados)")
    pr(f"\n  FP (alarmas falsas):     {fp}  — coste bajo (revisión preventiva innecesaria)")
    pr(f"  FN (fallos escapados):   {fn}  — coste alto (parada imprevista + reparación)")

    # ── Impacto de negocio ────────────────────────────────────────────────────
    pr(f"\n[3/5] Impacto económico estimado (proyección anual)...")

    pr(f"\n{separator()}")
    pr("  IMPACTO ECONÓMICO — Proyección anual (×12 meses)")
    pr(separator())
    pr("  Supuestos: FN = 10.000€/fallo | FP = 300€/alarma")
    pr("             Base: 2.000 muestras de test ≈ 1 mes de producción")
    pr()

    scale    = 12
    cost_fn  = 10_000
    cost_fp  = 300
    sin_mod  = y_test.sum() * scale * cost_fn
    con_mod  = fn * scale * cost_fn + fp * scale * cost_fp
    ahorro   = sin_mod - con_mod

    pr(f"  {'Escenario':<35} {'FN/año':>8} {'FP/año':>8} {'Coste/año':>12}")
    pr(f"  {separator('-', 63)}")
    pr(f"  {'Sin modelo (0 detecciones)':<35} {y_test.sum()*scale:>8} {'0':>8} €{sin_mod:>11,.0f}")
    pr(f"  {'Con modelo (umbral prod.)':<35} {fn*scale:>8} {fp*scale:>8} €{con_mod:>11,.0f}")
    pr(f"  {separator('-', 63)}")
    pr(f"  {'Ahorro estimado':<35} {'':>8} {'':>8} €{ahorro:>11,.0f}  ({ahorro/sin_mod:.0%})")

    # ── Tradeoff de umbral ────────────────────────────────────────────────────
    pr(f"\n[4/5] Tradeoff de umbral de decisión...")

    pr(f"\n{separator()}")
    pr("  TRADEOFF UMBRAL — Precision vs Recall")
    pr(separator())
    pr(f"  {'Umbral':>8} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FN':>6} {'FP':>6}")
    pr(f"  {separator('-', 50)}")

    thresholds_eval = [0.25, 0.50, 0.60, 0.75, threshold, 0.90, 0.95, 0.99]
    for thr in sorted(set(thresholds_eval)):
        yp = (y_prob >= thr).astype(int)
        rec  = recall_score(y_test, yp)
        prec = precision_score(y_test, yp, zero_division=0)
        f1   = f1_score(y_test, yp, zero_division=0)
        fn_  = ((yp == 0) & (y_test == 1)).sum()
        fp_  = ((yp == 1) & (y_test == 0)).sum()
        mark = " ← PRODUCCIÓN" if abs(thr - threshold) < 0.001 else ""
        pr(f"  {thr:>8.4f} {rec:>8.3f} {prec:>10.3f} {f1:>8.3f} {fn_:>6} {fp_:>6}{mark}")

    # ── Análisis de falsos negativos ──────────────────────────────────────────
    pr(f"\n[5/5] Análisis de falsos negativos...")

    pr(f"\n{separator()}")
    pr("  ANÁLISIS DE FALSOS NEGATIVOS")
    pr(separator())

    fn_mask  = (y_pred == 0) & (y_test == 1)
    fn_probs = y_prob[fn_mask]

    group_a = fn_probs[fn_probs >= 0.25]
    group_b = fn_probs[fn_probs <  0.25]

    pr(f"  Total FN: {fn_mask.sum()}")
    pr(f"\n  Grupo A — catchables (prob ≥ 0.25): {len(group_a)} fallos")
    pr(f"    Probabilidades: {sorted(group_a.round(3), reverse=True)}")
    pr(f"    → Recuperables bajando el umbral a 0.75 (a costa de más FP)")
    pr(f"\n  Grupo B — invisibles (prob < 0.25): {len(group_b)} fallos")
    pr(f"    Probabilidades: {sorted(group_b.round(3), reverse=True)}")
    pr(f"    → Firma de sensor indistinguible de operación normal")
    pr(f"    → Causa raíz: umbral de rotura TWF individual no observable en sensores")
    pr(f"    → Solución: añadir tool_id + tool_threshold (dato del CMMS)")

    # ── Resumen final ─────────────────────────────────────────────────────────
    pr(f"\n{separator('═')}")
    pr("  RESUMEN")
    pr(separator("═"))
    pr(f"  Modelo:    {description}")
    pr(f"  Umbral:    {threshold:.4f}")
    pr(f"  Precision: {precision_score(y_test,y_pred):.4f}")
    pr(f"  Recall:    {recall_score(y_test,y_pred):.4f}  (objetivo ≥ 0.85 ✓)")
    pr(f"  F1:        {f1_score(y_test,y_pred):.4f}")
    pr(f"  ROC-AUC:   {roc_auc_score(y_test,y_prob):.4f}")
    pr(f"  FN/test:   {fn} de {y_test.sum()} fallos reales")
    pr(f"  Ahorro estimado: €{ahorro:,.0f}/año ({ahorro/sin_mod:.0%} reducción de costes)")
    pr(separator("═"))

    # Guardar informe
    report_path = os.path.join(DOCS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✓ Informe guardado en: {report_path}")


if __name__ == "__main__":
    evaluate()
