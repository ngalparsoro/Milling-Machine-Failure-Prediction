"""
spark.py
────────
Constantes y carga del dataset SPARK TEC (monitorización eléctrica,
11 máquinas-herramienta, año 2024).

Única fuente de verdad para MACHINES / VARS / FEATURE_COLS, que antes
estaban copiadas en 5 notebooks.

Uso en notebooks:
    import sys; sys.path.insert(0, '../../src')
    from spark import MACHINES, VARS, FEATURE_COLS, GAP_W, load_machine
"""

import os

import pandas as pd

from paths import RAW

RAW_TEC = RAW / "tec_extracted"

MACHINES = sorted([
    "TEC_48S", "TEC_CFST161", "TEC_CTX800TC", "TEC_Chiron800",
    "TEC_DMF3008", "TEC_DMU125MB", "TEC_DNG50evo", "TEC_E110",
    "TEC_E30D2", "TEC_JWA24", "TEC_MV2400R",
])

VARS = ["P_total", "P1", "P2", "P3", "I1", "I2", "I3",
        "Freq", "THD_I1", "THD_I2", "THD_I3", "PF_total"]

FEATURE_COLS = [f"{v}_{s}" for v in VARS for s in ["mean", "std", "max"]]

GAP_W = 50  # W — por debajo se considera máquina parada (gap)


def load_machine(machine, raw_tec=None, vars_list=None):
    """Carga y agrega a 1 min los CSVs de una máquina desde tec_extracted/."""
    raw_tec = RAW_TEC if raw_tec is None else raw_tec
    vars_list = VARS if vars_list is None else vars_list

    dfs = []
    for var in vars_list:
        base = os.path.join(raw_tec, machine, var, f"2024_{var}")
        if os.path.exists(base + ".csv"):
            csv_path = base + ".csv"
        elif os.path.exists(base + ".csv.xz"):
            csv_path = base + ".csv.xz"
        else:
            continue
        df_var = pd.read_csv(csv_path, parse_dates=["WsDateTime"],
                             index_col="WsDateTime", low_memory=False)
        df_var.columns = [var]
        dfs.append(df_var)
    if not dfs:
        return None

    raw = pd.concat(dfs, axis=1).sort_index()
    agg = {}
    for var in [c for c in vars_list if c in raw.columns]:
        agg[f"{var}_mean"] = pd.NamedAgg(column=var, aggfunc="mean")
        agg[f"{var}_std"] = pd.NamedAgg(column=var, aggfunc="std")
        agg[f"{var}_max"] = pd.NamedAgg(column=var, aggfunc="max")
    df_1min = raw.resample("1min").agg(**agg)
    df_1min["machine"] = machine
    return df_1min
