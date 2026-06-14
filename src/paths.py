"""
paths.py
────────
Rutas del proyecto en un único sitio, independientes de la profundidad
desde la que se ejecute el notebook o script.

Uso en notebooks:
    import sys; sys.path.insert(0, '../../src')   # '../src' si el notebook está en notebooks/
    from paths import RAW, PROCESSED, MODELS
"""

from pathlib import Path

# Este archivo vive en <proyecto>/src/, así que la raíz es su abuelo.
ROOT = Path(__file__).resolve().parent.parent

RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS    = ROOT / "models"
REPORTS   = ROOT / "reports"
