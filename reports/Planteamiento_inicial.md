# Predicción de Parada de Máquina — Mantenimiento Predictivo

## Descripción del problema

Las averías inesperadas en entornos industriales generan costosas paradas de producción. Este proyecto aborda ese problema mediante **mantenimiento predictivo**: anticipar el fallo de una máquina antes de que ocurra, usando datos de sensores recogidos durante el proceso de fabricación.

El objetivo es entrenar un clasificador binario que prediga si una máquina va a fallar (`Machine failure = 1`) en función de variables operacionales como temperatura, velocidad de rotación, par motor y desgaste de herramienta.

## Datos

Se utilizan dos datasets complementarios. El foco principal del modelado es **AI4I**, que ofrece variables de proceso bien definidas y una etiqueta de fallo clara. El dataset CNC se usa como fuente secundaria para enriquecer el análisis de desgaste de herramienta.

---

### Dataset 1 — AI4I 2020 Predictive Maintenance

**Fuente:** UCI Machine Learning Repository (CC BY 4.0)  
**Registros:** 10 000 · **Variables:** 14 · **Sin valores nulos ni duplicados**

Datos sintéticos generados a partir de un proceso de fabricación real. Cada registro representa una operación de mecanizado con sus condiciones de proceso y si se produjo un fallo.

**Variables de entrada:**

| Variable | Tipo | Rango | Descripción |
|---|---|---|---|
| `Type` | Categórica | L / M / H | Calidad del producto: Low (60%), Medium (30%), High (10%) |
| `Air temperature [K]` | Numérica | 295–305 K | Temperatura del aire (media 300 K, σ = 2 K) |
| `Process temperature [K]` | Numérica | 306–314 K | Temperatura del proceso (≈ aire + 10 K, σ = 1,5 K) |
| `Rotational speed [rpm]` | Numérica | 1168–2886 rpm | Velocidad de rotación (media 1539 rpm) |
| `Torque [Nm]` | Numérica | 3,8–76,6 Nm | Par motor (media 40 Nm, σ = 10 Nm) |
| `Tool wear [min]` | Numérica | 0–253 min | Minutos acumulados de uso de la herramienta |

**Variables objetivo:**

| Variable | Descripción |
|---|---|
| `Machine failure` | Etiqueta binaria: 1 si hay fallo por cualquier causa (339 casos, **3,4 %**) |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Indicadores individuales de cada modo de fallo |

**Distribución de fallos por tipo de producto:** L → 3,9 % · M → 2,8 % · H → 2,1 %

---

### Dataset 2 — CNC Mill Tool Wear

**Fuente:** Kaggle  
**Registros:** 5 287 · **Variables:** 27 · **Sin valores nulos**

Señales recogidas en tiempo real de una fresadora CNC durante 18 experimentos con distintas combinaciones de material, velocidad de avance y presión de sujeción.

**Variables de configuración del experimento:**

| Variable | Valores | Descripción |
|---|---|---|
| `Experiment` | 1–18 | Identificador del experimento |
| `Material` | cast iron / wax | Material mecanizado |
| `Feed rate [mm/rev]` | 0,5 / 1,0 / 2,0 | Velocidad de avance de la herramienta |
| `Clamp pressure [bar]` | 1 / 4 / 6 | Presión de sujeción de la pieza |
| `Machining_Process` | Prep, Layer 1/2 Up/Down, end | Fase del proceso de mecanizado |

**Señales de sensores** (agrupadas por eje del CNC):

| Eje | Señales disponibles |
|---|---|
| **X** | Posición, velocidad y aceleración (actual y comando), corriente, tensión DC, corriente y tensión de salida |
| **Y** | Posición, velocidad, corriente |
| **Z** | Posición, velocidad, corriente |
| **S** (husillo) | Posición, velocidad, aceleración, corriente, tensión de salida |

**Variable objetivo:**

| Variable | Distribución | Descripción |
|---|---|---|
| `Tool_condition` | unworn (66%) / worn (34%) | Estado de la herramienta al final del experimento |

---

### Modos de fallo (AI4I)

El dataset registra cinco causas de fallo independientes, todas ellas determinadas por reglas físicas sobre las variables del proceso:

| Código | Nombre | Condición de activación |
|---|---|---|
| **TWF** | Tool Wear Failure — Fallo por desgaste de herramienta | La herramienta se reemplaza o falla al alcanzar un umbral de desgaste entre 200 y 240 min (asignado aleatoriamente por tipo de producto) |
| **HDF** | Heat Dissipation Failure — Fallo por disipación de calor | La diferencia entre temperatura de proceso y temperatura del aire es < 8,6 K y la velocidad de rotación es < 1380 rpm |
| **PWF** | Power Failure — Fallo por potencia | La potencia (par × velocidad angular) cae por debajo de 3500 W o supera los 9000 W |
| **OSF** | Overstrain Failure — Fallo por sobreesfuerzo | El producto de desgaste de herramienta y par motor supera un umbral según el tipo de producto (L: 11 000, M: 12 000, H: 13 000 Nm·min) |
| **RNF** | Random Failure — Fallo aleatorio | Ocurre con una probabilidad fija del 0,1 %, independientemente del proceso |

Si alguno de estos modos se activa, la columna `Machine failure` toma valor 1. OSF y HDF son los más frecuentes en el dataset.

## Enfoque de modelado

El problema presenta **fuerte desbalanceo de clases** (≈3 % de fallos en AI4I). Las decisiones de diseño priorizaran **Recall** como métrica principal: en un contexto de mantenimiento industrial, un falso negativo (fallo no detectado) es mucho más costoso que un falso positivo.

