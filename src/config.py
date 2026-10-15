"""
config.py

Umbrales y límites centrales del análisis.

Todas las constantes que controlan la sensibilidad del pipeline
viven aquí para que sean fáciles de localizar y ajustar.
Los módulos las importan; no deben redefinirlas.
"""

from __future__ import annotations

# --- Limpieza y calidad -----------------------------------------------------

# Multiplicador del rango intercuartílico para detectar outliers.
OUTLIER_IQR_MULTIPLIER = 1.5

# Multiplicador del rango intercuartílico para valores extremos.
EXTREME_IQR_MULTIPLIER = 3.0

# Fracción mínima de valores convertibles a numérico para considerar
# que una columna "parece" numérica.
NUMERIC_CONVERSION_THRESHOLD = 0.90

# --- Selección de relaciones ------------------------------------------------

# Percentil de las puntuaciones por encima del cual una relación
# se considera relevante para el dataset.
RELATIONSHIP_SELECTION_PERCENTILE = 75.0

# Fuerza mínima absoluta de una asociación (Cramér's V).
MIN_CRAMERS_V = 0.3

# Fuerza mínima absoluta de una correlación (Pearson/Spearman).
MIN_CORRELATION_STRENGTH = 0.5

# Fuerza mínima absoluta de separación de grupos (cat↔num).
MIN_GROUP_SEPARATION = 0.25

# Correlación por encima de la cual un par numérico siempre se
# selecciona, sin importar el percentil del dataset (p.ej. price~carat
# en diamonds: 0.963 se perdía contra un p75 inflado por pares
# geométricos).
RELATIONSHIP_STRONG_FLOOR = 0.9

# Cada columna hueca (sin ningún par seleccionado por el percentil)
# conserva su mejor par como mínimo, aunque quede por debajo del
# percentil; nunca por debajo del piso absoluto de su familia
# (MIN_CORRELATION_STRENGTH / MIN_CRAMERS_V / MIN_GROUP_SEPARATION).
RELATIONSHIP_PER_COLUMN_TOP = 1

# En cat↔num top-1 no alcanza: los mejores partners moderados de una
# columna suelen ser su 2do/3ro (en diamonds el de price es color
# 0.56 y cut 0.28; en adult el de income es hours 0.53). Cada columna
# hueca (sin ningún par en el percentil) conserva sus mejores pares
# mientras superen MIN_GROUP_SEPARATION.
GROUP_SEPARATION_PER_COLUMN_TOP = 3

# --- Visualizaciones --------------------------------------------------------

# Scatters sin muestreo para muestras pequeñas.
SCATTER_NO_SAMPLING_MAX = 1000

# Scatters con muestreo ligero a partir de este tamaño de muestra.
SCATTER_LIGHT_SAMPLING_MAX = 5000

# Puntos del muestreo ligero de scatters.
SCATTER_LIGHT_SAMPLING_POINTS = 1500

# Puntos máximo de un scatter (muestreo pesado).
SCATTER_MAX_POINTS = 2000

# Categorías máximas de un heatmap de contingencia.
HEATMAP_MAX_CATEGORIES = 15

# Categorías máximas de una barra ordenada.
BAR_MAX_CATEGORIES = 10

# Puntos máximos enviados para un boxplot.
BOXPLOT_MAX_POINTS = 5000

# Gráficos construidos on-demand para insights cuyo chart_id apunta a
# una relación que no pasó los umbrales de selección (el insight nulo
# necesita su gráfico aunque la relación haya sido descartada).
INSIGHT_CHARTS_MAX_EXTRA = 12

# --- Insights ---------------------------------------------------------------

# Cantidad de hallazgos del resumen ejecutivo.
EXECUTIVE_SUMMARY_TOP_N = 8

# Máximo de warnings de calidad en el resumen ejecutivo (el resto
# se completa con el mejor score).
EXECUTIVE_SUMMARY_MAX_WARNINGS = 2

# --- Insights comparativos --------------------------------------------------

# Gap mínimo top/segundo para emitir un ranking de grupo (total).
RANK_TOP_GAP = 1.3

# Cobertura mínima del par (grupo, valor) para poder afirmar
# "% of all X"; por debajo el share es de subconjunto y se califica.
COVERAGE_FULL_PCT = 99.9

# En columnas boolean, share mínimo del valor positivo (1/true)
# para emitir una alerta de dominancia; la dominancia del valor
# negativo (estado esperado) nunca se emite.
BOOLEAN_DOMINANT_ALERT_SHARE = 80.0

# La dispersión de medias debe superar esta razón respecto de la
# dispersión de conteos para declarar un driver "valor sobre volumen".
DRIVER_SPREAD_RATIO = 2.0

# La dispersión relativa de conteos debe ser menor o igual a esto
# para que "volumen similar entre grupos" sea una premisa válida.
DRIVER_VOLUME_MAX = 0.3

# Gap mínimo del segundo grupo para emitir consistencia entre métricas.
CONSISTENCY_GAP = 1.5

# Cuota máxima del valor dominante para considerar una distribución
# balanceada (y la cantidad mínima de valores).
BALANCED_MAX_SHARE = 35.0
BALANCED_MIN_VALUES = 3
BALANCED_MAX_VALUES = 8

# Relación max/min entre cuotas para considerar la distribución
# razonablemente pareja.
BALANCED_SKEW = 1.6

# --- Hallazgos nulos --------------------------------------------------------

# Separación de grupos máxima (en desviaciones estándar del overall)
# para declarar que un grupo binario prácticamente no diferencia:
# la diferencia de medias sola es engañosa con columnas de alta varianza.
NULL_CATNUM_MAX_SEPARATION = 0.10

# Correlación absoluta máxima para declarar "sin asociación" num↔num.
NULL_CORR_MAX = 0.2

# Cramér's V máximo y p-value mínimo para declarar independencia
# cat↔cat (p por encima de este valor = no se rechaza independencia).
NULL_V_MAX = 0.15
NULL_P_MIN = 0.05

# Observaciones mínimas para emitir un hallazgo nulo.
NULL_MIN_OBSERVATIONS = 30

# --- Caps por familia de insight comparativo --------------------------------

COMPARATIVE_RANK_MAX = 5
COMPARATIVE_DRIVER_MAX = 5
COMPARATIVE_CONSISTENCY_MAX = 3
COMPARATIVE_BALANCED_MAX = 3
COMPARATIVE_NULL_MAX = 5
