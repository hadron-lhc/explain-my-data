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

# --- Equivalencia y columnas derivadas (P1) ----------------------------------

# Cramér's V a partir del cual dos columnas categóricas se consideran
# la misma información (renombre puro, p.ej. embarked/embark_town).
EQUIVALENCE_MIN_CRAMERS_V = 0.98

# Pearson absoluto a partir del cual dos numéricas se consideran la
# misma información (variables geométricamente proporcionales).
EQUIVALENCE_MIN_CORR = 0.995

# Cardinalidad máxima de una categórica para considerarla derivada
# (una columna sintética no supera este número de valores).
DERIVED_MAX_CARDINALITY = 10

# --- Trivialidades de distribución (P2) --------------------------------------

# Máximo de valores únicos de una numérica discreta para considerarla
# conteo. Por encima (p.ej. fare, 248 valores) es continua y sus
# insights de forma siguen siendo informativos.
COUNT_MAX_UNIQUE = 30

# Skew absoluto mínimo para que una variable de conteo emita su
# asimetría. Un conteo con |skew| por debajo casi nunca es informativo
# (la asimetría positiva es la forma esperada de un conteo); por
# encima es tan extremo que cuenta algo (p.ej. una cola larga real).
COUNT_SKIP_SKEW = 5.0

# Cuota del 5% superior mínima para que una variable de conteo emita
# su concentración. Un conteo siempre concentra algo en la cola
# superior; solo interesa si lo hace de forma desproporcionada.
COUNT_SKIP_TOP_PCT = 80.0

# Máximo de valores únicos de una numérica discreta para suprimir el
# insight de modalidad. En 3 valores posibles, 3 modas es ruido de
# histograma, no subpoblaciones mezcladas.
DISCRETE_SKIP_MODES_MAX = 5

# --- Validación contextual (P3) ---------------------------------------------

# Máximo de % de missing en una columna para que sus relaciones se
# emitan con normalidad. Por encima, la tarjeta se califica con un
# caveat dominante; por debajo del doble, se emite en una sección de
# baja confianza; por encima del doble, no se emite.
RELATIONSHIP_MAX_MISSING_PCT = 30.0

# Tamaño mínimo por grupo tras dropna para que un grupo sostenga una
# afirmación comparativa.
RELATIONSHIP_MIN_GROUP_N = 30

# Dispersión relativa mínima entre medias de grupo para que una
# diferencia sea práctica (no solo estadística). Si la dispersión es
# menor, la separación en desviaciones estándar puede ser artefacto
# de poca varianza global.
MIN_RELATIVE_DIFF = 0.15

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

# Presupuesto global de insights por dataset (P4). Es una GUARD patológica:
# con los quotas por familia de abajo ningún dataset del corpus QA lo
# alcanza. El corpus prueba que los `need`/`claims` verificados viven
# diseminidos por las familias con scores bajos (nulls comparativos,
# patrones de grupos pequeños, asociaciones cat-cat con V<0.5): un
# recorte global agresivo por score los elimina a todos. El tope real
# de UX lo fija el quota por familia.
INSIGHT_BUDGET_TOTAL = 70

# Presupuesto por familia (P4). Cotas derivadas de la línea base del
# corpus QA: cada familia debe conservar al menos la posición de score
# de su `need` más bajo verificado (p.ej. adult: comparativo null en
# posición 9, patrón income>50K en posición 7, asociación
# marital~income con V=0.45 en la posición ~9 de relationship).
# Corte por score dentro de la familia; nunca por chart compartido.
INSIGHT_BUDGET_BY_FAMILY = {
    "quality_insights": 8,
    "relationship_insights": 12,
    "inference_insights": 10,
    "distribution_insights": 6,
    "pattern_insights": 10,
    "structure_insights": 4,
    "temporal_insights": 8,
    "ml_insights": 2,
    "comparative_insights": 12,
    "aggregation_insights": 6,
}

# Tamaño del resumen ejecutivo narrativo (P4): cuántas frases como
# máximo. Cada frase es un insight (category="executive_summary") que
# referencia su tarjeta fuente.
EXECUTIVE_SUMMARY_TOP_N = 8

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

# Caps por métrica (columna numérica) dentro de cada subfamilia
# comparativa (P4). Sin ellos, una métrica concentrada (p.ej.
# capital_gain en adult) genera un ranking/driver/consistencia por
# cada columna categórica del dataset —4-5 tarjetas diciendo quién
# lidera la misma métrica— y ese ruido crowdseda a los nulls
# verificados del corpus QA dentro del quota por familia.
COMPARATIVE_RANK_PER_METRIC_MAX = 2
COMPARATIVE_DRIVER_PER_METRIC_MAX = 1
COMPARATIVE_CONSISTENCY_PER_METRIC_MAX = 1

# --- Curación de insights (un hallazgo = una tarjeta) ------------------------

# Máximo de patrones "stands out" por par (grupo, numérico): los grupos
# intermedios de un mismo par añaden página sin añadir hallazgo; quedan
# los extremos más estandarizados.
PATTERN_PER_PAIR_CAP = 2

# --- Análisis temporal (F3) --------------------------------------------------

# Columnas datetime analizadas por dataset y métricas por fecha.
TEMPORAL_MAX_DATE_COLUMNS = 2
TEMPORAL_MAX_METRICS_PER_DATE = 3

# Periodos mínimos (resampleados) para emitir cualquier hallazgo.
TEMPORAL_MIN_PERIODS = 8

# Selección de pares: score máximo entre |spearman| (tendencia) y
# autocorrelación en el lag natural (estacionalidad); por debajo del
# piso la métrica no se empareja con la fecha.
TEMPORAL_PAIR_MIN_SCORE = 0.4

# Cambio total mínimo (%) para declarar tendencia.
TEMPORAL_TREND_MIN_PCT = 5.0

# Autocorrelación mínima (serie detrendeada) para declarar
# estacionalidad en el lag natural (día→7, semana→52, mes→12, trimestre→4).
TEMPORAL_SEASONAL_MIN_AUTOCORR = 0.5

# Separación mínima (en desviaciones del overall) para declarar un
# cambio de régimen.
TEMPORAL_REGIME_MIN_SCORE = 0.8

# Multiplicador de IQR para extremos sobre medias de período.
TEMPORAL_OUTLIER_IQR_MULT = 3.0

# --- Inferencia estadística (F4) ---------------------------------------------

# Observaciones mínimas y grupos mínimos para emitir un test cat-num.
INFERENCE_MIN_OBSERVATIONS = 30
INFERENCE_MIN_GROUPS = 2

# Umbral de significancia (raw y BH) y de n chico para caveat.
INFERENCE_ALPHA = 0.05
INFERENCE_SMALL_SAMPLE = 50

# Corrección Benjamini-Hochberg sobre la familia de tests del dataset.
INFERENCE_BH_ENABLED = True

# Tamaños de efecto: Cohen's d para top-vs-rest y η² global; por
# debajo de "prácticamente nulo" el hallazgo se califica como
# "estadísticamente significativo pero efecto mínimo".
INFERENCE_D_NEGLIGIBLE = 0.2
INFERENCE_ETA2_NEGLIGIBLE = 0.01

# Máximo de tests inferenciales por dataset (los de mayor |separación|
# primero) para no inflar el informe.
INFERENCE_MAX_TESTS = 10

# --- Calidad → limpieza recomendada (F5) ------------------------------------

# Nunca se aplica limpieza automáticamente: solo se recomienda.
CLEANING_AUTO_APPLY = False

# Máximo de % de missing en una columna para recomendar dropna por
# columna (por encima, se recomienda imputar).
CLEANING_DROPNA_MISSING_PCT = 40.0

# Multiplicador IQR para el clip extremo recomendado (más
# conservador que el outlier estándar).
CLEANING_CLIP_IQR_MULTIPLIER = 3.0

# Máximo de acciones de limpieza recomendadas por dataset.
CLEANING_MAX_ACTIONS = 12

# --- Target & recomendación ML (F6) ------------------------------------------

# Cardinalidad mínima de una columna para ser candidata a target.
ML_MIN_TARGET_CARDINALITY = 2

# Clases máximas para clasificación binaria / multiclase.
ML_BINARY_MAX_CLASSES = 2
ML_MULTICLASE_MAX_CLASSES = 20

# Valores únicos mínimos de una numérica para considerar regresión.
ML_REGRESSION_MIN_UNIQUE = 20

# Máximo de candidatos a target devueltos (ordenados por score).
ML_MAX_TARGET_CANDIDATES = 5
