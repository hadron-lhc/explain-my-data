# PLAN_V2 — Ruta de mejora incremental

> Visión a largo plazo: [`ROADMAP.md`](ROADMAP.md). Este documento es
> el **plan de ejecución**: fases pequeñas, cada una cierra con la
> matriz QA sin regresiones. Juez de todo: el playground
> `explain-my-data-qa/` (local, fuera de git).

## Principios

1. Una fase = un alcance acotado; no saltar features entre fases.
2. Cierre de fase = `pytest` + `ruff` + matriz QA (needs/claims/
   absent/⚠) sin regresiones + docs actualizadas.
3. Un commit por fase completada (mensaje describiendo la fase).
4. Precisión medida antes que cantidad de features: si `verify` no lo
   cubre, el insight es de segunda clase.
5. Decisiones ya tomadas: sin Pyright por ahora; ML en F6 = solo
   detección + recomendación (sin sklearn, sin entrenar).

## Estado

- [x] **F0 — Cimientos** (2026-10-15): Fase B commiteada, README
  actualizado, CI (pytest + ruff + smoke). Sin Pyright.
- [x] **F1 — Verificación ampliada** (2026-10-15): verify cubre
  comparative/pattern/distribution/quality-extreme. Precisión 100%
  (366/366), **cobertura 70.1%** (23.4% → 70.1%). +25 needs
  comparativos/pattern en `known/` (✓69 total). Directiva nueva
  `skip:` para needs deliberadamente ausentes; `check --strict` ahora
  es un gate real (exit 0).
- [x] **F2 — Curación de insights** (2026-10-15): un hallazgo = una
  tarjeta. `src/curation.py`: cap de patrones por par (extremos
  firmados, `PATTERN_PER_PAIR_CAP=2`) + dedupe relationship↔pattern
  (la tarjeta "differs across" se pliega como evidence/caveat del
  patrón primario). Resumen ejecutivo estratificado por categoría
  (`EXECUTIVE_PER_CATEGORY_MAX=3`, backfill por score). Páginas del
  informe ordenadas por columna (agrupación temática). QA ya no
  cuenta las reimpresiones del ejecutivo. 522 → **390 insights**
  (promedio 39/dataset).
- [x] **F3 — Análisis temporal (V3a)** (2026-10-15): `src/temporal.py`
  estudia cada columna datetime (detectada por el profiling) contra
  sus mejores métricas numéricas: granularidad inferida, tendencia
  (pendiente + % first→last, con guard de robustez ante outliers de
  palanca), estacionalidad (autocorr en el lag natural sobre la serie
  detrendeada), huecos de período, cambio de régimen (sobre la serie
  detrendeada para no confundirlo con una tendencia) y extremos
  (3×IQR de medias de período). Charts de serie (línea + rolling) y
  sección "Temporal series" en el informe. Corpus QA: 3 datasets
  sintéticos con fechas (monthly_sales, daily_temperature,
  weekly_orders) + extractores verify temporales (tendencia,
  estacionalidad, régimen, huecos, extremos).
- [x] **F4 — Inferencia estadística (V3b)**
- [x] **F5 — Calidad → limpieza recomendada (V5)**
- [x] **F6 — Target & recomendación ML (V4, sin entrenar)**

Línea base QA al entrar en F0: `✓44/46 · claims 31/31 · absent✗0 ·
⚠0 · ℹ37` sobre 522 insights; verify precisión 100% (122/122),
cobertura 23.4%; 253 tests.

Línea base al cerrar F1: `✓69/69 · ⊘2 skips deliberados · claims
31/31 · absent✗0 · ⚠0 · ℹ37` sobre 522 insights; verify precisión
100% (366/366), **cobertura 70.1%**; `check --strict` exit 0.

Línea base al cerrar F2: `✓69/69 · ⊘2 · claims 31/31 · absent✗0 ·
⚠0 · ℹ28` sobre **390 insights** (39/dataset, titanic 79); verify
precisión **100% (276/276)**, **cobertura 70.8%**; `check --strict`
exit 0; 263 tests.

Línea base al cerrar F3: `✓77/77 · ⊘2 · claims 42/42 · absent✗0 ·
⚠0 · ℹ29` sobre **403 insights** (13 datasets, 3 temporales nuevos);
verify precisión **100% (287/287)**, **cobertura 71.2%**;
`check --strict` exit 0; 275 tests.

Línea base al cerrar F4: `✓77/77 · ⊘2 · claims 52/52 · absent✗0 ·
⚠0 · ℹ29` sobre **413 insights** (13 datasets, 10 inference
insights nuevos); verify precisión **100% (297/297)**, **cobertura
71.9%**; `check --strict` exit 0; 290 tests.

Línea base al cerrar F5: `✓87/87 · ⊘2 · claims 55/55 · absent✗0 ·
⚠0 · ℹ29` sobre **444 insights** (14 datasets, dirty_sales nuevo
con type-mismatch/missing/inf/duplicates); verify precisión
**100% (302/302)**, **cobertura 68.0%**; `check --strict` exit 0;
298 tests.

Línea base al cerrar F6: `✓94/94 · ⊘2 · claims 56/56 · absent✗0 ·
⚠0 · ℹ29` sobre **550 insights** (15 datasets, churn nuevo con
target binario churned + 5 features mixtas); verify precisión
**100% (375/375)**, **cobertura 68.2%**; `check --strict` exit 0;
318 tests.

---

## F0 — Cimientos (ingeniería)

**Objetivo**: main verde, commiteada, con red de seguridad.

**Alcance**:

- Commit de la Fase B pendiente (`src/config.py`, `src/insights.py`,
  `src/planner.py`, `tests/test_planner.py`) — fix de selección de
  relaciones + umbral de insight cat-num alineado.
- Actualizar `README.md` del repo: estado V2, conteo real de tests,
  mención del QA harness y de la mejora de selección.
- CI GitHub Actions: `pytest -q` + `ruff check .` + smoke
  (`python -m src.analyze data/messy_sales.csv`). El corpus QA es
  local (fuera de git): la matriz completa se corre a mano en el
  cierre de cada fase.
- Sin Pyright (decisión: fuera por ahora).

**Validación**: CI verde en el commit; smoke local OK.

**Corte**: no entrar a features nuevas en F0.

## F1 — Verificación ampliada (cierra el agujero del 23%)

**Objetivo**: que `verify` cubra las familias actualmente opacas
(sobre todo `comparative_insights`, 79 insights con cero needs y
cero verificación).

**Alcance**:

- Extractores nuevos en `qa.py verify`:
  - comparative ranking ("Group 'G' leads/covers X% of …") → recomputa
    totales/tasas;
  - comparative driver ("driven by average values / group sizes
    rather than …") → recomputa spread de medias vs volumen;
  - comparative consistency ("the same group leads …") → recomputa;
  - comparative nulls ("too similar to conclude", weak corr,
    independent pair con V/pearson citados) → recomputa;
  - patterns ("Group 'G' of C stands out in N") → recomputa medias
    ≥1σ;
  - distribution (skew/outliers citados) → recomputa.
- Añadir `need:` de comparative a `known/*.md` (~2-4 por dataset
  donde sean deterministas).
- Sin tocar `src/`.

**Validación**: precisión ≥99% (cualquier ✗ se investiga antes de
bajar el número), **cobertura ≥60%**, 0 needs ausentes, matriz
preexistente intacta.

**Corte**: no "arreglar" el analizador dentro de F1 — los falsos
positivos detectados se anotan como backlog para su fase.

## F2 — Curación de insights

**Objetivo**: menos volumen (~522 insights / 10 datasets), misma
señal.

**Alcance**:

- Cap por familia en `insights.py`/`comparative_insights.py`
  (constantes en `config.py`).
- Dedupe de dobles rutas (mismo hallazgo vía pattern+relationship,
  cat-num por ambos lados).
- Scoring/ordenado cross-family para el resumen ejecutivo.
- Agrupación temática en `reporting.py`.

**Validación**: insights/dataset ≤~40 promedio sin perder ningún
need/claim; verify ≥ la precisión de F1; matrix intacta.

**Corte**: no cambiar umbrales de selección de relaciones (ya
tunados en la Fase B).

## F3 — Análisis temporal (V3a)

**Objetivo**: explotar las columnas datetime (hoy detectadas por
profiling y muertas para el análisis).

**Alcance**:

- `src/temporal.py`: granularidad inferida, tendencia (slope /
  % cambio período a período), estacionalidad (si ≥2 ciclos),
  períodos faltantes, cambio de régimen (medias por ventana),
  outliers temporales.
- Planner enlaza fechas con métricas (mes→métrica, año→métrica).
- Charts de serie (línea + rolling).
- Corpus QA con datasets de fechas (sintéticos + reales) y claims
  de tendencia.

**Validación**: nuevos needs temporales ✓; verify de tendencias/PoP;
matriz preexistente intacta; tests + ruff.

**Corte**: sin forecasting ni ML de series.

## F4 — Inferencia estadística (V3b)

**Objetivo**: pasar de "se separan" a "con qué confianza y tamaño
de efecto".

**Alcance**:

- ANOVA / Kruskal-Wallis en cat-num (caveats Levene/Shapiro cuando
  aplique), post-hoc simplificado (top group vs resto).
- CIs (t o bootstrap) para medias y diferencias.
- Effect sizes: Cohen's d / η² (V ya existe en cat-cat).
- Corrección Benjamini-Hochberg sobre la familia de tests del
  dataset.
- Mensajes: distinguir significancia estadística de efecto
  práctico; caveat automático si n chico.

**Validación**: extractores verify de p-values/effect sizes; matrix
intacta; claims nuevos de significancia.

**Corte**: no sustituir `group_separation` (sigue siendo el score
de selección).

## F5 — Calidad → limpieza recomendada (V5)

**Objetivo**: cerrar el loop problema→acción.

**Alcance**:

- `quality.py` emite recomendaciones estructuradas (candidato a
  `CleaningAction` + porqué + impacto estimado).
- `cleaning.py`: log estructurado (qué, cuántas filas, antes/después)
  y pipeline reproducible (lista de acciones serializable y
  re-aplicable).
- UI muestra recomendaciones.
- Corpus de datasets sucios en QA (categorías inconsistentes, fechas
  rotas, type mismatches, missing) con needs de calidad y
  recomendaciones.

**Validación**: recommendations precisas en corpus sucio; limpieza
idempotente y loggeada; matrix intacta.

**Corte**: nunca auto-aplicar limpieza (principio del ROADMAP:
los datos del usuario no se modifican en silencio).

## F6 — Target & recomendación ML (V4, sin entrenar)

**Objetivo**: decir "tu target es X, el problema es Y, prueba con Z
modelos".

**Alcance**:

- Detección de candidatos a target (cardinalidad, variabilidad,
  role inferido, missingness, "¿otras columnas lo explican?").
- Clasificación del problema (binaria / multiclase / regresión).
- Features candidatas / excluidas (ids).
- Recomendación determinista de modelos por tipo y tamaño de
  dataset (tabla de reglas, sin sklearn).
- Needs de target en el corpus QA (churn, precios clásicos).

**Validación**: needs de target ✓; verify de tasas de target;
matrix intacta.

**Corte explícito**: entrenamiento/evaluación real (scikit-learn, CV,
feature importance) queda como **F7 futura**, solo si se decide.

---

## Fuera de alcance (documentado, no prometido)

Pyright, forecasting, LLM, clustering, deployment.

## Ritual de cierre de fase

```bash
# repo
cd ../explain-my-data && uv run pytest -q && uv run ruff check .

# playground QA (local)
cd ../explain-my-data-qa
uv run --project ../explain-my-data python qa.py run
uv run --project ../explain-my-data python qa.py check --strict
uv run --project ../explain-my-data python qa.py verify
# comparar contra la línea base y actualizar "Estado" acá arriba
```
