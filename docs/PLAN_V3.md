# PLAN_V3 — Calidad de insights: menos ruido, más sustancia

> Continúa [`PLAN_V2.md`](PLAN_V2.md) (F0–F6 completadas). Visión a
> largo plazo: [`ROADMAP.md`](ROADMAP.md). Este documento ataca el
> problema de **utilidad**: hoy el analizador emite demasiadas
> tarjetas, muchas triviales o redundantes, y pocas cuentan una
> historia completa. Juez de todo: el playground
> `explain-my-data-qa/` (local, fuera de git).

## Diagnóstico (evidencia de los outputs reales)

Sobre titanic (93 insights), diamonds (68), adult (78):

1. **Tautologías**: pares de columnas equivalentes reportados como
   "asociaciones" — `survived~alive` (V=1.00), `embarked~embark_town`
   (V=1.00), `who~adult_male` (V=1.00), `sex~who` (0.95),
   `sex~adult_male` (0.91), `pclass~class`. ~10 tarjetas diciendo lo
   mismo.
2. **Eco por columnas derivadas**: `who` = f(sex, age). Todas sus
   tarjetas reiteran hechos de sex/age (`who~survived`,
   `child stands out in age`, `woman in pclass`). ~8 tarjetas.
3. **Verdades triviales**: `sibsp`/`parch` (conteos) "right-skewed";
   `pclass` (3 valores) "multiple subpopulations"; "top 5% concentran
   42.5% de sibsp". Trivial en conteos y discretos.
4. **Ceguera por subconjunto**: `deck` 77.2% missing, pero emite
   `age differs across deck groups (separación 1.92)` sobre n=203 no
   aleatorio. Fuerza engañosa. ~6 tarjetas.
5. **Significancia vs. práctica**: `pclass differs across sex
   (separación 0.28)`: male 2.39 vs female 2.16 en escala 1-3.
   Pasa el umbral 0.25 pero es practicamente nada.
6. **Volumen sin presupuesto**: 93 tarjetas; nadie lee 93. El
   executive summary es top-8 por score, desconectado.
7. **Tarjetas poco completas**: "carat~price (Pearson 0.92)" sin R²,
   sin "no lineal (Spearman > Pearson)", sin interpretación.

## Principios (heredados de V2)

1. Una fase = un alcance acotado; no saltar features entre fases.
2. Cierre de fase = `pytest` + `ruff` + matriz QA sin regresiones +
   docs actualizadas.
3. Un commit por fase completada.
4. Precisión medida antes que cantidad: si `verify` no lo cubre, el
   insight es de segunda clase.
5. **Determinista**: sin LLM, sin sklearn, sin entrenar. El techo
   honesto es: detectar redundancia/trivialidad/validez y templar un
   resumen narrativo. Semántica de dominio y narrativa causal libre
   quedan fuera (posible F7 futura con LLM, no prometida).

## Estado

- [x] **P1 — Redundancia: equivalencias y derivadas** (2026-10-16):
  módulo `src/equivalence.py`. Detección de equivalencia exacta
  (cat↔cat V≥0.98, num↔num |r|≥0.995 **y biyectivo**, cat↔num con la
  numérica como fuente) y de columnas derivadas (test direccional
  `_is_step_function` dentro de cada grupo del socio; umbral
  `DERIVED_MIN_GROUP_ROWS=40`; descarta derivación mutua ambigua).
  Fold de insights en **todos** los generadores
  (`filter_insights`: relationship, pattern, aggregation,
  comparative, distribution, temporal, inference, ml) + filtrado de
  pares/relaciones/agregados. Una sola tarjeta de equivalencia por
  fold. Preferencia fuente: menos missing (ahora visible: `_prepare`
  conserva nulos), cardinalidad natural, empate alfabético. Chi2 sin
  corrección de Yates en el V de detección. Corpus QA adaptado
  (`titanic.md`, `adult.md`) + dataset sintético `dup_columns.csv`
  con `known/dup_columns.md` (regresa: copia numérica, etiqueta
  cat, y no-pliegue de casi-lineal no biyectivo). 21 tests en
  `tests/test_equivalence.py`.
- [x] **P2 — Trivialidades: conteos y discretos** (2026-10-16):
  clasificación de columnas en `NumericDistribution` (`n_unique`,
  `looks_like_count`, `is_low_cardinality_discrete`). Conteo =
  entero, no negativo y cardinalidad ≤ `COUNT_MAX_UNIQUE=30`. En un
  conteo se suprime el skew salvo que |skew| ≥ `COUNT_SKIP_SKEW=5.0`
  y la concentración del 5% superior salvo que ≥
  `COUNT_SKIP_TOP_PCT=80.0` (la asimetría y la concentración son la
  forma esperada del conteo, no un hallazgo). En una numérica
  discreta con ≤ `DISCRETE_SKIP_MODES_MAX=5` valores se suprime
  "multiple subpopulations" (casi todos los valores son modos).
  `fare`/`capital_gain` (continuas) conservan sus insights.
  Corpus: `absent:` de regresión en `titanic.md` (sibsp/parch skew,
  pclass modas) y `adult.md` (education_num modas). 12 tests nuevos
  en `tests/test_distributions.py`.
- [ ] **P3 — Validez contextual: missingness y significancia práctica**
- [ ] **P4 — Narrativa y presupuesto de insights**
- [ ] **P5 — Tarjetas más ricas (detalle y cross-links)**
- [x] **G1 — QA transversal: `noise:` a nivel info** (2026-10-16):
  `_evaluate_dataset` extiende el ruido a `severity=info`; un insight
  info que matchea un `noise:` sin `need` que lo justifique es ✗
  (`✗info`) y hace fallar `--strict` (los warnings siguen solo
  silenciándose). Precedencia need > noise. Los 15 `known/*.md`
  anclan sus `noise:` a títulos reales de warning (`... in`,
  `duplicate rows detected`) para no cazar por subcadena las
  recomendaciones de limpieza de F5. Línea base tras G1: `✓94/94 ·
  ⊘2 · claims 56/56 · absent✗0 · ⚠0 · ℹ29 · ✗info0` sobre **550
  insights** / 15 datasets; verify **100% (375/375)**, **cobertura
  68.2%**; `check --strict` exit 0; 318 tests.

Línea base tras P2 (16 datasets con `dup_columns`): `✓100/100 · ⊘2 ·
claims 59/59 · absent✗0 · ⚠0 · ℹ29 · ✗info0` sobre **521 insights**;
verify **100% (362/362)**, **cobertura 69.5%**; `check --strict` exit
0; **344 tests**, ruff limpio. títanic 63→58 (skew de sibsp/parch y
modas de pclass suprimidos); adult se mantiene en 76 pero con
`education_num` ya sin tarjeta de modas — su skew era negativo, así
que no era trivial y se conserva. `fare` y `capital_gain` conservan
sus insights de skew y concentración.)

(Línea base tras P1: `✓100/100 · ⊘2 · claims 59/59 · absent✗0 · ⚠0 ·
ℹ29 · ✗info0` sobre **539 insights**; verify **100% (369/369)**,
**cobertura 68.5%**; `check --strict` exit 0; 335 tests, ruff
limpio. títanic 93→63, adult 78→61.)

(Línea base al entrar en P1 = línea base al cerrar F6: `✓94/94 · ⊘2 ·
claims 56/56 · absent✗0 · ⚠0 · ℹ29` sobre **550 insights** / 15
datasets; verify **100% (375/375)**, **cobertura 68.2%**;
`check --strict` exit 0; 318 tests.)

---

## P1 — Redundancia: equivalencias y derivadas

**Objetivo**: eliminar tarjetas que repiten la misma información por
columnas equivalentes o derivadas.

**Alcance**:

- `src/relationships.py` (o módulo nuevo `src/equivalence.py`):
  - Detección de **equivalencia exacta** entre columnas:
    - cat↔cat: `cramers_v ≥ 0.98` (V=1.00 es renombre puro;
      `alive`/`survived`, `embarked`/`embark_town`).
    - num↔num: `|pearson| ≥ 0.995` y mismo `n` (variables
      geométricamente proporcionales).
    - cat↔num: la cat codifica la num con ≤K grupos (p.ej.
      `class` etiqueta `pclass`).
  - Detección de **columna derivada** (heurística, sin LLM): una cat
    con cardinalidad baja cuyo nombre o estructura la vincula a otras
    (p.ej. `who` ≈ f(sex, age-bins), `adult_male` ≈ sex∧age>18).
    Enfoque pragmático: si la cat tiene V≥0.9 con OTRA cat y además
    separa fuertemente una num, se marca "posiblemente derivada de
    X/Y" y sus insights se pliegan o se anotan con caveat.
  - **Fold de insights**: al detectar un grupo de columnas
    equivalentes/derivadas, se conserva UNA tarjeta "X e Y contienen
    la misma información" (o se elige la columna "fuente" por
    preferencia: sin missing, cardinalidad natural, nombre más
    específico) y el resto de insights del grupo se descartan o se
    pliegan como caveat en la tarjeta superviviente.
  - Preferencia fuente>derivada al **generar** insights: si `who` es
    derivada de sex+age, los insights de relación se emiten contra
    sex/age, no contra who. (Implementación: filtrar o degradar
    columnas derivadas en la selección de relaciones.)
- `src/config.py`: constantes `EQUIVALENCE_MIN_CRAMERS_V=0.98`,
  `EQUIVALENCE_MIN_CORR=0.995`, `DERIVED_MAX_CARDINALITY=10`.
- `src/analysis.py`: integrar detección de equivalencias antes de
  generar insights de relación; pasar el mapa de fold a los
  generadores.
- Corpus QA: añadir `noise:` o `absent:` en `known/*.md` para las
  tarjetas tautológicas que deben desaparecer (p.ej. titanic:
  `absent: survived alive`, `absent: embarked embark_town`).
  Añadir un dataset sintético con columnas duplicadas/derivadas
  explícitas para testear la detección (p.ej. `dup_columns.csv` con
  `price` y `price_copy`, `sex` y `is_male`).
- Tests: `tests/test_equivalence.py` — detección de equivalencia
  exacta, fold de insights, preferencia fuente>derivada, no-fold de
  columnas distintas con V alto pero no equivalente.

**Validación**: tarjetas tautológicas desaparecen (absent✗ en QA);
matrix intacta; conteo de insights baja ~10-15% en titanic/adult.

**Corte**: no plegar columnas con V alto pero información distinta
(p.ej. `survived`~`sex` V=0.54 NO es equivalencia; es asociación real
y debe permanecer).

---

## P2 — Trivialidades: conteos y discretos

**Objetivo**: no emitir tarjetas cuya conclusión es trivialmente
cierta para la forma de la variable.

**Alcance**:

- `src/distributions.py` + `src/insights.py`:
  - **Conteo**: detectar variable de conteo (enteros, no-negativos,
    max razonable ≤30, sin missing estructural). Si es conteo:
    - Suprimir insight "right-skewed" (trivial: los conteos casi
      siempre lo son).
    - Suprimir "top X% concentran Y% del total" (trivial en conteos).
    - Conservar solo si el skew es extremo e inesperado (p.ej.
      skew > 5, o concentración > 80%).
  - **Discreto de baja cardinalidad**: si `n_unique ≤ 5` (y es
    numérica discreta), suprimir "multiple subpopulations" (3 modas
    en 3 valores posibles es meaningless).
  - `src/config.py`: `COUNT_MAX_UNIQUE=30`, `COUNT_SKIP_SKEW=…`,
    `DISCRETE_SKIP_MODES_MAX=5`, umbrales de excepción.
- Corpus QA: `noise:` o `absent:` para las tarjetas triviales en
  titanic (`sibsp right-skewed`, `pclass multiple subpopulations`).
- Tests: `tests/test_distributions.py` — supresión en conteos,
  conservación si skew extremo, supresión de modas en discretos.

**Validación**: tarjetas triviales desaparecen; matrix intacta;
conteo baja ~5-10%.

**Corte**: no suprimir distribuciones informativas de numéricas
continuas ni de conteos con señal real (p.ej. fare sí tiene skew
informativo; sibsp no).

---

## P3 — Validez contextual: missingness y significancia práctica

**Objetivo**: una relación fuerte sobre un subconjunto pequeño o una
diferencia estadística pero no práctica no debe sonar fuerte.

**Alcance**:

- `src/insights.py` (generación de relationship/pattern):
  - **Missingness-aware**: si la columna de grupo (o la numérica)
    tiene `pct_missing > 30%`, la tarjeta lleva caveat dominante
    "based on only X% of rows" y el score se degrada; si el n por
    grupo < 30 tras dropna, **no emitir** la tarjeta (o emitirla como
    "low confidence" separada). Evitar que `deck` (77% missing)
    genere relaciones de fuerza aparente 1.92.
  - **Significancia práctica**: gate adicional a
    `group_separation ≥ MIN_GROUP_SEPARATION`: exigir también
    diferencia relativa `(|max−min| / median) ≥ MIN_RELATIVE_DIFF`
    (p.ej. 15%) O estandarizada ≥ 0.5. `pclass across sex` (0.28,
    ~10% rel) queda fuera.
  - `src/config.py`: `RELATIONSHIP_MAX_MISSING_PCT=30`,
    `RELATIONSHIP_MIN_GROUP_N=30`, `MIN_RELATIVE_DIFF=0.15`.
- Corpus QA: `absent:` para `age differs across deck` (o su
  equivalente) y para `pclass differs across sex` en titanic.
- Tests: `tests/test_relationships.py` — degradación/skip por
  missingness, gate de significancia práctica, no-regresión de
  relaciones válidas.

**Validación**: tarjetas engañosas desaparecen o se califican;
matrix intacta; verify sin regresiones.

**Corte**: no suprimir relaciones legítimas con missing moderado
(<30%) ni diferencias pequeñas pero consistentes en n grande si
superan el gate relativo.

---

## P4 — Narrativa y presupuesto de insights

**Objetivo**: el informe se lee como un resumen, no como un vertedero
de tarjetas.

**Alcance**:

- **Presupuesto global**: `src/config.py` `INSIGHT_BUDGET_TOTAL=40`
  (o por familia). Tras la curación (F2) y los folds (P1-P3), si el
  total supera el presupuesto, recortar por prioridad:
  calidad(warning) > relaciones más fuertes > distribuciones >
  patrones > aggregation. Nunca recortar warnings de calidad salvo
  duplicados.
- **Executive summary narrativo (determinista)**:
  - `src/insights.py` `generate_executive_summary`: en vez de
    top-N por score, construir 4-6 frases por template:
    1. "Dataset con N filas × M columnas. [Calidad: X warning
       principal]."
    2. "Relaciones más fuertes: A~B (fuerza), C~D (fuerza)."
    3. "Distribuciones a tener en cuenta: E sesgada, F multimodal."
    4. "[Si temporal] Tendencia/estacionalidad de G."
    5. "[Si ML] Target probable: H (tipo)."
  - Cada frase referencia (por id/título) la tarjeta fuente; el
    resumen sigue siendo una lista de `Insight` (para no romper
    `_INSIGHT_LIST_FIELDS`), pero con `category="executive_summary"`
    y mensajes enlazados.
  - Dedupe del resumen contra tautologías (hereda los folds de P1).
- `src/reporting.py`: la sección Executive summary del HTML muestra
  las frases narrativas en orden, no tarjetas dispersas.
- Corpus QA: adaptar needs de executive summary si cambian los
  títulos; mantener claims numéricos verificables.
- Tests: `tests/test_executive.py` — presupuesto respetado, frases
  presentes, sin tautologías en el resumen.

**Validación**: total insights cae a ≤ presupuesto en datasets
medianos; exec summary lee como párrafo; matrix intacta.

**Corte**: no perder información verificable (los claims de verify
deben seguir encontrando sus números en las tarjetas fuente, no en
el resumen templado).

---

## P5 — Tarjetas más ricas (detalle y cross-links)

**Objetivo**: cada tarjeta superviviente explica el hallazgo completo.

**Alcance**:

- `src/insights.py` (enriquecimiento de mensajes):
  - Correlaciones: añadir R² (`r²`) e interpretación "lineal/no
    lineal" cuando `|spearman| − |pearson| > 0.1`.
  - Group-differences: añadir rango completo (max/min), ratio
    max/min, y n por grupo (no solo el par extremo).
  - Skew: añadir min/q1/median/q3/max para contextualizar.
  - Asociaciones cat-cat: top-2-3 combos sobrerrepresentados (no
    solo 1).
  - Cross-links deterministas: cuando dos tarjetas comparten una
    columna fuerte (misma columna en dos relaciones top), añadir
    "relacionado con: <título corto>" en caveats.
- `src/config.py`: `CROSSLINK_MIN_STRENGTH`, límite de cross-links
  por tarjeta.
- Corpus QA: ajustar claims si los mensajes ganan números nuevos
  (verificar que no rompa `_extract_numbers`/`_groups_match`).
- Tests: `tests/test_enrichment.py` — R² presente, rango en
  group-diffs, cross-links no excesivos.

**Validación**: mensajes más completos; verify sigue al 100%; matrix
intacta (salvo claims que requieran ajuste numérico puntual).

**Corte**: no inflar mensajes con ruido; la riqueza es
contexto-verificable, no relleno.

---

## QA transversal (G1): `noise:` a nivel info

**Objetivo**: el harness puede verificar que el ruido de nivel
`info` fue suprimido (hoy `noise:` solo silencia `warnings`).

- `qa.py` `_evaluate_dataset`: extender `_noise_hit` a insights `info`
  (hoy solo cuenta `warnings_noised`). Un insight `info` que matchea
  un `noise:` **sin** `need` que lo justifique es ✗ (`✗info`): el
  ruido que debimos suprimir se emitió. Precedencia need > noise (si
  un need lo cubre, es contenido esperado y pasa). Los warnings
  mantienen la semántica de silencio (no fallan). `--strict` suma
  `info_noised` a los fallos; `summary.md` muestra `✗info` y el
  patrón/cita por dataset.
- Integrado en P1-P3: cada fase que suprime tarjetas añade
  `noise:`/`absent:` en `known/*.md` correspondiente.

---

## Fuera de alcance (documentado, no prometido)

- Semántica de dominio (¿importa "deck"?), narrativa causal libre,
  juicio de "interesante" según oficio del lector → posible **F7
  futura con LLM**, no prometida.
- Pyright, forecasting, clustering, deployment (heredado de V2).
- sklearn / entrenamiento / feature importance (heredado de V2 F6).

## Ritual de cierre de fase (idéntico a V2)

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

## Estimación de impacto (titanic, el más ruidoso)

| Fase | Δ insights (titanic) | Motivo |
|------|----------------------|--------|
| P1 | −10 a −15 | tautologías + derivadas plegadas |
| P2 | −5 a −8 | conteos/descartes triviales |
| P3 | −5 a −8 | deck (missing) + separaciones marginales |
| P4 | recorte a ≤40 total | presupuesto + narrativa |
| P5 | 0 (enriquece) | mensajes más ricos, no más tarjetas |

Meta: titanic de 93 → ~30-40 tarjetas de calidad, con executive
summary de 4-6 frases coherentes.
