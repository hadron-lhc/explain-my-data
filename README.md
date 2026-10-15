# Explain My Data

Mini-analizador de datasets CSV: detecta patrones, genera insights y
gráficas automáticamente. **Sin ML ni LLM** — todo el análisis es
estadístico y las conclusiones salen de plantillas deterministas.
Cero tokens, cero costos.

## Qué hace

Dado un CSV, el pipeline:

1. **Perfila** el dataset (roles de columna: numérico, categórico,
   booleano, fecha, identificador).
2. **Detecta problemas de calidad**: faltantes, duplicados, valores
   inválidos, outliers (IQR), conversiones de tipo fallidas.
3. **Analiza relaciones**: num↔num (Pearson/Spearman), cat↔num
   (separación de grupos) y cat↔cat (chi-cuadrado + Cramér's V).
4. **Describe distribuciones**: histograma (Freedman-Diaconis),
   asimetría, curtosis, concentración del top 5%, modalidad.
5. **Encuentra patrones**: grupos con medias muy distintas (≥1σ),
   outliers por columna, columnas constantes, agregados dominantes.
6. **Redacta insights** en inglés con evidencia numérica y un
   resumen ejecutivo (máx. 2 warnings de calidad + lo mejor por score).
7. **Insights comparativos** (reglas puras): ranking de grupo por
   total con contraste volumen/valor, deconposición de drivers
   (¿mueve el volumen o el valor?), consistencia entre métricas
   ("el mismo grupo lidera dos medidas"), distribuciones categóricas
   sin valor dominante, y hallazgos nulos (binarias casi iguales,
   correlaciones débiles, pares independientes) con los números
   exactos en el mensaje.
8. **Grafica** automáticamente: scatter con línea de tendencia, barras
   ordenadas, boxplots (muestreados), histogramas y heatmaps de
   contingencia.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.

```bash
uv sync
```

## Uso

### CLI

```bash
uv run python -m src.analyze data/messy_sales.csv
```

Imprime el perfil, calidad, estadísticas, distribuciones, relaciones,
candidatos de análisis, insights y el resumen ejecutivo.

### API + Web UI

```bash
uv run uvicorn web.backend.main:app --port 8000
```

Endpoints: `POST /analyze`, `POST /clean`, `POST /report` (informe HTML
autocontenido y listo para imprimir; con `?hide=id1,id2` excluye
insights que el usuario ocultó en la UI y con
`?hide_charts=histogram:col,bar:a:b` excluye charts de
"Additional charts"), `GET /health`.

El informe `/report` está pensado para **print-to-PDF**: cada sección
es una página (portada, overview, calidad, patterns, resumen ejecutivo,
un insight por página con sus gráficos, apéndice técnico), con CSS de
`@media print` y auto-impresión al abrirlo con el hash `#print`.

La UI es una SPA estática en `web/frontend/` (Plotly por CDN).
Para servirla en desarrollo:

```bash
python -m http.server 5500 --directory web/frontend
```

y abrí `http://localhost:5500`. La UI apunta a `http://127.0.0.1:8000`
(ver `web/frontend/api.js`). La carga es un dropzone grande con
drag & drop del CSV o click para abrir el explorador (al soltar o
seleccionar el archivo empieza el análisis solo). El flujo de la UI es:
overview (KPIs + perfil de columnas) → calidad → limpieza (drawer con
issues detectados, falta, duplicados, outliers, rangos; con deshacer) →
patterns → insights (resumen ejecutivo destacado + todos con
explicación, patrón asociado y gráficos) → gráficos adicionales →
descarga del informe **PDF** (diálogo de impresión del navegador), HTML
y CSV limpio.

Cada tarjeta de insight y cada chart de "Additional charts" tiene un
**✕** para ocultarlo: los ocultos se guardan en `localStorage` con la
huella (sha-256) del archivo, se restauran con "Restore all", y el
PDF/HTML descargado los respeta vía `/report?hide=...` y
`/report?hide_charts=...` (identidades `visualizationIdentity` de la
UI, p. ej. `histogram:revenue` o `bar:country:revenue`) para que
impreso y pantalla coincidan.

### Tests y lint

```bash
uv run pytest -q
uv run ruff check .
```

## Configuración de umbrales

Todos los thresholds del análisis (percentiles de selección de
relaciones, multiplicadores IQR, límites de muestreo de gráficas,
tamaño del resumen ejecutivo) viven en **`src/config.py`**.
Los módulos los importan de ahí.

## Estructura

```
src/
  analyze.py        CLI
  analysis.py       orquestador del pipeline
  profiling.py      roles y perfil de columnas
  quality.py        problemas por columna
  statistics.py     stats descriptivas
  distributions.py  histogramas y forma univariada
  relationships.py  num-num, cat-num, cat-cat
  aggregations.py   métricas por grupo
  patterns.py       diferencias de grupos y outliers
  planner.py        qué analizar y cómo graficarlo
  insights.py       plantillas de insights + resumen ejecutivo
  comparative_insights.py  rankings, drivers, nulos
  visualizations.py constructores de gráficas
  reporting.py      informe HTML (plotly embebido)
  cleaning.py       transformaciones de limpieza
  preparation.py    preparación para análisis
  config.py         umbrales centrales
  utils_stats.py    utilidades IQR compartidas
web/
  backend/          FastAPI
  frontend/         UI (JS vanilla + Plotly)
tests/              253 tests
docs/ROADMAP.md     especificación completa
docs/PLAN_V2.md     plan de mejora por fases (ejecución)
```

## Principios

- **Sin ML/LLM**: solo estadística clásica y reglas explícitas.
- **Sin dependencias pesadas**: pandas, numpy, scipy, plotly, fastapi.
- **Determinista**: mismos datos → mismos insights (muestreos con
  `random_state` fijo).
- **Sin escrituras sobre tus datos**: la limpieza devuelve copias.

## Estado

V2: pipeline completo + insights comparativos, **253 tests**, ruff
limpio. La selección de relaciones incluye overrides (piso fuerte
para pares ≥0.9 y top-N por columna "hueca") porque el percentil
p75 solo echaba relaciones clásicas en datasets con columnas
redundantes; validado contra un playground de QA local
(`explain-my-data-qa`, 10 datasets clásicos con ground truth:
recall de needs 44/46, claims 31/31, precisión medida 100%).

Ruta de mejora por fases en [`docs/PLAN_V2.md`](docs/PLAN_V2.md);
visión a largo plazo en [`docs/ROADMAP.md`](docs/ROADMAP.md).
