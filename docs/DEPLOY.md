# Deploy en Cloudflare

Arquitectura: un **Worker** proxy enruta todas las peticiones a un
**Container** administrado por un Durable Object con el scheduling
policy `durable_object` (FastAPI + SPA servida en el mismo origen).
Scale-to-zero tras 10 minutos de inactividad; solo se cobra el tiempo
con instancias activas.

```
usuario → Worker (explain-my-data.<subdomain>.workers.dev)
            └→ Container (python:3.12-slim, uvicorn :8000)
                 ├─ /           → SPA (StaticFiles)
                 ├─ /health     → ok
                 ├─ /analyze    → análisis
                 ├─ /clean      → limpieza (json | ?format=csv)
                 └─ /report     → informe HTML autocontenido
```

## Prerrequisitos

- Cuenta Cloudflare con **Workers Paid** (los Containers requieren el
  plan de pago).
- Docker corriendo localmente (`docker info` debe salir bien) con el
  plugin **buildx** (`docker buildx version`). En Ubuntu se instala con
  `sudo apt install docker-buildx`.
- Node 22+ y `npm`.
- `uv` (para el workflow de CI; el build del contenedor usa la imagen
  oficial de uv, no requiere uv local).

## Deploy manual

```bash
npm install                 # dependencias del Worker (wrangler)
npx wrangler login          # o exportar CLOUDFLARE_API_TOKEN
npx wrangler deploy
```

Wrangler construye la imagen con Docker (`linux/amd64`), la sube al
registro de Cloudflare y despliega el Worker. El primer build es el
más lento; los siguientes reutilizan capas en caché.

Tras el primer deploy, esperá unos minutos antes de probar: Cloudflare
puede tardar en aprovisionar los containers. URL final:

```
https://explain-my-data.<tu-subdominio>.workers.dev
```

## CI/CD (GitHub Actions)

`.github/workflows/deploy.yml` despliega en cada push a `main` (y
manualmente vía *workflow_dispatch*). Requiere dos secrets en el
repositorio de GitHub:

| Secret | Cómo obtenerlo |
|--------|----------------|
| `CLOUDFLARE_API_TOKEN` | Dashboard → My Profile → API Tokens → Create Token (permisos: Account / Containers Edit, Workers Scripts Edit, Workers Bundles Edit) |
| `CLOUDFLARE_ACCOUNT_ID` | Dashboard → Workers & Pages → URL de la cuenta (o `npx wrangler whoami`) |

El workflow de CI (`.github/workflows/ci.yml`) corre `ruff` y `pytest`
en cada push/PR sin necesitar secrets.

## Configuración

| Variable | Default | Descripción |
|----------|---------|-------------|
| `EMD_MAX_UPLOAD_BYTES` | 20971520 (20MB) | Límite de upload; se configura en `worker/index.js` (`env` de `container.start()`) o en el Dockerfile |

CSVs de trabajo esperados: <5MB. El límite de 20MB es un techo con
holgura, no un objetivo.

## Costos y ciclo de vida

- **Workers Paid**: $5/mes (incluye Containers).
- Instancia `standard-1` (½ vCPU, 4 GiB, 8 GB de disco). Se elige en
  `worker/index.js` (`instance` de `container.start()`). El policy
  `durable_object` no acepta `basic`, `dev` ni `standard`; usar `lite`
  o `standard-1` a `standard-4`.
- Scale-to-zero a los 10 minutos de inactividad
  (`setInactivityTimeout` en `worker/index.js`); el primer request de
  un período en frío tarda mientras el contenedor arranca.
- El Worker siempre enruta al Durable Object `main`, así que hay una
  sola instancia. El policy `durable_object` no usa `max_instances`.

## Operación

```bash
npx wrangler containers list              # aplicaciones de containers
npx wrangler containers instances <ID>    # instancias de una aplicación
npx wrangler containers images list       # imágenes en el registro
npx wrangler tail                         # logs del Worker
```

Logs y métricas del container: Dashboard → Workers & Pages →
Containers.

## Troubleshooting

- **`unknown flag: --load` al construir**: falta el plugin buildx.
  Instalar con `sudo apt install docker-buildx` y verificar con
  `docker buildx version`.
- **`This account cannot create applications with the "default"
  scheduling policy`**: usar `scheduling_policy: "durable_object"` en
  `wrangler.jsonc` (ya configurado).
- **`unknown instance type` al iniciar**: el policy `durable_object`
  acepta `lite`, `standard-1`, `standard-2`, `standard-3`,
  `standard-4` o un objeto custom; no acepta `basic`, `dev` ni
  `standard`.
- **El Worker responde pero `/analyze` da error o 503**: el container
  aún se está aprovisionando (solo los primeros minutos tras el
  deploy) o está arrancando en frío. Esperar y reintentar.
- **El container reinicia por falta de memoria**: subir `instance` a
  `standard-2` o superior en `worker/index.js`.
- **Apple Silicon / ARM**: el build es `linux/amd64`; Docker Desktop
  lo maneja vía emulation. Si el build falla, verificar que la
  plataforma emulada esté habilitada en Docker Desktop.
- **Deploy desde CI falla en el build**: verificar que el token tenga
  permisos de Containers y que `CLOUDFLARE_ACCOUNT_ID` esté seteado.
- **Upload rechazado con 413**: supera `EMD_MAX_UPLOAD_BYTES`
  (default 20MB).
