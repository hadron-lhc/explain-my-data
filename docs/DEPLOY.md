# Deploy en Cloudflare

Arquitectura: un **Worker** proxy enruta todas las peticiones a un
**Container** (FastAPI + SPA servida en el mismo origen). Scale-to-zero
tras 10 minutos de inactividad; solo se cobra el tiempo con instancias
activas.

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
  plan de pago; incluye el uso del container `basic`).
- Docker corriendo localmente (`docker info` debe salir bien).
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
| `EMD_MAX_UPLOAD_BYTES` | 20971520 (20MB) | Límite de upload; se configura en `worker/index.js` (`envVars` del Container) o en el Dockerfile |

CSVs de trabajo esperados: <5MB. El límite de 20MB es un techo con
holgura, no un objetivo.

## Costos y ciclo de vida

- **Workers Paid**: $5/mes (incluye Containers).
- Instancia `basic` (¼ vCPU, 1GiB). Scale-to-zero a los 10 minutos de
  inactividad (`sleepAfter` en `worker/index.js`); el primer request
  de un período en frío tarda mientras el contenedor arranca
  (segundos).
- `max_instances: 1`: suficiente para uso personal; subirlo en
  `wrangler.jsonc` si hace falta concurrencia.

## Operación

```bash
npx wrangler containers list        # estado de las instancias
npx wrangler containers images list # imágenes en el registro
npx wrangler tail                   # logs del Worker
```

Logs y métricas del container: Dashboard → Workers & Pages →
Containers.

## Troubleshooting

- **El Worker responde pero `/analyze` da error**: el container aún se
  está aprovisionando (solo los primeros minutos tras el deploy).
  Esperar y reintentar.
- **Apple Silicon / ARM**: el build es `linux/amd64`; Docker Desktop
  lo maneja vía emulation. Si el build falla, verificar que la
  plataforma emulada esté habilitada en Docker Desktop.
- **Deploy desde CI falla en el build**: verificar que el token tenga
  permisos de Containers y que `CLOUDFLARE_ACCOUNT_ID` esté seteado.
- **Upload rechazado con 413**: supera `EMD_MAX_UPLOAD_BYTES`
  (default 20MB).
