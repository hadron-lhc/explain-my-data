// Worker proxy para Cloudflare Containers (D1).
// Enruta todas las peticiones al contenedor FastAPI (que a su vez
// sirve la SPA en el mismo origen). Una sola instancia con
// scale-to-zero: el primer request de un período en frío tarda
// mientras el contenedor arranca.
import { Container } from "@cloudflare/containers";

export class ExplainMyData extends Container {
  defaultPort = 8000;
  // Scale-to-zero tras 10 minutos de inactividad ( Workers Paid:
  // solo se cobra el tiempo con instancias activas).
  sleepAfter = "10m";
}

export default {
  async fetch(request, env) {
    const container = env.EXPLAIN_MY_DATA.getByName("main");
    return container.fetch(request);
  },
};
