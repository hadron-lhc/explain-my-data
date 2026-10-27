import { DurableObject } from "cloudflare:workers";

const PORT = 8000;
const INACTIVITY_TIMEOUT_MS = 10 * 60 * 1000;
const HEALTH_TIMEOUT_MS = 30_000;
const HEALTH_POLL_INTERVAL_MS = 300;

export class ExplainMyData extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);

    if (ctx.container?.running) {
      void ctx.blockConcurrencyWhile(() =>
        ctx.container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS)
      );
    }
  }

  async ensureContainer() {
    const container = this.ctx.container;

    if (!container) {
      throw new Error("No container is configured for this Durable Object.");
    }

    if (!container.running) {
      container.start({
        image: container.images.base,
        instance: "standard-1",
        enableInternet: false,
        env: {
          EMD_MAX_UPLOAD_BYTES: "20971520",
        },
      });

      await container.setInactivityTimeout(INACTIVITY_TIMEOUT_MS);
      this.ctx.waitUntil(
        container.monitor().catch((error) => {
          console.error("Container stopped:", error);
        })
      );
    }

    return container;
  }

  async waitForHealth(container) {
    const port = container.getTcpPort(PORT);
    const deadline = Date.now() + HEALTH_TIMEOUT_MS;
    let lastError;

    while (Date.now() < deadline) {
      try {
        const response = await port.fetch("http://container/health");

        if (response.ok) {
          return;
        }
      } catch (error) {
        lastError = error;
      }

      await new Promise((resolve) =>
        setTimeout(resolve, HEALTH_POLL_INTERVAL_MS)
      );
    }

    throw new Error(
      `Container did not become healthy: ${lastError?.message ?? "timeout"}`
    );
  }

  async fetch(request) {
    try {
      const container = await this.ensureContainer();
      await this.waitForHealth(container);

      const containerUrl = request.url.replace(/^https:/, "http:");
      return container.getTcpPort(PORT).fetch(containerUrl, request);
    } catch (error) {
      console.error("Container request failed:", error);
      return new Response(
        `Container unavailable: ${
          error instanceof Error ? error.message : String(error)
        }`,
        { status: 503 }
      );
    }
  }
}

export default {
  async fetch(request, env) {
    const container = env.EXPLAIN_MY_DATA.getByName("main");
    return container.fetch(request);
  },
};
