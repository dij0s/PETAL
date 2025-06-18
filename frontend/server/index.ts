// @ts-expect-error: Bun APIs are not fully supported in TypeScript tooling yet
import { serve } from "bun";
import { join } from "path";

// @ts-expect-error: Bun extension, not in standard TS types
const DIST_DIR = join(import.meta.dir, "../dist");

// use env variables with fallback defaults
const BACKEND_API_URL = process.env.BACKEND_API_URL || "http://localhost:8000";
const PORT = process.env.PORT ? Number(process.env.PORT) : 3000;

serve({
  idleTimeout: 120,
  async fetch(req) {
    const url = new URL(req.url);

    // proxy API SSE streaming separately
    if (url.pathname === "/api/stream") {
      const apiUrl = BACKEND_API_URL + "/api/stream" + url.search;
      const apiRes = await fetch(apiUrl, {
        method: req.method,
        headers: req.headers,
      });

      // set headers for streaming
      const headers = new Headers(apiRes.headers);
      headers.set("Content-Type", "text/event-stream");
      headers.set("Cache-Control", "no-cache");
      headers.set("Connection", "keep-alive");

      return new Response(apiRes.body, {
        status: apiRes.status,
        headers,
      });
    }

    // proxy API requests to Python backend
    if (url.pathname.startsWith("/api/")) {
      const apiUrl = BACKEND_API_URL + url.pathname + url.search;
      const bodyText = await req.text();
      const apiReq = new Request(apiUrl, {
        method: req.method,
        headers: req.headers,
        body: bodyText,
      });

      const apiRes = await fetch(apiReq);
      return new Response(apiRes.body, {
        status: apiRes.status,
        headers: apiRes.headers,
      });
    }

    // serve static files from Vite build
    const filePath = url.pathname === "/" ? "/index.html" : url.pathname;
    try {
      // @ts-expect-error: Bun global
      const file = Bun.file(DIST_DIR + filePath);
      if (await file.exists()) {
        return new Response(file);
      }
      // fallback to index.html for SPA routing
      // @ts-expect-error: Bun global
      const indexFile = Bun.file(DIST_DIR + "/index.html");
      return new Response(indexFile);
    } catch {
      return new Response("Not found", { status: 404 });
    }
  },
  port: PORT,
});
console.log(`[LOG] Serving frontend build on port ${PORT}`);
