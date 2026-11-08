import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(fileURLToPath(import.meta.url));

export default defineConfig(({ command }) => {
  const env = loadEnv(command === "build" ? "production" : "development", projectRoot, "");
  const proxyTarget = env.VITE_API_PROXY_TARGET || "http://localhost:8001";

  // The API base is injected here rather than left to api.js's fallback, because
  // the two runtimes are not the same:
  //
  //   dev  -- the SPA is served by Vite on :5174 and the API is on :8001, so this
  //           must be an absolute cross-origin URL.
  //   prod -- the standalone app serves the SPA and the API from ONE port, so this
  //           must be empty, making every call same-origin. Baking in an absolute
  //           URL instead means the page at 127.0.0.1:8001 fetches
  //           localhost:8001, which is cross-origin, and WebKit blocks every call
  //           with a bare "Load failed" while curl and the API both look healthy.
  const apiBase = command === "serve" ? proxyTarget : "";

  return {
    plugins: [react()],
    define: {
      "import.meta.env.VITE_API_BASE": JSON.stringify(apiBase),
    },
    server: {
      port: 5174,
      strictPort: true,
      proxy: {
        "/api": {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
