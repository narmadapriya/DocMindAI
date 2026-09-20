import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false,
        timeout: 600000,
        proxyTimeout: 600000,
      },
    },
  },

  preview: {
    host: "127.0.0.1",
    port: 4173,
  },

  build: {
    // Let Rollup determine the dependency graph. The previous broad manualChunks
    // rules split mutually-dependent packages and created circular chunk warnings.
    // Heavy PDF code is lazy-loaded at the call site instead.
    sourcemap: false,
  },
});
