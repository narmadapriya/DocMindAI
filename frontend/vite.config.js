import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  // GitHub Pages project site:
  // https://narmadapriya.github.io/DocMindAI/
  base: "/DocMindAI/",

  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,

    // Allows VS Code Dev Tunnel URLs
    allowedHosts: [".devtunnels.ms"],

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
    strictPort: true,
    allowedHosts: [".devtunnels.ms"],
  },

  build: {
    // Let Rollup determine the dependency graph.
    // Heavy PDF code remains lazy-loaded at the call site.
    sourcemap: false,
  },
});