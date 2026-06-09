import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  base: "/python-lucide/search/",
  build: {
    outDir: "../docs/search",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1100,
  },
});
