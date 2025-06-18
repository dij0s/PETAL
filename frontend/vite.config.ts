import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import { VitePluginFonts } from "vite-plugin-fonts";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePluginFonts({
      google: {
        families: [
          {
            name: "Funnel Display",
            styles: "wght@400;600",
          },
        ],
      },
    }),
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          "react-vendor": [
            "react",
            "react-dom",
            "react-i18next",
            "react-markdown",
          ],
          mapping: ["ol", "proj4"],
          vendors: [
            "i18next",
            "remark-gfm",
            "@fortawesome/react-fontawesome",
            "@fortawesome/free-solid-svg-icons",
          ],
        },
      },
    },
  },
});
