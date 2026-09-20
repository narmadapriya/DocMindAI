/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#020617",
          900: "#060b1d",
          850: "#091126",
          800: "#0d162b",
          700: "#15203a"
        },
        brand: {
          400: "#a879ff",
          500: "#8b5cf6",
          600: "#7c3aed"
        },
        cyanx: "#22d3ee",
        mint: "#22c55e"
      },
      boxShadow: {
        panel: "0 16px 50px rgba(0,0,0,.28)",
        glow: "0 0 26px rgba(124,58,237,.20)"
      }
    }
  },
  plugins: []
};
