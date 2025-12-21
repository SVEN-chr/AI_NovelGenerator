import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#0b0f1a",
        foreground: "#e6e9f5",
        accent: "#6366f1",
        muted: "#1f2433",
        card: "#121826"
      }
    }
  },
  plugins: []
};

export default config;
