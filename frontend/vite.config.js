import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy /api requests to the FastAPI backend during development.
    // This avoids CORS entirely: the browser sees one origin (localhost:5173)
    // and Vite forwards the request internally to localhost:8000.
    // In production, a real reverse proxy (nginx, Caddy) does the same job.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
