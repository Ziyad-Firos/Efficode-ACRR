import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Proxy API routes to the Flask backend during development.
      // Must stay in sync with the port in backend/app/main.py (8000).
      '/review':  { target: 'http://localhost:8000', changeOrigin: true },
      '/refactor': { target: 'http://localhost:8000', changeOrigin: true },
      '/health':  { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
