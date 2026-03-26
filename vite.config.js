import { defineConfig } from 'vite';

const apiPort = process.env.API_PORT || 8081;

export default defineConfig({
  root: '.',
  publicDir: false,
  server: {
    open: '/viewer/',
    proxy: {
      '/api': `http://localhost:${apiPort}`,
    },
  },
});
