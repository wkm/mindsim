import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.mjs',
  timeout: 120_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: 'http://localhost:5174',
    viewport: { width: 1280, height: 800 },
    ignoreHTTPSErrors: true,
    launchOptions: {
      args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
  webServer: [
    {
      command: 'uv run mjpython main.py web --port 8765 --no-open',
      cwd: '../..',
      port: 8765,
      timeout: 60_000,
      reuseExistingServer: !process.env.CI,
    },
    {
      command: 'API_PORT=8765 pnpm exec vite --port 5174',
      cwd: '../..',
      port: 5174,
      timeout: 15_000,
      reuseExistingServer: !process.env.CI,
    },
  ],
  reporter: [['list']],
  retries: 0,
  workers: 1,
});
