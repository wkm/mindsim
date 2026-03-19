import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: '*.spec.mjs',
  timeout: 120_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: 'http://localhost:8765',
    viewport: { width: 1280, height: 800 },
    ignoreHTTPSErrors: true,
    launchOptions: {
      args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    },
  },
  webServer: {
    command: 'uv run mjpython main.py web --port 8765',
    cwd: '../..',
    port: 8765,
    timeout: 60_000,
    reuseExistingServer: !process.env.CI,
  },
  reporter: [['list']],
  retries: 0,
  workers: 1,  // serial — single server
});
