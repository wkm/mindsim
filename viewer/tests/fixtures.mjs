/**
 * Shared fixtures for viewer Playwright tests.
 *
 * - Intercepts CDN requests and serves from local cache
 * - Provides a waitForViewer helper that waits for loading overlay to hide
 */

import { test as base } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CDN_CACHE = path.join(__dirname, '..', '.cdn_cache');

const CDN_ROUTES = {
  'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js':
    path.join(CDN_CACHE, 'three.module.js'),
  'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js':
    path.join(CDN_CACHE, 'OrbitControls.js'),
  'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js':
    path.join(CDN_CACHE, 'mujoco_wasm.js'),
};

// Pre-read CDN files into memory
const CDN_BUFFERS = {};
for (const [url, localPath] of Object.entries(CDN_ROUTES)) {
  try {
    CDN_BUFFERS[url] = fs.readFileSync(localPath);
  } catch {}
}

/** Register CDN route interceptors on a Playwright browser context. */
async function setupCdnRoutes(context) {
  for (const [cdnUrl, localPath] of Object.entries(CDN_ROUTES)) {
    const body = CDN_BUFFERS[cdnUrl];
    if (!body) continue;
    await context.route(cdnUrl, async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/javascript', body });
    });
  }
}

/**
 * Extended test fixture that sets up CDN interception and provides helpers.
 */
export const test = base.extend({
  // Auto-setup CDN routes for every test's browser context
  context: async ({ context }, use) => {
    await setupCdnRoutes(context);
    await use(context);
  },
});

export { setupCdnRoutes };

/**
 * Navigate to the viewer and wait for it to finish loading.
 * Returns true if the viewer initialized, false otherwise.
 */
export async function waitForViewer(page, bot = 'wheeler_arm') {
  await page.goto(`/viewer/?bot=${bot}`, { timeout: 90_000 });
  await page.waitForFunction(
    () => {
      const el = document.getElementById('loading');
      return el && el.style.display === 'none';
    },
    { timeout: 90_000 },
  );
}
