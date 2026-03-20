/**
 * Shared fixtures for viewer Playwright tests.
 *
 * With Vite serving the frontend, Three.js resolves from node_modules
 * automatically — no CDN interception needed.
 */

import { test as base } from '@playwright/test';

export const test = base;

/**
 * Navigate to the viewer and wait for it to finish loading.
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

/** Set up CDN routes for contexts that bypass Vite (e.g., error-states test). */
export async function setupCdnRoutes(context) {
  // No-op: Vite handles all module resolution
}
