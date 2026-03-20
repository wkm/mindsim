/**
 * Viewer initialization: loads, renders canvas, no JS errors.
 */

import { expect } from '@playwright/test';
import { test, waitForViewer } from './fixtures.mjs';

test.describe('Viewer initialization', () => {
  test('loads without JS errors', async ({ page }) => {
    const pageErrors = [];
    page.on('pageerror', err => pageErrors.push(err.message));

    await waitForViewer(page);
    expect(pageErrors).toHaveLength(0);
  });

  test('canvas element exists and has rendered content', async ({ page }) => {
    await waitForViewer(page);

    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();

    // A rendered 3D scene compresses to a larger PNG than a blank canvas
    const screenshot = await canvas.screenshot();
    expect(screenshot.length).toBeGreaterThan(10_000);
  });

  test('bot name is displayed', async ({ page }) => {
    await waitForViewer(page);
    await expect(page.locator('#bot-name')).toHaveText('wheeler_arm');
  });

  test('four mode tabs are present', async ({ page }) => {
    await waitForViewer(page);
    const tabs = page.locator('#mode-tabs .btn-ghost');
    await expect(tabs).toHaveCount(4);
  });
});
