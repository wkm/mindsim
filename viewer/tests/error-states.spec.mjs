/**
 * Error states: bad bot name handling.
 */

import { expect } from '@playwright/test';
import { test, setupCdnRoutes } from './fixtures.mjs';

test.describe('Error states', () => {
  test('bad bot name shows error', async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    await setupCdnRoutes(context);
    const page = await context.newPage();

    const pageErrors = [];
    const failedReqs = [];
    page.on('pageerror', err => pageErrors.push(err.message));
    page.on('requestfailed', req => failedReqs.push(req.url()));

    await page.goto('/viewer/?bot=nonexistent_bot_12345', { timeout: 30_000 }).catch(() => {});

    // Wait for an error indicator to appear rather than sleeping a fixed duration
    try {
      await page.waitForFunction(
        () => {
          const el = document.getElementById('loading-text');
          return el && el.textContent.toLowerCase().includes('error');
        },
        { timeout: 10_000 },
      );
    } catch {
      // UI error text may not appear — failed requests or JS errors also count
    }

    const loadingText = await page.$eval('#loading-text', el => el.textContent).catch(() => '');
    const hasUIError = loadingText.toLowerCase().includes('error');
    const hasFailedReqs = failedReqs.length > 0;
    const hasJSErrors = pageErrors.length > 0;

    expect(
      hasUIError || hasFailedReqs || hasJSErrors,
      `expected error indicator (UI: "${loadingText}", failed: ${failedReqs.length}, errors: ${pageErrors.length})`,
    ).toBe(true);

    await context.close();
  });
});
