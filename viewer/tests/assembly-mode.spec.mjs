/**
 * Assembly mode: step-through, show-all.
 */

import { expect } from '@playwright/test';
import { test, waitForViewer } from './fixtures.mjs';

test.describe('Assembly mode', () => {
  test.beforeEach(async ({ page }) => {
    await waitForViewer(page);
    await page.click('#mode-tabs .bp5-button[data-mode="assembly"]');
    await page.waitForTimeout(500);
  });

  test('step through all assembly steps', async ({ page }) => {
    // Go to step 1
    await page.$eval('#asm-step-slider', el => {
      el.value = 0;
      el.dispatchEvent(new Event('input'));
    });
    await page.waitForTimeout(300);

    const stepText = await page.textContent('#asm-step-val');
    expect(stepText).toContain('1 / ');

    // Extract total steps
    const total = parseInt(stepText.split('/')[1].trim());

    let prevShot = await page.screenshot();
    for (let step = 2; step <= total; step++) {
      await page.click('#asm-next-btn');
      await page.waitForTimeout(300);

      const text = await page.textContent('#asm-step-val');
      expect(text).toContain(`${step} / ${total}`);

      const currentShot = await page.screenshot();
      expect(prevShot.equals(currentShot), `step ${step} differs from ${step - 1}`).toBe(false);
      prevShot = currentShot;
    }
  });

  test('show all renders full robot', async ({ page }) => {
    await page.click('#asm-show-all-btn');
    await page.waitForTimeout(300);

    const shot = await page.screenshot();
    expect(shot.length).toBeGreaterThan(30_000);
  });
});
