/**
 * Joint mode: sliders functional, reset button.
 */

import { expect } from '@playwright/test';
import { test, waitForViewer } from './fixtures.mjs';

/** Click a slider at 25% of its width to move it from center. */
async function clickSlider(page, slider) {
  const box = await slider.boundingBox();
  await page.mouse.click(box.x + box.width * 0.25, box.y + box.height / 2);
  await page.waitForTimeout(200);
}

/** Get the displayed value text for a slider by its ID. */
async function sliderValueText(page, slider) {
  const sliderId = await slider.getAttribute('id');
  return page.locator(`#${sliderId}-val`).textContent();
}

test.describe('Joint mode', () => {
  test.beforeEach(async ({ page }) => {
    await waitForViewer(page);
    await page.click('#mode-tabs .btn-ghost[data-mode="joint"]');
    await page.waitForTimeout(300);
  });

  test('has slider controls', async ({ page }) => {
    const sliders = page.locator('#side-panel input[type="range"]');
    await expect(sliders.first()).toBeVisible();
    expect(await sliders.count()).toBeGreaterThan(0);
  });

  test('all sliders change value when clicked', async ({ page }) => {
    await page.click('#joint-reset-btn');
    await page.waitForTimeout(200);

    const sliders = page.locator('#side-panel input[type="range"]');
    const count = await sliders.count();

    for (let i = 0; i < count; i++) {
      const slider = sliders.nth(i);
      const originalText = await sliderValueText(page, slider);

      await clickSlider(page, slider);

      const newText = await sliderValueText(page, slider);
      const sliderId = await slider.getAttribute('id');
      expect(newText, `slider ${sliderId}`).not.toBe(originalText);
    }
  });

  test('reset button zeroes all sliders', async ({ page }) => {
    // Move a slider first
    const firstSlider = page.locator('#side-panel input[type="range"]').first();
    await clickSlider(page, firstSlider);

    // Reset
    await page.click('#joint-reset-btn');
    await page.waitForTimeout(300);

    const sliders = page.locator('#side-panel input[type="range"]');
    const count = await sliders.count();
    for (let i = 0; i < count; i++) {
      const text = await sliderValueText(page, sliders.nth(i));
      const sliderId = await sliders.nth(i).getAttribute('id');
      expect(text.trim(), `${sliderId} after reset`).toBe('0.0°');
    }
  });
});
