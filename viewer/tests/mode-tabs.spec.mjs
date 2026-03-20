/**
 * Mode tab switching and side panel population.
 */

import { expect } from '@playwright/test';
import { test, waitForViewer } from './fixtures.mjs';

const MODES = ['explore', 'joint', 'assembly', 'ik'];

test.describe('Mode tabs', () => {
  test.beforeEach(async ({ page }) => {
    await waitForViewer(page);
  });

  for (const mode of MODES) {
    test(`${mode} tab activates and populates side panel`, async ({ page }) => {
      await page.click(`#mode-tabs .bp5-button[data-mode="${mode}"]`);
      await page.waitForTimeout(500);

      const tab = page.locator(`#mode-tabs .bp5-button[data-mode="${mode}"]`);
      await expect(tab).toHaveClass(/bp5-active/);

      const panel = page.locator('#side-panel');
      const html = await panel.innerHTML();
      expect(html.trim().length).toBeGreaterThan(0);
    });
  }
});
