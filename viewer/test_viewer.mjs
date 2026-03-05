/**
 * Headless browser screenshot tests for the MindSim 3D viewer.
 *
 * Starts a local HTTP server, loads the viewer in headless Chromium via
 * Playwright, validates that MuJoCo WASM + Three.js initialize correctly,
 * and captures screenshots of each mode.
 *
 * CDN assets (Three.js, MuJoCo WASM) are intercepted via Playwright route()
 * and served from a local cache to work in sandboxed/offline environments.
 *
 * Usage:  node viewer/test_viewer.mjs [--bot NAME] [--port PORT]
 * Output: viewer/test_screenshots/*.png
 */

import { chromium } from 'playwright';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, '..');
const CDN_CACHE = path.join(__dirname, '.cdn_cache');

// Parse args
const args = process.argv.slice(2);
const botName = args.includes('--bot') ? args[args.indexOf('--bot') + 1] : 'wheeler_arm';
const port = args.includes('--port') ? parseInt(args[args.indexOf('--port') + 1]) : 8765;
const screenshotDir = path.join(__dirname, 'test_screenshots');

// CDN URL → local file mapping
const CDN_ROUTES = {
  'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js':
    path.join(CDN_CACHE, 'three.module.js'),
  'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js':
    path.join(CDN_CACHE, 'OrbitControls.js'),
  'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js':
    path.join(CDN_CACHE, 'mujoco_wasm.js'),
};

// -- Helpers -----------------------------------------------------------------

function startServer(root, listenPort) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      let filePath = path.join(root, decodeURIComponent(req.url.split('?')[0]));
      if (filePath.endsWith('/')) filePath += 'index.html';

      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes = {
        '.html': 'text/html',
        '.js': 'application/javascript',
        '.mjs': 'application/javascript',
        '.json': 'application/json',
        '.stl': 'application/octet-stream',
        '.xml': 'application/xml',
        '.css': 'text/css',
        '.png': 'image/png',
      };

      fs.readFile(filePath, (err, data) => {
        if (err) {
          res.writeHead(404);
          res.end(`Not found: ${req.url}`);
          return;
        }
        res.writeHead(200, {
          'Content-Type': mimeTypes[ext] || 'application/octet-stream',
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(data);
      });
    });
    server.listen(listenPort, () => resolve(server));
    server.on('error', reject);
  });
}

let passed = 0;
let failed = 0;
const failures = [];

function assert(condition, name) {
  if (condition) {
    passed++;
    console.log(`  PASS: ${name}`);
  } else {
    failed++;
    failures.push(name);
    console.log(`  FAIL: ${name}`);
  }
}

// -- Main --------------------------------------------------------------------

async function main() {
  console.log(`\nMindSim Viewer Screenshot Tests`);
  console.log(`Bot: ${botName}, Port: ${port}\n`);

  // Pre-flight: check meshes
  const meshDir = path.join(PROJECT_ROOT, 'bots', botName, 'meshes');
  const stlFiles = fs.readdirSync(meshDir).filter(f => f.endsWith('.stl'));
  assert(stlFiles.length > 0, `Bot "${botName}" has STL meshes (found ${stlFiles.length})`);

  const firstStl = fs.readFileSync(path.join(meshDir, stlFiles[0]));
  assert(firstStl.length > 500, `STL "${stlFiles[0]}" is real data (${firstStl.length} bytes), not LFS pointer`);

  // Pre-flight: check CDN cache
  for (const [url, localPath] of Object.entries(CDN_ROUTES)) {
    const exists = fs.existsSync(localPath);
    const basename = path.basename(localPath);
    assert(exists, `CDN cache has ${basename}`);
    if (!exists) {
      console.log(`    Missing: ${localPath}`);
      console.log(`    Download: curl -sL "${url}" -o "${localPath}"`);
    }
  }
  if (failed > 0) {
    console.log('\nCDN cache incomplete. Run the download commands above, then retry.');
    process.exit(1);
  }

  // Start server
  const server = await startServer(PROJECT_ROOT, port);
  console.log(`\nHTTP server on http://localhost:${port}`);

  // Ensure screenshot dir
  fs.mkdirSync(screenshotDir, { recursive: true });

  // Launch browser
  console.log(`Using browser: ${chromium.executablePath()}`);
  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-gpu'],
  });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    ignoreHTTPSErrors: true,
  });

  // Intercept CDN requests and serve from local cache
  for (const [cdnUrl, localPath] of Object.entries(CDN_ROUTES)) {
    await context.route(cdnUrl, async (route) => {
      const body = fs.readFileSync(localPath);
      console.log(`  CDN intercept: ${path.basename(localPath)} (${(body.length / 1024).toFixed(0)} KB)`);
      await route.fulfill({
        status: 200,
        contentType: 'application/javascript',
        body,
      });
    });
  }

  const page = await context.newPage();

  // Collect console messages and errors
  const consoleMsgs = [];
  const consoleErrors = [];
  page.on('console', msg => {
    consoleMsgs.push(`[${msg.type()}] ${msg.text()}`);
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });

  const pageErrors = [];
  page.on('pageerror', err => pageErrors.push(err.message));

  // Log failed network requests
  page.on('requestfailed', req => {
    console.log(`  Request FAILED: ${req.url()} — ${req.failure()?.errorText}`);
  });

  // Navigate
  const url = `http://localhost:${port}/viewer/?bot=${botName}`;
  console.log(`\nLoading ${url} ...`);
  await page.goto(url, { timeout: 90000 });
  console.log(`  Page navigated`);

  // Wait for loading overlay to disappear (MuJoCo + meshes loaded)
  console.log('Waiting for viewer to initialize (MuJoCo WASM + mesh loading)...');
  let viewerReady = false;
  try {
    await page.waitForFunction(
      () => {
        const el = document.getElementById('loading');
        return el && el.style.display === 'none';
      },
      { timeout: 90000 }
    );
    viewerReady = true;
    assert(true, 'Viewer initialized (loading overlay hidden)');
  } catch {
    const loadingText = await page.$eval('#loading-text', el => el.textContent).catch(() => 'unknown');
    assert(false, `Viewer initialized (stuck at: "${loadingText}")`);
    await page.screenshot({ path: path.join(screenshotDir, 'error_state.png') });
    console.log(`  Saved error state screenshot`);
    if (pageErrors.length) {
      console.log(`\n  Page errors:`);
      pageErrors.forEach(e => console.log(`    ${e}`));
    }
    if (consoleErrors.length) {
      console.log(`\n  Console errors:`);
      consoleErrors.forEach(e => console.log(`    ${e}`));
    }
  }

  // Check for JS errors
  assert(pageErrors.length === 0, `No uncaught JS errors (got ${pageErrors.length})`);
  if (pageErrors.length > 0) {
    pageErrors.forEach(e => console.log(`    Error: ${e}`));
  }

  // Only run visual/interactive tests if the viewer initialized
  if (viewerReady) {
    // Verify canvas exists
    const canvasExists = await page.$('canvas');
    assert(canvasExists !== null, 'Three.js canvas element exists');

    if (canvasExists) {
      // Check canvas has rendered content by taking a screenshot of just the
      // canvas area and checking pixel variance. readPixels doesn't work
      // reliably in headless Chromium's software renderer.
      const canvasShot = await canvasExists.screenshot();
      // A blank/uniform canvas compresses to a very small PNG (<5KB).
      // A rendered 3D scene with geometry, grid, and lighting is much larger.
      const canvasShotSize = canvasShot.length;
      assert(canvasShotSize > 10000, `Canvas has rendered content (screenshot ${(canvasShotSize/1024).toFixed(0)} KB, expect >10 KB)`);
    }

    // Verify UI elements
    const botNameText = await page.$eval('#bot-name', el => el.textContent);
    assert(botNameText === botName, `Bot name displayed correctly ("${botNameText}")`);

    const tabs = await page.$$('.mode-tab');
    assert(tabs.length === 3, `Three mode tabs present (${tabs.length})`);

    // Screenshot each mode
    const modeNames = ['joint', 'assembly', 'ik'];
    for (const mode of modeNames) {
      await page.click(`.mode-tab[data-mode="${mode}"]`);
      await page.waitForTimeout(500);

      const isActive = await page.$eval(
        `.mode-tab[data-mode="${mode}"]`,
        el => el.classList.contains('active')
      );
      assert(isActive, `${mode} mode tab activates`);

      const panelHTML = await page.$eval('#side-panel', el => el.innerHTML.trim());
      assert(panelHTML.length > 0, `${mode} mode populates side panel`);

      const ssPath = path.join(screenshotDir, `${mode}_mode.png`);
      await page.screenshot({ path: ssPath });
      const ssSize = fs.statSync(ssPath).size;
      console.log(`  Screenshot: ${ssPath} (${(ssSize / 1024).toFixed(0)} KB)`);
    }

    // Joint mode slider test — move shoulder_pitch (index 3) for visible arm movement
    await page.click('.mode-tab[data-mode="joint"]');
    await page.waitForTimeout(200);
    const sliders = await page.$$('#side-panel input[type="range"]');
    assert(sliders.length > 0, `Joint mode has slider controls (found ${sliders.length})`);

    if (sliders.length >= 4) {
      // Take before screenshot for comparison
      const beforeShot = await page.screenshot();

      // Move shoulder_pitch slider (index 3) — visibly tilts the arm
      const sliderBB = await sliders[3].boundingBox();
      if (sliderBB) {
        await page.mouse.click(
          sliderBB.x + sliderBB.width * 0.25,
          sliderBB.y + sliderBB.height / 2
        );
        await page.waitForTimeout(300);
        const ssPath = path.join(screenshotDir, 'joint_slider_moved.png');
        const afterShot = await page.screenshot({ path: ssPath });
        console.log(`  Screenshot after slider move: ${ssPath}`);

        // Verify the 3D scene actually changed (screenshots differ)
        const differs = Buffer.compare(beforeShot, afterShot) !== 0;
        assert(differs, 'Joint slider changes 3D scene (screenshots differ)');
      }
    }

    // Assembly mode step-through test
    await page.click('.mode-tab[data-mode="assembly"]');
    await page.waitForTimeout(500);

    // Step 1 should show something (base body)
    const asmStep1Shot = await page.screenshot();
    const asmStep1Size = asmStep1Shot.length;

    // Click "Show All" to see the full robot
    const showAllBtn = await page.$('#asm-show-all-btn');
    if (showAllBtn) {
      await showAllBtn.click();
      await page.waitForTimeout(300);
      const showAllShot = await page.screenshot({ path: path.join(screenshotDir, 'assembly_show_all.png') });
      assert(showAllShot.length > 30000, `Assembly "Show All" renders full robot (${(showAllShot.length/1024).toFixed(0)} KB)`);
    }

    // Click "Next" a few times and verify step counter advances
    const nextBtn = await page.$('#asm-next-btn');
    if (nextBtn) {
      await nextBtn.click();
      await page.waitForTimeout(300);
      const stepText = await page.$eval('#asm-step-val', el => el.textContent);
      assert(stepText.includes('2'), `Assembly step advances to 2 (got "${stepText}")`);

      // Take screenshot at step 2
      await page.screenshot({ path: path.join(screenshotDir, 'assembly_step2.png') });
    }
  }

  // Summary
  console.log(`\n${'='.repeat(50)}`);
  console.log(`Results: ${passed} passed, ${failed} failed`);
  if (failures.length > 0) {
    console.log(`\nFailures:`);
    failures.forEach(f => console.log(`  - ${f}`));
  }
  if (consoleErrors.length > 0) {
    console.log('\nBrowser console errors:');
    consoleErrors.forEach(e => console.log(`  ${e}`));
  }
  console.log('');

  // Cleanup
  await browser.close();
  server.close();

  process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
  console.error('Test runner error:', err);
  process.exit(2);
});
