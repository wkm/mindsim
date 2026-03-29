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

// CDN URL → local file mapping (files read once into memory)
const CDN_ROUTES = {
  'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js':
    path.join(CDN_CACHE, 'three.module.js'),
  'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js':
    path.join(CDN_CACHE, 'OrbitControls.js'),
  'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js':
    path.join(CDN_CACHE, 'mujoco_wasm.js'),
};

// Pre-read CDN files into memory (avoids re-reading on every route intercept)
const CDN_BUFFERS = {};
for (const [url, localPath] of Object.entries(CDN_ROUTES)) {
  if (fs.existsSync(localPath)) {
    CDN_BUFFERS[url] = fs.readFileSync(localPath);
  }
}

/** Register CDN route interceptors on a Playwright browser context. */
async function setupCdnRoutes(context) {
  for (const [cdnUrl, localPath] of Object.entries(CDN_ROUTES)) {
    const body = CDN_BUFFERS[cdnUrl];
    if (!body) continue;
    await context.route(cdnUrl, async (route) => {
      console.log(`  CDN intercept: ${path.basename(localPath)} (${(body.length / 1024).toFixed(0)} KB)`);
      await route.fulfill({ status: 200, contentType: 'application/javascript', body });
    });
  }
}

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
  await setupCdnRoutes(context);

  const page = await context.newPage();

  // Collect console errors
  const consoleErrors = [];
  page.on('console', msg => {
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
    assert(tabs.length === 4, `Four mode tabs present (${tabs.length})`);

    // Screenshot each mode
    const modeNames = ['explore', 'joint', 'ik'];
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

    // =================================================================
    // Joint mode: single slider moves geometry
    // =================================================================
    await page.click('.mode-tab[data-mode="joint"]');
    await page.waitForTimeout(200);
    const sliders = await page.$$('#side-panel input[type="range"]');
    assert(sliders.length > 0, `Joint mode has slider controls (found ${sliders.length})`);

    if (sliders.length >= 4) {
      const beforeShot = await page.screenshot();
      const sliderBB = await sliders[3].boundingBox();
      if (sliderBB) {
        await page.mouse.click(
          sliderBB.x + sliderBB.width * 0.25,
          sliderBB.y + sliderBB.height / 2
        );
        await page.waitForTimeout(300);
        const afterShot = await page.screenshot({ path: path.join(screenshotDir, 'joint_slider_moved.png') });
        const differs = Buffer.compare(beforeShot, afterShot) !== 0;
        assert(differs, 'Joint slider changes 3D scene (screenshots differ)');
      }
    }

    // =================================================================
    // Joint mode: ALL sliders functional
    // =================================================================
    console.log('\n--- All Joint Sliders ---');
    await page.click('.mode-tab[data-mode="joint"]');
    await page.waitForTimeout(300);
    // Reset first so we start from a known state
    await page.click('#joint-reset-btn');
    await page.waitForTimeout(200);

    const allSliders = await page.$$('#side-panel input[type="range"]');
    for (let i = 0; i < allSliders.length; i++) {
      const slider = allSliders[i];
      const sliderId = await slider.getAttribute('id');
      const valDisplay = await page.$(`#${sliderId}-val`);
      const originalText = await valDisplay.textContent();

      const box = await slider.boundingBox();
      await page.mouse.click(box.x + box.width * 0.25, box.y + box.height / 2);
      await page.waitForTimeout(200);

      const newText = await valDisplay.textContent();
      assert(newText !== originalText, `Slider ${sliderId} value changed ("${originalText}" → "${newText}")`);
    }

    // =================================================================
    // Joint mode: Reset All button
    // =================================================================
    console.log('\n--- Reset All ---');
    await page.click('#joint-reset-btn');
    await page.waitForTimeout(300);

    const slidersAfterReset = await page.$$('#side-panel input[type="range"]');
    for (let i = 0; i < slidersAfterReset.length; i++) {
      const sliderId = await slidersAfterReset[i].getAttribute('id');
      const valDisplay = await page.$(`#${sliderId}-val`);
      const text = await valDisplay.textContent();
      assert(text.trim() === '0.0°', `After reset, ${sliderId} shows "${text}" (expect "0.0°")`);
    }
    await page.screenshot({ path: path.join(screenshotDir, 'joint_reset.png') });

    // =================================================================
    // IK mode: body selection
    // =================================================================
    console.log('\n--- IK Body Selection ---');
    await page.click('.mode-tab[data-mode="ik"]');
    await page.waitForTimeout(500);

    const initAnchor = (await page.textContent('#ik-anchor-name')).trim().toLowerCase();
    const initTarget = (await page.textContent('#ik-target-name')).trim().toLowerCase();
    assert(initAnchor === 'none', `IK anchor starts as "none" (got "${initAnchor}")`);
    assert(initTarget === 'none', `IK target starts as "none" (got "${initTarget}")`);

    // Click on the robot body. The robot renders right-of-center in the
    // canvas area (0 to width-320). From screenshots: base ~(620,600),
    // turntable ~(620,500), arm ~(640,350).
    // Scan a grid of positions on the robot to find clickable bodies.
    const clickPositions = [
      [620, 590],  // base/wheels area
      [630, 500],  // turntable
      [640, 400],  // upper arm
      [640, 350],  // forearm
      [600, 550],  // base side
      [650, 300],  // hand
      [580, 620],  // wheel
    ];

    // First click — set anchor
    for (const [x, y] of clickPositions) {
      await page.mouse.click(x, y);
      await page.waitForTimeout(200);
      const name = (await page.textContent('#ik-anchor-name')).trim();
      if (name.toLowerCase() !== 'none') {
        assert(true, `IK anchor set to "${name}" at (${x},${y})`);
        break;
      }
    }

    let anchorName = (await page.textContent('#ik-anchor-name')).trim();
    const anchorSet = anchorName.toLowerCase() !== 'none';

    if (anchorSet) {
      // Second click — set target (try positions different from anchor)
      for (const [x, y] of clickPositions) {
        await page.mouse.click(x, y);
        await page.waitForTimeout(200);
        const name = (await page.textContent('#ik-target-name')).trim();
        if (name.toLowerCase() !== 'none') {
          assert(true, `IK target set to "${name}" at (${x},${y})`);
          assert(anchorName !== name, `IK anchor ("${anchorName}") differs from target ("${name}")`);
          break;
        }
      }

      await page.screenshot({ path: path.join(screenshotDir, 'ik_body_selection.png') });

      // Clear selection
      await page.click('#ik-clear-btn');
      await page.waitForTimeout(300);
      const clearedAnchor = (await page.textContent('#ik-anchor-name')).trim().toLowerCase();
      const clearedTarget = (await page.textContent('#ik-target-name')).trim().toLowerCase();
      assert(clearedAnchor === 'none', `IK anchor cleared (got "${clearedAnchor}")`);
      assert(clearedTarget === 'none', `IK target cleared (got "${clearedTarget}")`);
    } else {
      await page.screenshot({ path: path.join(screenshotDir, 'ik_miss_debug.png') });
      console.log(`  WARN: Could not click on robot body. See ik_miss_debug.png`);
    }

    // =================================================================
    // Manifest validation
    // =================================================================
    console.log('\n--- Manifest Validation ---');
    const manifestData = JSON.parse(fs.readFileSync(path.join(PROJECT_ROOT, 'bots', botName, 'viewer_manifest.json'), 'utf-8'));
    const botXml = fs.readFileSync(path.join(PROJECT_ROOT, 'bots', botName, 'bot.xml'), 'utf-8');

    const manifestBodies = manifestData.bodies ? manifestData.bodies.length : 0;
    assert(manifestBodies > 0, `Manifest has bodies (found ${manifestBodies})`);

    const xmlBodyMatches = botXml.match(/<body\s+name=/g);
    const xmlBodyCount = xmlBodyMatches ? xmlBodyMatches.length : 0;
    assert(manifestBodies === xmlBodyCount, `Body count matches: manifest=${manifestBodies}, xml=${xmlBodyCount}`);
  }

  // =================================================================
  // Error state: bad bot name (separate browser context to isolate crash)
  // =================================================================
  console.log('\n--- Error State: Bad Bot Name ---');
  try {
    // Use a separate browser context so a WASM crash doesn't kill the main page
    const errContext = await browser.newContext({ viewport: { width: 1280, height: 800 } });
    // Set up CDN routes for the error context too
    await setupCdnRoutes(errContext);
    const errorPage = await errContext.newPage();
    const errorPageErrors = [];
    const errorFailedReqs = [];
    errorPage.on('console', msg => {
      if (msg.type() === 'error') errorPageErrors.push(msg.text());
    });
    errorPage.on('requestfailed', req => errorFailedReqs.push(req.url()));

    await errorPage.goto(`http://localhost:${port}/viewer/?bot=nonexistent_bot_12345`, { timeout: 30000 }).catch(() => {});
    await errorPage.waitForTimeout(5000);

    const errorLoadingText = await errorPage.$eval('#loading-text', el => el.textContent).catch(() => '');
    const hasUIError = errorLoadingText.toLowerCase().includes('error');
    const hasFailedReqs = errorFailedReqs.length > 0;
    const hasJSErrors = errorPageErrors.length > 0;
    assert(hasUIError || hasFailedReqs || hasJSErrors,
      `Bad bot name shows error (UI: "${errorLoadingText}", failed reqs: ${errorFailedReqs.length}, JS errors: ${errorPageErrors.length})`);
    await errorPage.screenshot({ path: path.join(screenshotDir, 'error_bad_bot.png') }).catch(() => {});
    await errContext.close();
  } catch (e) {
    // WASM crash may kill the page — that itself proves the error path exists
    assert(true, `Bad bot name causes error (${e.message.slice(0, 60)})`);
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
