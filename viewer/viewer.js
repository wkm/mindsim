/**
 * MindSim Viewer — Main entry point.
 *
 * Routes between four states based on URL params:
 *   /viewer/                  → landing page
 *   /viewer/?bot=X            → MuJoCo bot viewer
 *   /viewer/?component=X      → component browser (Three.js, no MuJoCo)
 *   /viewer/?cadsteps=X:Y     → CAD steps debugger (Three.js, no MuJoCo)
 */

// ---------------------------------------------------------------------------
// URL routing — decide which mode to launch
// ---------------------------------------------------------------------------
const params = new URLSearchParams(window.location.search);
const botName = params.get('bot');
const componentParam = params.get('component');
const cadstepsParam = params.get('cadsteps');

if (cadstepsParam) {
  // CAD steps debugger — hide landing, show top bar (side panel managed by cad-steps-mode)
  document.getElementById('landing').style.display = 'none';
  document.getElementById('top-bar').style.display = '';
  document.getElementById('side-panel').style.display = 'none';
  document.getElementById('mode-tabs').style.display = 'none';
  const cadParts = cadstepsParam.split(':');
  const fromBot = params.get('from');
  const botNameEl = document.getElementById('bot-name');
  if (cadParts[0] === 'component') {
    botNameEl.textContent = `${cadParts[1]} — ShapeScript`;
    botNameEl.style.cursor = 'pointer';
    botNameEl.addEventListener('click', () => {
      if (fromBot) {
        window.location.href = `?bot=${encodeURIComponent(fromBot)}`;
      } else {
        window.location.href = `?component=${encodeURIComponent(cadParts[1])}`;
      }
    });
  } else {
    botNameEl.textContent = `${cadParts[0]} — ShapeScript`;
    botNameEl.style.cursor = 'pointer';
    botNameEl.addEventListener('click', () => {
      window.location.href = `?bot=${encodeURIComponent(cadParts[0])}`;
    });
  }
  import('./cad-steps-mode.js').then(m => m.initCadSteps(cadstepsParam));

} else if (botName) {
  // Bot viewer — hide landing, show bot UI, load MuJoCo
  document.getElementById('landing').style.display = 'none';
  document.getElementById('loading').style.display = '';
  document.getElementById('top-bar').style.display = '';
  document.getElementById('side-panel').style.display = '';
  document.getElementById('bot-name').textContent = botName;
  import('./bot-viewer.js').then(m => m.initBotViewer(botName));

} else if (componentParam) {
  // Component browser — hide landing, show component UI
  document.getElementById('landing').style.display = 'none';
  document.getElementById('top-bar').style.display = '';
  document.getElementById('side-panel').style.display = '';
  document.getElementById('component-browser').style.display = 'block';
  document.getElementById('axis-gizmo').style.display = '';
  document.getElementById('bot-name').textContent = 'Components';
  // Show component browser navbar elements
  document.getElementById('cb-nav-divider').style.display = '';
  document.getElementById('cb-view-group').style.display = '';
  document.getElementById('cb-tools-group').style.display = '';
  import('./component-browser.js').then(m => m.initComponentBrowser(componentParam));

} else {
  // Landing page — already visible by default, hide everything else
  document.getElementById('component-browser').style.display = 'none';
}
