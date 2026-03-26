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
  import('./cad-steps-mode.ts').then((m) => m.initCadSteps(cadstepsParam));
} else if (botName) {
  // Bot viewer — hide landing, show UI with Design/Sim tabs
  document.getElementById('landing').style.display = 'none';
  document.getElementById('top-bar').style.display = '';
  document.getElementById('bot-name').textContent = botName;

  // Build tab buttons in the navbar (replace mode-tabs content)
  const modeTabs = document.getElementById('mode-tabs');
  modeTabs.innerHTML = '';

  const designTab = document.createElement('button');
  designTab.className = 'btn-ghost active';
  designTab.dataset.tab = 'design';
  designTab.textContent = 'Design';
  modeTabs.appendChild(designTab);

  const simTab = document.createElement('button');
  simTab.className = 'btn-ghost';
  simTab.dataset.tab = 'sim';
  simTab.textContent = 'Sim';
  modeTabs.appendChild(simTab);

  // Show tree panel for Design mode
  const treePanel = document.getElementById('tree-panel');
  treePanel.style.display = 'block';
  const treePanelContent = document.getElementById('tree-content');

  // Side panel hidden in Design mode (only used by Sim/Explore)
  document.getElementById('side-panel').style.display = 'none';

  // Import Viewport3D and set up shared container
  import('./viewport3d.ts').then(async ({ Viewport3D }) => {
    const container = document.getElementById('canvas-container');

    // Create a perspective viewport for Design
    const designViewport = new Viewport3D(container, {
      cameraType: 'perspective',
      grid: true,
    });

    // Load Design viewer immediately
    const { initDesignViewer } = await import('./design-viewer.ts');
    const designCtx = await initDesignViewer(botName, designViewport, treePanelContent);

    // Start the design viewport animation loop
    designViewport.animate(() => {});

    // Track active tab and lazy-loaded Sim viewer
    type SimHandle = { pause(): void; resume(): void };
    let activeTab: 'design' | 'sim' = 'design';
    let simHandle: SimHandle | null = null;

    function updateTabUI() {
      designTab.classList.toggle('active', activeTab === 'design');
      simTab.classList.toggle('active', activeTab === 'sim');
    }

    async function switchTab(tab: 'design' | 'sim') {
      if (tab === activeTab) return;
      activeTab = tab;
      updateTabUI();

      if (tab === 'sim') {
        // Hide Design scene
        designViewport.scene.visible = false;
        treePanel.style.display = 'none';

        // Show Sim UI
        document.getElementById('side-panel').style.display = '';
        document.getElementById('loading').style.display = '';

        if (!simHandle) {
          // First time — lazy-load MuJoCo viewer
          const { initBotViewer } = await import('./bot-viewer.ts');
          simHandle = await initBotViewer(botName);
        } else {
          simHandle.resume();
        }

        document.getElementById('loading').style.display = 'none';
      } else {
        // Pause Sim
        simHandle?.pause();

        // Hide Sim scene (the MuJoCo root group)
        document.getElementById('side-panel').style.display = 'none';
        document.getElementById('bot-tools-group').style.display = 'none';

        // Show Design scene
        designViewport.scene.visible = true;
        treePanel.style.display = 'block';
        designCtx.syncVisibility();
        designViewport.resize();
      }
    }

    designTab.addEventListener('click', () => switchTab('design'));
    simTab.addEventListener('click', () => switchTab('sim'));
  });
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
  import('./component-browser.ts').then((m) => m.initComponentBrowser(componentParam));
} else {
  // Landing page — already visible by default, hide everything else
  document.getElementById('component-browser').style.display = 'none';
}
