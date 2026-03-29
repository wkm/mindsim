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

  const dfmTab = document.createElement('button');
  dfmTab.className = 'btn-ghost';
  dfmTab.dataset.tab = 'dfm';
  dfmTab.textContent = 'Assembly';
  modeTabs.appendChild(dfmTab);

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
    const TREE_PANEL_WIDTH = 280;
    const SIDE_PANEL_WIDTH = 320;

    /** Offset the canvas container so it doesn't overlap the tree or side panel. */
    function updateCanvasLayout(treePanelVisible: boolean, sidePanelVisible = false) {
      const left = treePanelVisible ? TREE_PANEL_WIDTH : 0;
      const right = sidePanelVisible ? SIDE_PANEL_WIDTH : 0;
      container.style.left = `${left}px`;
      container.style.right = `${right}px`;
    }

    // Initial layout: Design tab is active, tree panel visible
    updateCanvasLayout(true);

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

    // Track active tab and lazy-loaded viewers
    type SimHandle = { pause(): void; resume(): void };
    type DFMHandle = { pause(): void; resume(): void; dispose(): void; rootEl: HTMLElement };
    let activeTab: 'design' | 'dfm' | 'sim' = 'design';
    let simHandle: SimHandle | null = null;
    let dfmHandle: DFMHandle | null = null;

    const sidePanel = document.getElementById('side-panel');

    function updateTabUI() {
      designTab.classList.toggle('active', activeTab === 'design');
      dfmTab.classList.toggle('active', activeTab === 'dfm');
      simTab.classList.toggle('active', activeTab === 'sim');
    }

    async function switchTab(tab: 'design' | 'dfm' | 'sim') {
      if (tab === activeTab) return;
      const prevTab = activeTab;
      activeTab = tab;
      updateTabUI();

      const simModeTabs = document.getElementById('sim-mode-tabs');

      // Deactivate previous tab
      if (prevTab === 'sim') {
        simHandle?.pause();
        document.getElementById('bot-tools-group').style.display = 'none';
        if (simModeTabs) simModeTabs.style.display = 'none';
      } else if (prevTab === 'dfm') {
        dfmHandle?.pause();
        if (dfmHandle) dfmHandle.rootEl.style.display = 'none';
      } else {
        // was design
        designViewport.setVisible(false);
      }

      // Hide common panels between switches
      sidePanel.style.display = 'none';
      treePanel.style.display = 'none';

      if (tab === 'design') {
        // Show Design scene
        container.style.display = '';
        updateCanvasLayout(true);
        designViewport.setVisible(true);
        treePanel.style.display = 'block';
        designCtx.syncVisibility();
        designViewport.resize();
      } else if (tab === 'dfm') {
        // Assembly viewer owns its own layout (nav/detail/viewport split).
        // Hide the shared canvas container; the assembly viewer's rootEl
        // is mounted as a sibling that fills the same area.
        container.style.display = 'none';

        if (!dfmHandle) {
          const { initAssemblyViewer } = await import('./assembly-viewer.ts');
          dfmHandle = await initAssemblyViewer(botName);
          // Mount the assembly root as a sibling of canvas-container
          const parent = container.parentElement!;
          dfmHandle.rootEl.style.cssText = 'position: absolute; top: 40px; left: 0; right: 0; bottom: 0;';
          parent.appendChild(dfmHandle.rootEl);
        } else {
          dfmHandle.rootEl.style.display = '';
          dfmHandle.resume();
        }
      } else if (tab === 'sim') {
        // Show Sim UI
        container.style.display = '';
        updateCanvasLayout(false);
        sidePanel.style.display = '';
        document.getElementById('loading').style.display = '';
        if (simModeTabs) simModeTabs.style.display = '';

        if (!simHandle) {
          const { initBotViewer } = await import('./bot-viewer.ts');
          simHandle = await initBotViewer(botName);
        } else {
          simHandle.resume();
        }

        document.getElementById('loading').style.display = 'none';
      }
    }

    designTab.addEventListener('click', () => switchTab('design'));
    dfmTab.addEventListener('click', () => switchTab('dfm'));
    simTab.addEventListener('click', () => switchTab('sim'));
  });
} else if (componentParam) {
  // Component browser — hide landing, show unified layout (tree + canvas + side panel)
  document.getElementById('landing').style.display = 'none';
  document.getElementById('top-bar').style.display = '';
  document.getElementById('side-panel').style.display = '';
  document.getElementById('tree-panel').style.display = 'block';
  document.getElementById('axis-gizmo').style.display = '';
  document.getElementById('bot-name').textContent = 'Components';
  // Offset canvas for tree panel (same as Design mode)
  document.getElementById('canvas-container').style.left = '280px';
  // View presets available via 1-7 keyboard shortcuts (Viewport3D)
  import('./component-browser.ts').then((m) => m.initComponentBrowser(componentParam));
} else {
  // Landing page — already visible by default, hide everything else
  const cbEl = document.getElementById('component-browser');
  if (cbEl) cbEl.style.display = 'none';
}
