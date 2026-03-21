/**
 * MindSim CAD Steps Viewer — split-pane code debugger for ShapeScript.
 *
 * Left pane: 3D viewport (Viewport3D) showing body solid + tool overlay.
 * Right pane: syntax-highlighted code editor showing all ShapeScript ops.
 * Click a line to scrub. Current step highlighted with blue bar.
 * Resizable panes with draggable divider.
 *
 * Smart camera:
 *   Holistic mode — frames once on the final step's bounding box; stays fixed.
 *   Isolate mode — re-frames on the current step's geometry with animation.
 *
 * URL: ?cadsteps=<bot_name>:<body_name>
 */

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { clearGroup } from './utils.js';
import { Viewport3D } from './viewport3d.js';

// Edge rendering — matches bot-viewer.js and component-browser.js
const EDGE_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x000000, transparent: true, opacity: 0.6,
});
const EDGE_THRESHOLD = 28; // degrees — only sharp edges

// Op type → color
const OP_COLORS = {
  create: 0x2B95D6,  // blue — new primitive
  cut:    0xDB3737,  // red — subtraction
  union:  0x0F9960,  // green — addition
  locate: 0x9179F2,  // purple — spatial placement
};

// ── Styles for the code editor pane ──
const EDITOR_STYLES = `
  .cs-layout {
    position: absolute; top: 40px; bottom: 0; left: 0; right: 0;
    display: flex; z-index: 1;
  }
  .cs-viewport {
    flex: 0 0 60%; min-width: 200px; position: relative; overflow: hidden;
  }
  .cs-divider {
    flex: 0 0 6px; background: var(--border); cursor: col-resize;
    transition: background 0.15s; z-index: 10;
  }
  .cs-divider:hover, .cs-divider.dragging { background: var(--primary); }
  .cs-editor-pane {
    flex: 1 1 40%; min-width: 200px; display: flex; flex-direction: column;
    background: #1C2127; overflow: hidden;
  }

  /* Toolbar row inside editor pane */
  .cs-toolbar {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-bottom: 1px solid #30363D;
    flex-shrink: 0;
  }
  .cs-toolbar .cs-step-count {
    font-family: var(--font-mono); font-size: 12px; color: #ABB3BF;
    white-space: nowrap;
  }
  .cs-toolbar input[type="range"] {
    flex: 1; height: 4px; -webkit-appearance: none; appearance: none;
    background: #30363D; border-radius: 2px; outline: none;
  }
  .cs-toolbar input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 12px; height: 12px;
    border-radius: 50%; background: var(--primary); cursor: pointer;
  }
  .cs-toolbar label {
    font-size: 11px; color: #738694; cursor: pointer;
    display: flex; align-items: center; gap: 4px; white-space: nowrap;
  }
  .cs-toolbar input[type="checkbox"] {
    width: 12px; height: 12px; accent-color: var(--primary); cursor: pointer;
  }

  /* Code block */
  .cs-code-scroll {
    flex: 1; overflow-y: auto; overflow-x: hidden;
  }
  .cs-code-scroll::-webkit-scrollbar { width: 8px; }
  .cs-code-scroll::-webkit-scrollbar-track { background: #1C2127; }
  .cs-code-scroll::-webkit-scrollbar-thumb { background: #30363D; border-radius: 4px; }
  .cs-code-scroll::-webkit-scrollbar-thumb:hover { background: #3E4A55; }

  .cs-code {
    display: table; width: 100%; border-collapse: collapse;
    font-family: 'Input Mono Narrow', 'SF Mono', 'Menlo', monospace;
    font-size: 12px; line-height: 20px;
  }
  .cs-line {
    display: table-row; cursor: pointer; transition: background 0.08s;
  }
  .cs-line:hover { background: rgba(255,255,255,0.04); }
  .cs-line.cs-active {
    background: rgba(43,149,214,0.12);
  }
  .cs-gutter {
    display: table-cell; width: 44px; padding: 0 8px 0 0;
    text-align: right; color: #3E4A55; user-select: none;
    vertical-align: top; white-space: nowrap;
  }
  .cs-line.cs-active .cs-gutter { color: #738694; }
  .cs-line-bar {
    display: table-cell; width: 3px; padding: 0;
    vertical-align: top;
  }
  .cs-line.cs-active .cs-line-bar { background: #2B95D6; }
  .cs-line-content {
    display: table-cell; padding: 0 12px 0 8px;
    white-space: pre; color: #ABB3BF;
    vertical-align: top;
  }

  /* Sub-program section */
  .cs-line.cs-subprogram { background: rgba(255,255,255,0.015); }
  .cs-line.cs-subprogram .cs-line-content { padding-left: 24px; }
  .cs-line.cs-group-header .cs-line-content {
    color: #738694; font-weight: 600; font-size: 11px;
    padding-top: 6px;
  }

  /* Syntax coloring */
  .ss-prim { color: #2B95D6; }      /* Box, Cylinder, Sphere */
  .ss-cut { color: #DB3737; }       /* Cut */
  .ss-fuse { color: #0F9960; }      /* Fuse */
  .ss-loc { color: #9179F2; }       /* Locate — purple, matches 3D view */
  .ss-call { color: #9179F2; }      /* Call */
  .ss-pre { color: #738694; }       /* Prebuilt */
  .ss-comment { color: #5C7080; font-style: italic; }
  .ss-num { color: #C87619; }
  .ss-ref { color: #8A9BA8; }

  /* Hover parameter highlight */
  .ss-param-hover { background: rgba(255,255,255,0.1); border-radius: 2px; }

  /* Body switcher dropdown in navbar */
  .cs-body-select {
    background: transparent; border: 1px solid var(--border); border-radius: var(--radius-sm);
    color: var(--foreground); font-family: var(--font); font-size: 12px;
    padding: 2px 6px; height: 28px; cursor: pointer; outline: none;
  }
  .cs-body-select:hover { background: var(--secondary); }
`;

export async function initCadSteps(param) {
  const parts = param.split(':');
  if (parts.length < 2) {
    console.error('cadsteps param must be bot:body or component:name, got:', param);
    return;
  }

  // Detect mode: "component:OV5647" vs "wheeler_arm:base"
  if (parts[0] === 'component') {
    const viewer = new CadStepsViewer(null, null, parts[1]);
    await viewer.init();
  } else {
    const viewer = new CadStepsViewer(parts[0], parts[1], null);
    await viewer.init();
  }
}

class CadStepsViewer {
  constructor(botName, bodyName, componentName = null) {
    this.botName = botName;
    this.bodyName = bodyName;
    this.componentName = componentName;
    this.isComponentMode = componentName !== null;
    this.steps = [];
    this.currentStep = 0;
    this.stlLoader = new STLLoader();
    this.stlCache = {};      // step index → BufferGeometry
    this.toolStlCache = {};  // step index → BufferGeometry (tool solid)
    this.viewMode = 'outcome';  // 'outcome' | 'inputs'
    this.showContext = true;     // body-so-far overlay
    this.showFinal = false;
    this._ghostGeometry = null;           // cached final-step geometry for ghost
    this._leftBasis = 60; // percent
    this._hasFramedHolistic = false;      // true once we've framed on the final body
  }

  async init() {
    // Inject styles
    const styleEl = document.createElement('style');
    styleEl.textContent = EDITOR_STYLES;
    document.head.appendChild(styleEl);

    // Show loading overlay
    const loadingEl = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');
    loadingEl.style.display = '';
    const targetLabel = this.isComponentMode ? this.componentName : this.bodyName;
    loadingText.textContent = `Building CAD steps for ${targetLabel}...`;

    // Hide the default side panel — we use our own layout
    const sidePanel = document.getElementById('side-panel');
    if (sidePanel) sidePanel.style.display = 'none';

    this.allBodies = [];
    const fetches = [this._fetchSteps()];
    if (!this.isComponentMode) fetches.push(this._fetchBodies());
    await Promise.all(fetches);

    this._buildLayout();
    this._setupViewport();
    this._buildNavExtras();
    this._buildEditorPane();

    loadingEl.style.display = 'none';

    // Frame camera on final step geometry, then start animation
    await this._frameOnFinalBody();

    this.viewport.animate(() => {
      // per-frame hook — nothing needed currently
    });

    if (this.steps.length > 0) {
      await this._showStep(this.steps.length - 1);
    }
  }

  // ── Layout ──

  _buildLayout() {
    // Hide the default canvas container — we use our own viewport inside the split layout
    const container = document.getElementById('canvas-container');
    container.style.display = 'none';

    this.layoutEl = document.createElement('div');
    this.layoutEl.className = 'cs-layout';

    // Left: 3D viewport
    this.viewportEl = document.createElement('div');
    this.viewportEl.className = 'cs-viewport';
    this.viewportEl.style.flexBasis = this._leftBasis + '%';

    // Divider
    this.dividerEl = document.createElement('div');
    this.dividerEl.className = 'cs-divider';
    this._setupDividerDrag();

    // Right: code editor pane
    this.editorPaneEl = document.createElement('div');
    this.editorPaneEl.className = 'cs-editor-pane';

    this.layoutEl.appendChild(this.viewportEl);
    this.layoutEl.appendChild(this.dividerEl);
    this.layoutEl.appendChild(this.editorPaneEl);

    document.body.appendChild(this.layoutEl);
  }

  _setupDividerDrag() {
    let startX, startLeftBasis;
    const onMouseDown = (e) => {
      e.preventDefault();
      startX = e.clientX;
      startLeftBasis = this._leftBasis;
      this.dividerEl.classList.add('dragging');
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    };
    const onMouseMove = (e) => {
      const dx = e.clientX - startX;
      const totalW = this.layoutEl.clientWidth;
      const newBasis = startLeftBasis + (dx / totalW) * 100;
      this._leftBasis = Math.max(20, Math.min(80, newBasis));
      this.viewportEl.style.flexBasis = this._leftBasis + '%';
      this.viewport.resize();
    };
    const onMouseUp = () => {
      this.dividerEl.classList.remove('dragging');
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    this.dividerEl.addEventListener('mousedown', onMouseDown);
  }

  // ── Viewport3D setup ──

  _setupViewport() {
    this.viewport = new Viewport3D(this.viewportEl, {
      cameraType: 'perspective',
      grid: true,
    });
    this.meshGroup = this.viewport.addGroup('body');
    this.toolGroup = this.viewport.addGroup('tool');
    this.contextGroup = this.viewport.addGroup('context');  // body-so-far for create/locate steps
    this.ghostGroup = this.viewport.addGroup('ghost');
  }

  // ── Smart camera: frame on final body once for holistic mode ──

  async _frameOnFinalBody() {
    if (this.steps.length === 0) return;
    const finalGeometry = await this._loadStepSTL(this.steps.length - 1);
    if (finalGeometry) {
      this.viewport.frameOnGeometry(finalGeometry);
      this._hasFramedHolistic = true;
    }
  }

  // ── Nav bar extras (body switcher) ──

  _buildNavExtras() {
    const topBar = document.getElementById('top-bar');
    if (!topBar) return;

    // Add body switcher to the nav bar if multiple bodies
    if (this.allBodies.length > 1) {
      const sep = document.createElement('span');
      sep.className = 'nav-sep';
      topBar.appendChild(sep);

      const select = document.createElement('select');
      select.className = 'cs-body-select';
      for (const name of this.allBodies) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        if (name === this.bodyName) opt.selected = true;
        select.appendChild(opt);
      }
      select.addEventListener('change', () => {
        window.location.href = `?cadsteps=${encodeURIComponent(this.botName)}:${encodeURIComponent(select.value)}`;
      });
      topBar.appendChild(select);
    }
  }

  // ── Editor pane ──

  _buildEditorPane() {
    this.editorPaneEl.innerHTML = '';

    // Toolbar: slider + step count + tool toggle
    const toolbar = document.createElement('div');
    toolbar.className = 'cs-toolbar';

    this.stepCountEl = document.createElement('span');
    this.stepCountEl.className = 'cs-step-count';
    this.stepCountEl.textContent = `${this.steps.length} / ${this.steps.length}`;
    toolbar.appendChild(this.stepCountEl);

    this.slider = document.createElement('input');
    this.slider.type = 'range';
    this.slider.min = '0';
    this.slider.max = String(Math.max(0, this.steps.length - 1));
    this.slider.value = String(this.steps.length - 1);
    this.slider.addEventListener('input', () => this._onSliderChange());
    toolbar.appendChild(this.slider);

    // Segmented control: Outcome | Inputs
    const modeGroup = document.createElement('div');
    modeGroup.style.cssText = 'display:flex; border:1px solid #30363D; border-radius:4px; overflow:hidden; flex-shrink:0;';

    const segBtnBase = 'border:none; padding:3px 10px; font:500 11px system-ui,-apple-system,sans-serif; cursor:pointer; transition:background 0.12s,color 0.12s;';
    const segActive = 'background:#30363D; color:#E8EDF0;';
    const segInactive = 'background:transparent; color:#738694;';

    this._outcomeBtn = document.createElement('button');
    this._outcomeBtn.textContent = 'Outcome';
    this._outcomeBtn.style.cssText = segBtnBase + segActive;
    this._outcomeBtn.addEventListener('click', async () => {
      this.viewMode = 'outcome';
      this._updateModeButtons();
      await this._showStep(this.currentStep);
    });

    this._inputsBtn = document.createElement('button');
    this._inputsBtn.textContent = 'Inputs';
    this._inputsBtn.style.cssText = segBtnBase + segInactive;
    this._inputsBtn.addEventListener('click', async () => {
      this.viewMode = 'inputs';
      this._updateModeButtons();
      await this._showStep(this.currentStep);
    });

    modeGroup.appendChild(this._outcomeBtn);
    modeGroup.appendChild(this._inputsBtn);
    toolbar.appendChild(modeGroup);

    // "Context" checkbox — body-so-far overlay (on by default)
    const contextLabel = document.createElement('label');
    const contextCb = document.createElement('input');
    contextCb.type = 'checkbox';
    contextCb.checked = true;
    contextCb.addEventListener('change', async (e) => {
      this.showContext = e.target.checked;
      await this._showStep(this.currentStep);
    });
    contextLabel.appendChild(contextCb);
    contextLabel.appendChild(document.createTextNode('Context'));
    toolbar.appendChild(contextLabel);

    // "Final" checkbox — ghost of the completed body (off by default)
    const finalLabel = document.createElement('label');
    const finalCb = document.createElement('input');
    finalCb.type = 'checkbox';
    finalCb.checked = false;
    finalCb.addEventListener('change', async (e) => {
      this.showFinal = e.target.checked;
      await this._rebuildGhost();
      await this._showStep(this.currentStep);
    });
    finalLabel.appendChild(finalCb);
    finalLabel.appendChild(document.createTextNode('Final'));
    toolbar.appendChild(finalLabel);

    this.editorPaneEl.appendChild(toolbar);

    // Code scroll area
    this.codeScrollEl = document.createElement('div');
    this.codeScrollEl.className = 'cs-code-scroll';

    const codeTable = document.createElement('div');
    codeTable.className = 'cs-code';
    codeTable.style.padding = '4px 0';

    this.lineEls = [];
    let currentGroup = null;

    for (const step of this.steps) {
      // Group header for CallOp sub-programs
      if (step.group && step.group !== currentGroup) {
        const headerLine = document.createElement('div');
        headerLine.className = 'cs-line cs-group-header';
        headerLine.innerHTML = `
          <span class="cs-gutter"></span>
          <span class="cs-line-bar"></span>
          <span class="cs-line-content"># ${this._escapeHtml(step.group)}</span>
        `;
        codeTable.appendChild(headerLine);
      }
      currentGroup = step.group || null;

      const line = document.createElement('div');
      line.className = 'cs-line';
      if (step.group) line.classList.add('cs-subprogram');
      line.dataset.index = step.index;

      const scriptText = step.script || step.label;
      const highlighted = this._highlightScript(scriptText);

      line.innerHTML = `
        <span class="cs-gutter">${step.index + 1}</span>
        <span class="cs-line-bar"></span>
        <span class="cs-line-content">${highlighted}</span>
      `;

      line.addEventListener('click', () => {
        this.slider.value = String(step.index);
        this._onSliderChange();
      });

      // Parameter hover highlighting
      const contentEl = line.querySelector('.cs-line-content');
      contentEl.addEventListener('mouseover', (e) => {
        const span = e.target.closest('.ss-param-target');
        if (span) span.classList.add('ss-param-hover');
      });
      contentEl.addEventListener('mouseout', (e) => {
        const span = e.target.closest('.ss-param-target');
        if (span) span.classList.remove('ss-param-hover');
      });

      codeTable.appendChild(line);
      this.lineEls.push(line);
    }

    this.codeScrollEl.appendChild(codeTable);
    this.editorPaneEl.appendChild(this.codeScrollEl);

    // Keyboard shortcuts — prevent code pane from capturing arrow keys
    this.editorPaneEl.addEventListener('keydown', (e) => {
      if (e.key.startsWith('Arrow')) e.preventDefault();
    });

    document.addEventListener('keydown', async (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (this.currentStep > 0) {
          this.slider.value = String(this.currentStep - 1);
          this._onSliderChange();
        }
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        if (this.currentStep < this.steps.length - 1) {
          this.slider.value = String(this.currentStep + 1);
          this._onSliderChange();
        }
      } else if (e.key === 'f' && !e.ctrlKey && !e.metaKey) {
        // Toggle follow mode and immediately frame on current step
        e.preventDefault();
        const entering = !this.viewport.isFollowMode();
        this.viewport.setFollowMode(entering);
        if (entering) {
          const geom = await this._loadStepSTL(this.currentStep);
          if (geom) this.viewport.frameIfFollowing(geom);
        }
      }
    });
  }

  _escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  _updateModeButtons() {
    const segBtnBase = 'border:none; padding:3px 10px; font:500 11px system-ui,-apple-system,sans-serif; cursor:pointer; transition:background 0.12s,color 0.12s;';
    const segActive = 'background:#30363D; color:#E8EDF0;';
    const segInactive = 'background:transparent; color:#738694;';
    this._outcomeBtn.style.cssText = segBtnBase + (this.viewMode === 'outcome' ? segActive : segInactive);
    this._inputsBtn.style.cssText = segBtnBase + (this.viewMode === 'inputs' ? segActive : segInactive);
  }

  _highlightScript(script) {
    let s = this._escapeHtml(script);

    // Order matters: comments first, then keywords, then refs/numbers.
    // We use a token-replace approach to avoid double-matching.
    const tokens = [];
    let tokenId = 0;
    const placeholder = (cls, text) => {
      const key = `\x00T${tokenId++}\x00`;
      tokens.push({ key, html: `<span class="${cls}">${text}</span>` });
      return key;
    };

    // Comments: # to end of string
    s = s.replace(/(#[^\x00]*)$/gm, (m) => placeholder('ss-comment', m));

    // Keywords — order: specific ops first
    s = s.replace(/\b(Cut)\b/g, (m) => placeholder('ss-cut', m));
    s = s.replace(/\b(Fuse)\b/g, (m) => placeholder('ss-fuse', m));
    s = s.replace(/\b(Locate)\b/g, (m) => placeholder('ss-loc', m));
    s = s.replace(/\b(Call)\b/g, (m) => placeholder('ss-call', m));
    s = s.replace(/\b(Prebuilt)\b/g, (m) => placeholder('ss-pre', m));
    s = s.replace(/\b(Box|Cylinder|Sphere)\b/g, (m) => placeholder('ss-prim', m));

    // Parameter values (pos=(...), r=..., w=..., l=..., h=...) — wrap as hoverable targets
    s = s.replace(/((?:pos|rot|r|w|l|h|d)=\([^)]*\)|(?:pos|rot|r|w|l|h|d)=[\d.\-]+)/g,
      (m) => placeholder('ss-param-target', m));

    // Numbers (remaining, not inside tokens)
    s = s.replace(/\b(\d+\.?\d*)\b/g, (m) => placeholder('ss-num', m));

    // Reference IDs (word_digits pattern)
    s = s.replace(/\b(\w+_\d+)\b/g, (m) => placeholder('ss-ref', m));

    // Replace placeholders with actual HTML
    for (const { key, html } of tokens) {
      s = s.replace(key, html);
    }

    return s;
  }

  // ── Scroll sync ──

  _scrollToLine(idx) {
    const lineEl = this.lineEls[idx];
    if (!lineEl || !this.codeScrollEl) return;

    const scrollTop = this.codeScrollEl.scrollTop;
    const scrollH = this.codeScrollEl.clientHeight;
    const lineTop = lineEl.offsetTop - this.codeScrollEl.offsetTop;
    const lineH = lineEl.offsetHeight;

    // Keep current line roughly centered, but only scroll if off-screen
    if (lineTop < scrollTop + 40 || lineTop + lineH > scrollTop + scrollH - 40) {
      this.codeScrollEl.scrollTo({
        top: lineTop - scrollH / 3,
        behavior: 'smooth',
      });
    }
  }

  // ── Data fetching ──

  async _fetchSteps() {
    const url = this.isComponentMode
      ? `/api/components/${encodeURIComponent(this.componentName)}/shapescript`
      : `/api/bots/${this.botName}/body/${this.bodyName}/cad-steps`;
    const resp = await fetch(url);
    if (!resp.ok) {
      console.error('Failed to fetch CAD steps:', resp.status, await resp.text());
      return;
    }
    const data = await resp.json();
    this.steps = data.steps || [];
  }

  async _fetchBodies() {
    const url = `/bots/${this.botName}/viewer_manifest.json`;
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const manifest = await resp.json();
        this.allBodies = (manifest.bodies || []).map(b => b.name);
      }
    } catch { /* manifest not available */ }
  }

  // ── Step navigation ──

  async _onSliderChange() {
    const idx = parseInt(this.slider.value);
    await this._showStep(idx);
  }

  async _showStep(idx) {
    if (idx < 0 || idx >= this.steps.length) return;
    this.currentStep = idx;
    const step = this.steps[idx];

    // Update toolbar
    this.slider.value = String(idx);
    this.stepCountEl.textContent = `${idx + 1} / ${this.steps.length}`;

    // Highlight current line in code editor
    for (let i = 0; i < this.lineEls.length; i++) {
      this.lineEls[i].classList.toggle('cs-active', i === idx);
    }
    this._scrollToLine(idx);

    // Clear mesh groups
    const clearMG = (g) => {
      while (g.children.length > 0) {
        const c = g.children[0]; g.remove(c);
        if (c.isLineSegments && c.geometry) c.geometry.dispose();
        if (c.material && c.material !== EDGE_MATERIAL) c.material.dispose();
      }
    };
    clearMG(this.meshGroup);
    clearMG(this.contextGroup);
    clearGroup(this.toolGroup);

    const isBooleanStep = (step.op === 'cut' || step.op === 'union');

    const addTransparentMesh = (group, geom, color, opacity) => {
      const mat = new THREE.MeshPhysicalMaterial({
        color, transparent: true, opacity, roughness: 0.8, depthWrite: false, side: THREE.DoubleSide,
      });
      group.add(new THREE.Mesh(geom, mat));
      const edgeMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: opacity + 0.05 });
      const edges = new THREE.EdgesGeometry(geom, EDGE_THRESHOLD);
      const lines = new THREE.LineSegments(edges, edgeMat);
      lines.raycast = () => {};
      group.add(lines);
    };

    const addOpaqueMesh = (group, geom, color) => {
      group.add(new THREE.Mesh(geom, new THREE.MeshPhysicalMaterial({
        color, roughness: 0.5, metalness: 0.1, clearcoat: 0.1,
      })));
      const edges = new THREE.EdgesGeometry(geom, EDGE_THRESHOLD);
      const lines = new THREE.LineSegments(edges, EDGE_MATERIAL);
      lines.raycast = () => {};
      group.add(lines);
    };

    if (this.viewMode === 'inputs' && isBooleanStep && idx > 0) {
      // Inputs mode for cut/fuse: target (before-body) + tool overlay
      const targetGeom = await this._loadStepSTL(idx - 1);
      if (targetGeom) addTransparentMesh(this.meshGroup, targetGeom, 0xCED9E0, 0.35);
      const toolGeom = await this._loadSTL(this.toolStlCache, idx, 'tool-stl');
      const toolColor = step.op === 'cut' ? 0xDB3737 : 0x0F9960;
      if (toolGeom) addTransparentMesh(this.meshGroup, toolGeom, toolColor, 0.35);

    } else if (this.viewMode === 'inputs' && step.op === 'locate') {
      // Inputs mode for locate: pre-move faint gray + result purple
      const preGeom = await this._loadSTL(this.toolStlCache, idx, 'tool-stl');
      if (preGeom) addTransparentMesh(this.meshGroup, preGeom, 0x738694, 0.3);
      const resultGeom = await this._loadStepSTL(idx);
      if (resultGeom) addOpaqueMesh(this.meshGroup, resultGeom, 0x9179F2);

    } else {
      // Outcome mode (or inputs on create/call/prebuilt — same thing)
      const geom = await this._loadStepSTL(idx);
      if (!geom) return;
      let color = 0xCED9E0; // neutral gray for cut/fuse
      if (step.op === 'create') color = 0x2B95D6;
      else if (step.op === 'locate') color = 0x9179F2;
      addOpaqueMesh(this.meshGroup, geom, color);
    }

    // Context overlay: body-so-far for create/locate steps (not for cut/fuse where body IS the result)
    const needsContext = this.showContext && (step.op === 'create' || step.op === 'locate');
    if (needsContext) {
      const ctxIdx = this._findLastBodyStep(idx);
      if (ctxIdx >= 0) {
        const ctxGeom = await this._loadStepSTL(ctxIdx);
        if (ctxGeom) {
          const ctxMat = new THREE.MeshPhysicalMaterial({
            color: 0xCED9E0, transparent: true, opacity: 0.25,
            roughness: 0.8, depthWrite: false,
          });
          this.contextGroup.add(new THREE.Mesh(ctxGeom, ctxMat));
          const ctxEdges = new THREE.EdgesGeometry(ctxGeom, EDGE_THRESHOLD);
          this.contextGroup.add(new THREE.LineSegments(ctxEdges,
            new THREE.LineBasicMaterial({ color: 0x5C7080, transparent: true, opacity: 0.2 })));
        }
      }
    }
    this.contextGroup.visible = needsContext;

    // Re-apply section plane to new geometry if section is active
    if (this.viewport._secOn) {
      this.viewport._applySection();
    }

    // "Final" ghost
    this.ghostGroup.visible = this.showFinal;

    // Follow mode: auto-frame on current step geometry
    const frameGeom = await this._loadStepSTL(idx);
    if (frameGeom) this.viewport.frameIfFollowing(frameGeom);

    // Prefetch adjacent steps
    this._prefetch(idx - 1);
    this._prefetch(idx + 1);
  }

  /** Find the most recent cut/fuse step before `idx` — that's the body-so-far. */
  _findLastBodyStep(idx) {
    for (let i = idx - 1; i >= 0; i--) {
      const op = this.steps[i]?.op;
      if (op === 'cut' || op === 'union') return i;
    }
    return -1;
  }

  async _rebuildGhost() {
    clearGroup(this.ghostGroup);

    if (!this.showFinal || this.steps.length === 0) return;

    // Load (or reuse cached) final step geometry
    if (!this._ghostGeometry) {
      const finalIdx = this.steps.length - 1;
      this._ghostGeometry = await this._loadStepSTL(finalIdx);
    }
    if (!this._ghostGeometry) return;

    // Ghost solid — very transparent, no depth write
    const ghostMat = new THREE.MeshPhysicalMaterial({
      color: 0xCED9E0,
      transparent: true,
      opacity: 0.06,
      roughness: 1.0,
      depthWrite: false,
    });
    this.ghostGroup.add(new THREE.Mesh(this._ghostGeometry, ghostMat));

    // Ghost edges
    const ghostEdgeMat = new THREE.LineBasicMaterial({
      color: 0x5C7080,
      transparent: true,
      opacity: 0.15,
    });
    const edges = new THREE.EdgesGeometry(this._ghostGeometry, EDGE_THRESHOLD);
    const edgeLines = new THREE.LineSegments(edges, ghostEdgeMat);
    edgeLines.raycast = () => {};
    this.ghostGroup.add(edgeLines);
  }

  // ── STL loading ──

  async _loadSTL(cache, idx, suffix) {
    if (idx < 0 || idx >= this.steps.length) return null;
    if (cache[idx]) return cache[idx];

    const url = this.isComponentMode
      ? `/api/components/${encodeURIComponent(this.componentName)}/shapescript/${idx}/${suffix}`
      : `/api/bots/${this.botName}/body/${this.bodyName}/cad-steps/${idx}/${suffix}`;
    return new Promise((resolve) => {
      this.stlLoader.load(url, (geometry) => {
        geometry.computeVertexNormals();
        cache[idx] = geometry;
        resolve(geometry);
      }, undefined, () => resolve(null));
    });
  }

  _loadStepSTL(idx) { return this._loadSTL(this.stlCache, idx, 'stl'); }

  _prefetch(idx) {
    if (idx < 0 || idx >= this.steps.length) return;
    if (!this.stlCache[idx]) this._loadStepSTL(idx);
    if (this.steps[idx]?.has_tool && !this.toolStlCache[idx]) {
      this._loadSTL(this.toolStlCache, idx, 'tool-stl');
    }
  }
}
