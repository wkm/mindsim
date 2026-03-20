/**
 * MindSim CAD Steps Viewer — step-by-step body solid construction debugger.
 *
 * Shows each intermediate CAD boolean operation as a separate step,
 * with a slider to scrub through the construction sequence. Tool solids
 * (the shape being cut or unioned) are shown as transparent overlays.
 *
 * URL: ?cadsteps=<bot_name>:<body_name>
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { clearGroup } from './utils.js';

const SIDE_PANEL_WIDTH = 320;

// Edge rendering — matches bot-viewer.js and component-browser.js
const EDGE_MATERIAL = new THREE.LineBasicMaterial({
  color: 0x000000, transparent: true, opacity: 0.6,
});
const EDGE_THRESHOLD = 28; // degrees — only sharp edges

// Op type → color (for step list dots)
const OP_COLORS = {
  create: 0x2B95D6,  // blue
  cut:    0xDB3737,  // red
  union:  0x0F9960,  // green
};

export async function initCadSteps(param) {
  const [botName, bodyName] = param.split(':');
  if (!botName || !bodyName) {
    console.error('cadsteps param must be bot:body, got:', param);
    return;
  }

  const viewer = new CadStepsViewer(botName, bodyName);
  await viewer.init();
}

class CadStepsViewer {
  constructor(botName, bodyName) {
    this.botName = botName;
    this.bodyName = bodyName;
    this.steps = [];
    this.currentStep = 0;
    this.stlLoader = new STLLoader();
    this.stlCache = {};      // step index → BufferGeometry
    this.toolStlCache = {};  // step index → BufferGeometry (tool solid)
    this.meshGroup = new THREE.Group();   // result solid + its edges
    this.toolGroup = new THREE.Group();   // tool solid + its edges
    this.showTool = true;
  }

  async init() {
    // Show loading overlay — CAD step generation is slow on first hit
    const loadingEl = document.getElementById('loading');
    const loadingText = document.getElementById('loading-text');
    loadingEl.style.display = '';
    loadingText.textContent = `Building CAD steps for ${this.bodyName}...`;

    this._setupThreeJS();
    this.allBodies = [];
    await Promise.all([this._fetchSteps(), this._fetchBodies()]);

    loadingEl.style.display = 'none';

    this._buildUI();
    this._animate();
    if (this.steps.length > 0) {
      await this._showStep(this.steps.length - 1);
    }
  }

  _setupThreeJS() {
    const container = document.getElementById('canvas-container');
    container.style.right = SIDE_PANEL_WIDTH + 'px';

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xF5F8FA);

    const viewWidth = window.innerWidth - SIDE_PANEL_WIDTH;
    this.camera = new THREE.PerspectiveCamera(45, viewWidth / window.innerHeight, 0.0001, 10);
    this.camera.position.set(0.08, 0.08, 0.1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(viewWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.update();

    // Lighting — match bot viewer
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
    dirLight.position.set(0.3, 0.5, 0.4);
    this.scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-0.3, -0.2, -0.4);
    this.scene.add(fillLight);

    // Grid
    const grid = new THREE.GridHelper(0.3, 30, 0xCED9E0, 0xE8EDF0);
    grid.rotation.x = Math.PI / 2; // XY plane
    this.scene.add(grid);

    this.scene.add(this.meshGroup);
    this.scene.add(this.toolGroup);

    // Handle resize
    window.addEventListener('resize', () => {
      const w = window.innerWidth - SIDE_PANEL_WIDTH;
      const h = window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
  }

  async _fetchSteps() {
    const url = `/api/bots/${this.botName}/body/${this.bodyName}/cad-steps`;
    const resp = await fetch(url);
    if (!resp.ok) {
      console.error('Failed to fetch CAD steps:', resp.status, await resp.text());
      return;
    }
    const data = await resp.json();
    this.steps = data.steps || [];
  }

  async _fetchBodies() {
    // Load viewer manifest to get list of all bodies for this bot
    const url = `/bots/${this.botName}/viewer_manifest.json`;
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const manifest = await resp.json();
        this.allBodies = (manifest.bodies || []).map(b => b.name);
      }
    } catch { /* manifest not available — no body switcher */ }
  }

  _buildUI() {
    const panel = document.getElementById('side-panel');
    panel.innerHTML = '';

    // Title
    const title = document.createElement('h2');
    title.textContent = 'CAD Steps';
    panel.appendChild(title);

    // Body name badge
    const badge = document.createElement('div');
    badge.className = 'prop-badge body-badge';
    badge.textContent = `${this.botName} / ${this.bodyName}`;
    panel.appendChild(badge);

    // Step info box
    this.stepInfoEl = document.createElement('div');
    this.stepInfoEl.className = 'step-info';
    this.stepInfoEl.innerHTML = '<div class="step-title">Loading...</div>';
    panel.appendChild(this.stepInfoEl);

    // Step slider
    const sliderGroup = document.createElement('div');
    sliderGroup.className = 'slider-group';

    const sliderLabel = document.createElement('div');
    sliderLabel.className = 'slider-label';
    this.stepCountEl = document.createElement('span');
    this.stepCountEl.className = 'name';
    this.stepCountEl.textContent = 'Step';
    this.stepValueEl = document.createElement('span');
    this.stepValueEl.className = 'value';
    this.stepValueEl.textContent = '0';
    sliderLabel.appendChild(this.stepCountEl);
    sliderLabel.appendChild(this.stepValueEl);
    sliderGroup.appendChild(sliderLabel);

    this.slider = document.createElement('input');
    this.slider.type = 'range';
    this.slider.min = '0';
    this.slider.max = String(Math.max(0, this.steps.length - 1));
    this.slider.value = String(this.steps.length - 1);
    this.slider.addEventListener('input', () => this._onSliderChange());
    sliderGroup.appendChild(this.slider);
    panel.appendChild(sliderGroup);

    // Prev / Next buttons
    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display: flex; gap: 8px; margin-bottom: 12px;';

    const prevBtn = document.createElement('button');
    prevBtn.className = 'btn';
    prevBtn.textContent = 'Prev';
    prevBtn.addEventListener('click', () => {
      if (this.currentStep > 0) {
        this.slider.value = String(this.currentStep - 1);
        this._onSliderChange();
      }
    });
    btnRow.appendChild(prevBtn);

    const nextBtn = document.createElement('button');
    nextBtn.className = 'btn';
    nextBtn.textContent = 'Next';
    nextBtn.addEventListener('click', () => {
      if (this.currentStep < this.steps.length - 1) {
        this.slider.value = String(this.currentStep + 1);
        this._onSliderChange();
      }
    });
    btnRow.appendChild(nextBtn);
    panel.appendChild(btnRow);

    // Tool solid toggle (on by default)
    const toolRow = document.createElement('div');
    toolRow.style.cssText = 'display: flex; align-items: center; gap: 8px; margin-bottom: 16px;';
    const toolCb = document.createElement('input');
    toolCb.type = 'checkbox';
    toolCb.id = 'tool-toggle';
    toolCb.checked = true;
    toolCb.addEventListener('change', (e) => {
      this.showTool = e.target.checked;
      this._updateToolMesh();
    });
    const toolLbl = document.createElement('label');
    toolLbl.htmlFor = 'tool-toggle';
    toolLbl.style.cssText = 'font-size: 12px; color: var(--bp-gray1); cursor: pointer;';
    toolLbl.textContent = 'Show tool solid (cut/union shape)';
    toolRow.appendChild(toolCb);
    toolRow.appendChild(toolLbl);
    panel.appendChild(toolRow);

    // Other bodies in this bot
    if (this.allBodies.length > 1) {
      const bodiesTitle = document.createElement('h3');
      bodiesTitle.textContent = 'Bodies';
      panel.appendChild(bodiesTitle);
      const bodiesDiv = document.createElement('div');
      bodiesDiv.style.cssText = 'margin-bottom: 12px;';
      for (const name of this.allBodies) {
        const link = document.createElement('a');
        link.className = 'prop-chip body-chip';
        link.style.cssText = 'cursor: pointer; text-decoration: none;';
        if (name === this.bodyName) {
          link.style.background = 'rgba(19,124,189,0.15)';
          link.style.borderColor = 'var(--bp-blue3)';
        }
        link.textContent = name;
        link.href = `?cadsteps=${encodeURIComponent(this.botName)}:${encodeURIComponent(name)}`;
        bodiesDiv.appendChild(link);
      }
      panel.appendChild(bodiesDiv);
    }

    // Step list
    const listTitle = document.createElement('h3');
    listTitle.textContent = 'All Steps';
    panel.appendChild(listTitle);

    this.stepListEl = document.createElement('div');
    let currentGroup = null;
    for (const step of this.steps) {
      // Insert group header when entering a new CallOp group
      if (step.group && step.group !== currentGroup) {
        const header = document.createElement('div');
        header.style.cssText = `
          font-size: 11px; padding: 4px 6px 2px; margin-top: 6px;
          color: var(--bp-gray1); font-weight: 600; letter-spacing: 0.3px;
          border-bottom: 1px solid var(--bp-light-gray1, #E1E8ED);
        `;
        header.textContent = step.group;
        this.stepListEl.appendChild(header);
      }
      currentGroup = step.group || null;

      const row = document.createElement('div');
      const indent = step.group ? 'padding-left: 16px;' : '';
      row.style.cssText = `
        font-size: 12px; padding: 4px 6px; border-radius: 4px; cursor: pointer;
        margin-bottom: 2px; display: flex; align-items: center; gap: 6px;
        transition: background 0.1s; ${indent}
      `;
      row.dataset.index = step.index;
      if (step.group) row.classList.add('step-group-call');

      const dot = document.createElement('span');
      const color = OP_COLORS[step.op] || 0x5C7080;
      dot.style.cssText = `
        width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
        background: #${color.toString(16).padStart(6, '0')};
      `;
      row.appendChild(dot);

      const label = document.createElement('span');
      // Show ShapeScript code line if available, otherwise fall back to label
      label.textContent = step.script || step.label;
      label.style.cssText = `
        color: var(--bp-dark-gray5); white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis; font-family: 'Input Sans Narrow', 'SF Mono', monospace;
        font-size: 11px;
      `;
      row.appendChild(label);

      row.addEventListener('click', () => {
        this.slider.value = String(step.index);
        this._onSliderChange();
      });
      row.addEventListener('mouseenter', () => { row.style.background = 'rgba(206,217,224,0.5)'; });
      row.addEventListener('mouseleave', () => {
        if (parseInt(row.dataset.index) !== this.currentStep) {
          row.style.background = '';
        }
      });

      this.stepListEl.appendChild(row);
    }
    panel.appendChild(this.stepListEl);

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
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
      }
    });
  }

  async _onSliderChange() {
    const idx = parseInt(this.slider.value);
    await this._showStep(idx);
  }

  async _showStep(idx) {
    if (idx < 0 || idx >= this.steps.length) return;
    this.currentStep = idx;
    const step = this.steps[idx];

    // Update UI
    this.stepValueEl.textContent = `${idx + 1} / ${this.steps.length}`;
    const opLabel = step.op === 'cut' ? 'subtract' : step.op === 'union' ? 'add' : step.op;
    const scriptLine = step.script
      ? `<div class="step-script" style="font-family: 'Input Sans Narrow', 'SF Mono', monospace; font-size: 11px; color: var(--bp-dark-gray3); margin-top: 4px; padding: 4px 6px; background: var(--bp-dark-gray1, #1C2127); color: #ABB3BF; border-radius: 3px; overflow-x: auto; white-space: nowrap;">${step.script.replace(/</g, '&lt;')}</div>`
      : '';
    this.stepInfoEl.innerHTML = `
      <div class="step-title">${step.label}</div>
      <div class="step-desc">Step ${idx + 1} of ${this.steps.length} &mdash; <code>${opLabel}</code></div>
      ${scriptLine}
    `;

    // Highlight in step list
    for (const row of this.stepListEl.children) {
      const rowIdx = parseInt(row.dataset.index);
      row.style.background = rowIdx === idx ? 'rgba(19,124,189,0.15)' : '';
    }

    // For steps with a tool, show the state BEFORE this operation so the
    // tool overlay shows what's about to change. For step 0 (create) or steps
    // without a tool, show the result of this step.
    const hasPrev = idx > 0 && step.has_tool;
    const bodyIdx = hasPrev ? idx - 1 : idx;
    const geometry = await this._loadStepSTL(bodyIdx);
    if (!geometry) return;

    // Remove old meshes (don't dispose cached STL geometries — just materials + edges)
    while (this.meshGroup.children.length > 0) {
      const child = this.meshGroup.children[0];
      this.meshGroup.remove(child);
      // Dispose edge geometries (generated per-step, not cached) and materials
      if (child.isLineSegments && child.geometry) child.geometry.dispose();
      if (child.material && child.material !== EDGE_MATERIAL) child.material.dispose();
    }

    // Body solid — neutral color so the tool overlay stands out
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xCED9E0,
      roughness: 0.5,
      metalness: 0.1,
      clearcoat: 0.1,
    });
    this.meshGroup.add(new THREE.Mesh(geometry, material));

    // Edge outlines (matching bot viewer style)
    const edges = new THREE.EdgesGeometry(geometry, EDGE_THRESHOLD);
    const lines = new THREE.LineSegments(edges, EDGE_MATERIAL);
    lines.raycast = () => {};
    this.meshGroup.add(lines);

    // Tool solid overlay
    await this._updateToolMesh();

    // Prefetch adjacent steps (body + tool STLs)
    this._prefetch(idx - 1);
    this._prefetch(idx + 1);

    // Auto-frame on first load
    if (idx === this.steps.length - 1 && !this._hasFramed) {
      this._frameCamera(geometry);
      this._hasFramed = true;
    }
  }

  async _updateToolMesh() {
    clearGroup(this.toolGroup);

    const step = this.steps[this.currentStep];
    if (!this.showTool || !step || !step.has_tool) return;

    const geometry = await this._loadSTL(this.toolStlCache, this.currentStep, 'tool-stl');
    if (!geometry) return;

    // Color by op type: red for cut, green for union
    const isCut = step.op === 'cut';
    const toolColor = isCut ? 0xDB3737 : 0x0F9960;
    const material = new THREE.MeshPhysicalMaterial({
      color: toolColor,
      transparent: true,
      opacity: 0.35,
      roughness: 0.8,
      metalness: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.toolGroup.add(new THREE.Mesh(geometry, material));

    // Tool edges — matching tool color, lower opacity
    const toolEdgeMat = new THREE.LineBasicMaterial({
      color: toolColor, transparent: true, opacity: 0.4,
    });
    const edges = new THREE.EdgesGeometry(geometry, EDGE_THRESHOLD);
    const lines = new THREE.LineSegments(edges, toolEdgeMat);
    lines.raycast = () => {};
    this.toolGroup.add(lines);
  }

  /** Load an STL into a cache by step index and URL suffix. */
  async _loadSTL(cache, idx, suffix) {
    if (idx < 0 || idx >= this.steps.length) return null;
    if (cache[idx]) return cache[idx];

    const url = `/api/bots/${this.botName}/body/${this.bodyName}/cad-steps/${idx}/${suffix}`;
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

  _frameCamera(geometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 2.5;

    this.controls.target.copy(center);
    this.camera.position.set(
      center.x + dist * 0.6,
      center.y + dist * 0.6,
      center.z + dist * 0.8
    );
    this.controls.update();
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
