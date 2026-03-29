/**
 * MindSim Component Browser — Three.js-based component viewer.
 *
 * Delegates the shared viewing pipeline (manifest → scene tree → mesh loading →
 * tree → click-select → section → measure) to ManifestViewer. This module keeps
 * only component-specific features: specs panel, 3D markers, steps mode, axis
 * gizmo, view presets, SVG render, and quick switcher.
 */

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { info, error as logError, timedFetch } from './log.ts';
import type { ViewerManifest } from './manifest-types.ts';
import { initManifestViewer, type ManifestViewerContext, type StlUrlContext } from './manifest-viewer.ts';
import { BP, hexStr } from './presentation.ts';
import { clearGroup, orientToAxis } from './utils.ts';
import { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const SIDE_PANEL_WIDTH = 320;

// ---------------------------------------------------------------------------
// View presets — camera direction (from center) and up vector
// ---------------------------------------------------------------------------
const VIEW_PRESETS = {
  iso: { dir: new THREE.Vector3(1, -1, 0.8).normalize(), up: new THREE.Vector3(0, 0, 1), label: 'Iso', key: '1' },
  front: { dir: new THREE.Vector3(0, -1, 0), up: new THREE.Vector3(0, 0, 1), label: 'Front', key: '2' },
  back: { dir: new THREE.Vector3(0, 1, 0), up: new THREE.Vector3(0, 0, 1), label: 'Back', key: '3' },
  top: { dir: new THREE.Vector3(0, 0, 1), up: new THREE.Vector3(0, 1, 0), label: 'Top', key: '4' },
  bottom: { dir: new THREE.Vector3(0, 0, -1), up: new THREE.Vector3(0, -1, 0), label: 'Bottom', key: '5' },
  right: { dir: new THREE.Vector3(1, 0, 0), up: new THREE.Vector3(0, 0, 1), label: 'Right', key: '6' },
  left: { dir: new THREE.Vector3(-1, 0, 0), up: new THREE.Vector3(0, 0, 1), label: 'Left', key: '7' },
};

// ---------------------------------------------------------------------------
// ComponentBrowser class
// ---------------------------------------------------------------------------
class ComponentBrowser {
  components: any[];
  currentComponent: any;
  stlLoader: STLLoader;
  activePreset: string;
  stepsMode: boolean;
  stepsData: any;
  stepsCurrentIdx: number;
  stepsStlCache: Record<number, any>;
  stepsToolStlCache: Record<number, any>;
  stepsGroup: THREE.Group;
  stepsToolGroup: THREE.Group;
  showStepTool: boolean;
  viewport: Viewport3D;
  _viewerCtx: ManifestViewerContext | null;
  _markerGroup: THREE.Group | null;
  _markers: Record<string, any[]>;
  _stepsHasFramed: boolean;
  _gizmoSvg: HTMLElement | null;
  _gizmoAxes: any[];
  _keydownHandler: ((e: KeyboardEvent) => void) | null;

  constructor() {
    this.components = [];
    this.currentComponent = null;
    this.stlLoader = new STLLoader();

    this.activePreset = 'iso';
    this.stepsMode = false;
    this.stepsData = null;
    this.stepsCurrentIdx = 0;
    this.stepsStlCache = {};
    this.stepsToolStlCache = {};
    this.stepsGroup = new THREE.Group();
    this.stepsGroup.name = 'shapescript-steps';
    this.stepsToolGroup = new THREE.Group();
    this.stepsToolGroup.name = 'shapescript-tool';
    this.showStepTool = true;
    this._viewerCtx = null;
    this._markerGroup = null;
    this._markers = {};
    this._stepsHasFramed = false;
    this._gizmoSvg = null;
    this._gizmoAxes = [];
    this._keydownHandler = null;
  }

  async init() {
    this._setupViewport();
    this._setupViewToolbar();
    this._setupAxisGizmo();
    await this._fetchCatalog();
    this._buildSidePanel();
  }

  // -----------------------------------------------------------------------
  // Viewport3D setup
  // -----------------------------------------------------------------------

  _setupViewport() {
    const container = document.getElementById('canvas-container')!;
    container.style.right = `${SIDE_PANEL_WIDTH}px`;

    this.viewport = new Viewport3D(container, {
      cameraType: 'orthographic',
      grid: true,
    });

    // ShapeScript step groups (added directly to scene)
    this.viewport.scene.add(this.stepsGroup);
    this.viewport.scene.add(this.stepsToolGroup);

    // Start animation loop
    this.viewport.animate(() => {});
  }

  // -----------------------------------------------------------------------
  // View preset toolbar — keep: view dropdown, steps button, render SVG.
  // Section and measure wiring is handled by ManifestViewer.
  // -----------------------------------------------------------------------

  _setupViewToolbar() {
    // Keyboard shortcuts — view presets (lock rotation for axis-aligned), arrow keys
    const keyMap: Record<string, string> = {
      '1': 'iso',
      '2': 'front',
      '3': 'back',
      '4': 'top',
      '5': 'bottom',
      '6': 'right',
      '7': 'left',
    };
    // Store reference to prevent accumulating listeners on repeated calls
    if (this._keydownHandler) {
      document.removeEventListener('keydown', this._keydownHandler);
    }
    this._keydownHandler = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return;
      const ctrl = e.ctrlKey || e.metaKey;

      if (!ctrl && keyMap[e.key]) {
        e.preventDefault();
        this._setViewPreset(keyMap[e.key]);
      } else if (e.key.startsWith('Arrow')) {
        e.preventDefault();
        this._arrowKeyNav(e.key, e.shiftKey);
      }
    };
    document.addEventListener('keydown', this._keydownHandler);
  }

  _setViewPreset(key: string) {
    const preset = VIEW_PRESETS[key as keyof typeof VIEW_PRESETS];
    if (!preset) return;

    this.activePreset = key;
    this.viewport.setViewPreset(key);

    // Lock rotation for axis-aligned views, allow for iso
    this.viewport.controls.enableRotate = key === 'iso';

    // Fit ortho frustum to visible geometry
    const box = this._getVisibleBBox();
    if (box) this.viewport.fitOrthoFrustum(box);
    this.viewport.controls.update();
    this._updatePresetButtons();
  }

  /** Orbit or pan the camera via arrow keys. */
  _arrowKeyNav(key: string, shift: boolean) {
    const ORBIT_STEP = Math.PI / 12;
    const camera = this.viewport.camera;
    const controls = this.viewport.controls;
    const target = controls.target;
    const pos = camera.position;
    const offset = pos.clone().sub(target);

    if (shift) {
      const panStep = (camera as THREE.OrthographicCamera).top * 0.15;
      camera.updateMatrixWorld();
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      right.setFromMatrixColumn(camera.matrixWorld, 0);
      up.setFromMatrixColumn(camera.matrixWorld, 1);

      let dx = 0;
      let dy = 0;
      if (key === 'ArrowLeft') dx = -panStep;
      if (key === 'ArrowRight') dx = panStep;
      if (key === 'ArrowUp') dy = panStep;
      if (key === 'ArrowDown') dy = -panStep;

      const move = right.multiplyScalar(dx).add(up.multiplyScalar(dy));
      target.add(move);
      pos.add(move);
    } else {
      if (key === 'ArrowLeft' || key === 'ArrowRight') {
        const angle = key === 'ArrowLeft' ? ORBIT_STEP : -ORBIT_STEP;
        offset.applyAxisAngle(new THREE.Vector3(0, 0, 1), angle);
      } else {
        const angle = key === 'ArrowUp' ? -ORBIT_STEP : ORBIT_STEP;
        camera.updateMatrixWorld();
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
        offset.applyAxisAngle(right, angle);
        camera.up.applyAxisAngle(right, angle);
      }
      pos.copy(target).add(offset);
      camera.lookAt(target);
    }

    controls.enableRotate = true;
    this.activePreset = '';
    this._updatePresetButtons();
    controls.update();
  }

  _updatePresetButtons() {
    // View dropdown removed — presets available via 1-7 keyboard shortcuts
  }

  // -----------------------------------------------------------------------
  // Bounding box helper (for view presets + SVG render)
  // -----------------------------------------------------------------------

  _getVisibleBBox(): THREE.Box3 | null {
    const box = new THREE.Box3();
    let hasGeometry = false;

    const groups = this.viewport.groups;
    for (const group of Object.values(groups)) {
      if (!group.visible) continue;
      group.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (mesh.isMesh) {
          mesh.geometry.computeBoundingBox();
          const childBox = mesh.geometry.boundingBox!.clone();
          childBox.applyMatrix4(mesh.matrixWorld);
          box.union(childBox);
          hasGeometry = true;
        }
      });
    }

    return hasGeometry ? box : null;
  }

  // -----------------------------------------------------------------------
  // Component catalog + sidebar
  // -----------------------------------------------------------------------

  async _fetchCatalog() {
    try {
      const resp = await timedFetch('/api/components');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      this.components = await resp.json();
      info('components', `catalog loaded: ${this.components.length} components`);
    } catch (err: any) {
      logError('components', 'failed to fetch catalog', { error: err.message });
      document.getElementById('side-panel')!.innerHTML =
        `<p style="color:#ff6666; font-size:13px">Failed to load components: ${err.message}</p>`;
    }
  }

  // -----------------------------------------------------------------------
  // Side panel
  // -----------------------------------------------------------------------

  _buildSidePanel() {
    const panel = document.getElementById('side-panel')!;
    panel.innerHTML = '<p style="color:var(--bp-gray3); font-size:13px">Select a component</p>';
  }

  _updateSidePanel(comp: any) {
    const panel = document.getElementById('side-panel')!;
    let html = `<h2>${comp.name}</h2>`;

    html += '<h3>General</h3>';
    html += '<div style="font-size:12px; color:#999; line-height:1.8">';
    html += `<div>Category: <span style="color:#ccc">${comp.category}</span></div>`;
    html += `<div>Dimensions: <span style="color:#ccc">${comp.dimensions_mm.map((d: number) => d.toFixed(1)).join(' x ')} mm</span></div>`;
    html += `<div>Mass: <span style="color:#ccc">${comp.mass_g.toFixed(1)} g</span></div>`;
    html += '</div>';

    if (comp.servo) {
      html += '<h3>Servo</h3>';
      html += '<div style="font-size:12px; color:#999; line-height:1.8">';
      html += `<div>Torque: <span style="color:#ccc">${comp.servo.stall_torque_nm.toFixed(2)} N-m</span></div>`;
      html += `<div>Speed: <span style="color:#ccc">${comp.servo.no_load_speed_rpm.toFixed(0)} RPM</span></div>`;
      html += `<div>Voltage: <span style="color:#ccc">${comp.servo.voltage} V</span></div>`;
      html += `<div>Range: <span style="color:#ccc">${comp.servo.range_deg[0]}\u00B0 to ${comp.servo.range_deg[1]}\u00B0</span></div>`;
      html += `<div>Gear Ratio: <span style="color:#ccc">${comp.servo.gear_ratio}:1</span></div>`;
      if (comp.servo.continuous) html += '<div style="color:#8888cc">Continuous rotation</div>';
      html += '</div>';
    }

    if (comp.mounting_points.length > 0) {
      html += '<h3>Mounting Points</h3>';
      html += '<div style="font-size:12px; color:#999; line-height:1.6">';
      for (let i = 0; i < comp.mounting_points.length; i++) {
        const mp = comp.mounting_points[i];
        html += `<div class="highlight-item" data-marker-type="mp" data-marker-idx="${i}">${mp.label} <span style="color:#666">(${mp.diameter_mm.toFixed(1)}mm)</span></div>`;
      }
      html += '</div>';
    }

    if (comp.wire_ports.length > 0) {
      html += '<h3>Wire Ports</h3>';
      html += '<div style="font-size:12px; color:#999; line-height:1.6">';
      for (let i = 0; i < comp.wire_ports.length; i++) {
        const wp = comp.wire_ports[i];
        html += `<div class="highlight-item" data-marker-type="wp" data-marker-idx="${i}">${wp.label} <span style="color:#666">${wp.bus_type}</span></div>`;
      }
      html += '</div>';
    }

    panel.innerHTML = html;

    panel.querySelectorAll('.highlight-item').forEach((el) => {
      const hel = el as HTMLElement;
      hel.addEventListener('mouseenter', () => {
        const type = hel.dataset.markerType!;
        const idx = Number.parseInt(hel.dataset.markerIdx!, 10);
        this._highlightMarker(type, idx, true);
      });
      hel.addEventListener('mouseleave', () => {
        const type = hel.dataset.markerType!;
        const idx = Number.parseInt(hel.dataset.markerIdx!, 10);
        this._highlightMarker(type, idx, false);
      });
    });
  }

  // -----------------------------------------------------------------------
  // 3D markers (mounting points, wire ports)
  // -----------------------------------------------------------------------

  _createMarkers(comp: any) {
    if (this._markerGroup) {
      clearGroup(this._markerGroup);
      this.viewport.scene.remove(this._markerGroup);
    }
    this._markerGroup = new THREE.Group();
    this._markerGroup.name = 'markers';
    this.viewport.scene.add(this._markerGroup);

    this._markers = { mp: [], wp: [] };

    const markerMat = new THREE.MeshPhysicalMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.4,
      roughness: 0.3,
      metalness: 0.1,
    });
    const highlightMat = new THREE.MeshPhysicalMaterial({
      color: 0x44ff88,
      transparent: false,
      opacity: 1.0,
      roughness: 0.3,
      metalness: 0.1,
      emissive: 0x22aa44,
      emissiveIntensity: 0.5,
    });

    for (const mp of comp.mounting_points) {
      const r = (mp.diameter_mm / 1000) * 0.8;
      const h = 0.002;
      const geom = new THREE.CylinderGeometry(r, r, h, 12);
      const mesh = new THREE.Mesh(geom, markerMat.clone());
      mesh.position.set(mp.pos[0], mp.pos[1], mp.pos[2]);
      orientToAxis(mesh, new THREE.Vector3(mp.axis[0], mp.axis[1], mp.axis[2]));
      mesh.userData.defaultMat = mesh.material;
      mesh.userData.highlightMat = highlightMat;
      this._markerGroup.add(mesh);
      this._markers.mp.push(mesh);
    }

    const wpMat = new THREE.MeshPhysicalMaterial({
      color: 0xff8844,
      transparent: true,
      opacity: 0.4,
      roughness: 0.3,
      metalness: 0.1,
    });
    const wpHighlightMat = new THREE.MeshPhysicalMaterial({
      color: 0xffaa22,
      transparent: false,
      opacity: 1.0,
      roughness: 0.3,
      metalness: 0.1,
      emissive: 0xcc8800,
      emissiveIntensity: 0.5,
    });

    for (const wp of comp.wire_ports) {
      const geom = new THREE.SphereGeometry(0.002, 12, 12);
      const mesh = new THREE.Mesh(geom, wpMat.clone());
      mesh.position.set(wp.pos[0], wp.pos[1], wp.pos[2]);
      mesh.userData.defaultMat = mesh.material;
      mesh.userData.highlightMat = wpHighlightMat;
      this._markerGroup.add(mesh);
      this._markers.wp.push(mesh);
    }
  }

  _highlightMarker(type: string, idx: number, on: boolean) {
    const markers = this._markers?.[type];
    if (!markers || !markers[idx]) return;
    const mesh = markers[idx];
    mesh.material = on ? mesh.userData.highlightMat : mesh.userData.defaultMat;
    const s = on ? 1.8 : 1.0;
    mesh.scale.set(s, s, s);
  }

  // -----------------------------------------------------------------------
  // Component loading — delegates to ManifestViewer
  // -----------------------------------------------------------------------

  async loadComponent(name: string) {
    const comp = this.components.find((c: any) => c.name === name);
    if (!comp) return;
    this.currentComponent = comp;

    document.getElementById('bot-name')!.textContent = name;
    document.getElementById('mode-tabs')!.style.display = 'none';

    // Exit steps mode if active
    if (this.stepsMode) {
      this._exitStepsMode();
    }

    // Dispose previous ManifestViewer
    if (this._viewerCtx) {
      this._viewerCtx.dispose();
      this._viewerCtx = null;
    }

    // Fetch manifest
    const resp = await timedFetch(`/api/components/${name}/manifest`);
    if (!resp.ok) return;
    const manifest = (await resp.json()) as ViewerManifest;

    // Init ManifestViewer — reuses our viewport
    const compName = name;
    this._viewerCtx = await initManifestViewer({
      container: document.getElementById('canvas-container')!,
      treePanelEl: document.getElementById('tree-content')!,
      sidePanelEl: document.getElementById('side-panel')!,
      manifest,
      viewport: this.viewport,
      resolveStlUrl: (mesh: string, _context: StlUrlContext) => {
        if (mesh.startsWith('_')) return `/api/components/${mesh}/stl/body`;
        return `/api/components/${compName}/stl/${mesh}`;
      },
      onNodeSelected: (_nodeId) => {
        // Could update side panel context in future
      },
    });

    // Hide the ghost body node in tree
    const treeContent = document.getElementById('tree-content');
    if (treeContent) {
      const bodyRow = treeContent.querySelector(`[data-node-id="body:${name}"]`);
      if (bodyRow) (bodyRow as HTMLElement).style.display = 'none';
    }

    // Update specs panel
    this._updateSidePanel(comp);

    // Create 3D markers for mounting points and wire ports
    this._createMarkers(comp);
  }

  // -----------------------------------------------------------------------
  // ShapeScript steps mode
  // -----------------------------------------------------------------------

  async _toggleStepsMode() {
    if (!this.currentComponent) return;

    if (this.stepsMode) {
      this._exitStepsMode();
      return;
    }

    const btn = document.getElementById('steps-toggle') as HTMLButtonElement | null;
    if (btn) {
      btn.textContent = 'Loading...';
      btn.disabled = true;
    }

    try {
      const resp = await timedFetch(`/api/components/${this.currentComponent.name}/shapescript`);
      if (!resp.ok) {
        if (resp.status === 404) {
          this._showStepsUnavailable();
        } else {
          logError('cad-steps', `ShapeScript steps fetch failed: ${resp.status}`);
        }
        return;
      }

      this.stepsData = await resp.json();
      if (!this.stepsData.steps || this.stepsData.steps.length === 0) {
        this._showStepsUnavailable();
        return;
      }

      this._enterStepsMode();
    } catch (err) {
      logError('cad-steps', 'failed to fetch ShapeScript steps', {
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      if (btn) {
        btn.textContent = 'Steps';
        btn.disabled = false;
      }
    }
  }

  _showStepsUnavailable() {
    const panel = document.getElementById('side-panel')!;
    const msg = document.createElement('div');
    msg.style.cssText =
      'color: var(--bp-gray3); font-size: 13px; padding: 12px; background: rgba(206,217,224,0.1); border-radius: 6px; margin-top: 12px;';
    msg.textContent = 'ShapeScript not available for this component';
    msg.id = 'steps-unavailable';
    panel.appendChild(msg);
    setTimeout(() => msg.remove(), 3000);
  }

  _enterStepsMode() {
    this.stepsMode = true;
    const btn = document.getElementById('steps-toggle');
    if (btn) btn.classList.add('active');

    // Hide ManifestViewer mesh groups
    const groups = this.viewport.groups;
    for (const group of Object.values(groups)) {
      group.visible = false;
    }
    if (this._markerGroup) this._markerGroup.visible = false;
    this.stepsGroup.visible = true;
    this.stepsToolGroup.visible = true;

    this.stepsStlCache = {};
    this.stepsToolStlCache = {};

    this._buildStepsPanel();

    this._showComponentStep(this.stepsData.steps.length - 1);
  }

  _exitStepsMode() {
    this.stepsMode = false;
    const btn = document.getElementById('steps-toggle');
    if (btn) btn.classList.remove('active');

    clearGroup(this.stepsGroup);
    clearGroup(this.stepsToolGroup);
    this.stepsGroup.visible = false;
    this.stepsToolGroup.visible = false;

    if (this._markerGroup) this._markerGroup.visible = true;

    // Restore ManifestViewer mesh group visibility
    const groups = this.viewport.groups;
    for (const group of Object.values(groups)) {
      group.visible = true;
    }
    // Re-apply design scene visibility (respects hidden nodes)
    if (this._viewerCtx) {
      this._viewerCtx.syncVisibility();
    }

    const comp = this.currentComponent;
    if (comp) this._updateSidePanel(comp);

    this._fitCameraToVisibleMeshes();
  }

  _fitCameraToVisibleMeshes() {
    const box = this._getVisibleBBox();
    if (!box) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    const preset = VIEW_PRESETS[this.activePreset as keyof typeof VIEW_PRESETS] || VIEW_PRESETS.iso;
    const dist = maxDim * 2;
    this.viewport.camera.position.copy(center).addScaledVector(preset.dir, dist);
    this.viewport.camera.up.copy(preset.up);
    this.viewport.controls.target.copy(center);
    this.viewport.camera.lookAt(center);

    this.viewport.fitOrthoFrustum(box);
    this.viewport.controls.update();
  }

  _buildStepsPanel() {
    const panel = document.getElementById('side-panel')!;
    const comp = this.currentComponent;
    const steps = this.stepsData.steps;

    let html = `<h2>${comp.name}</h2>`;
    html += '<div class="prop-badge" style="margin-bottom:12px">ShapeScript Steps</div>';

    html += '<div id="step-info" class="step-info"><div class="step-title">Loading...</div></div>';

    html += '<div class="slider-group">';
    html +=
      '<div class="slider-label"><span class="name">Step</span><span class="value" id="step-value">0</span></div>';
    html += `<input type="range" id="steps-slider" min="0" max="${steps.length - 1}" value="${steps.length - 1}">`;
    html += '</div>';

    html += '<div style="display:flex;gap:8px;margin-bottom:12px">';
    html += '<button class="btn" id="step-prev">Prev</button>';
    html += '<button class="btn" id="step-next">Next</button>';
    html += '</div>';

    html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">';
    html += `<input type="checkbox" id="step-tool-toggle" ${this.showStepTool ? 'checked' : ''}>`;
    html +=
      '<label for="step-tool-toggle" style="font-size:12px;color:var(--bp-gray1);cursor:pointer">Show tool solid</label>';
    html += '</div>';

    html += '<h3>All Steps</h3>';
    html += '<div id="steps-list">';
    const OP_COLORS: Record<string, string> = { create: '#2B95D6', cut: '#DB3737', union: '#0F9960' };
    for (const step of steps) {
      html += `<div class="step-row" data-step-idx="${step.index}" style="font-size:12px;padding:4px 6px;border-radius:4px;cursor:pointer;margin-bottom:2px;display:flex;align-items:center;gap:6px;transition:background 0.1s">`;
      html += `<span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:${OP_COLORS[step.op] || '#5C7080'}"></span>`;
      html += `<span style="color:var(--bp-dark-gray5);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:'Input Sans Narrow','SF Mono',monospace;font-size:11px">${step.script || step.label}</span>`;
      html += '</div>';
    }
    html += '</div>';

    html += '<div style="margin-top:16px"><button class="btn" id="steps-exit">Exit Steps</button></div>';

    panel.innerHTML = html;

    const sliderEl = document.getElementById('steps-slider') as HTMLInputElement;
    sliderEl.addEventListener('input', () => this._showComponentStep(Number.parseInt(sliderEl.value, 10)));

    document.getElementById('step-prev')!.addEventListener('click', () => {
      if (this.stepsCurrentIdx > 0) {
        sliderEl.value = String(this.stepsCurrentIdx - 1);
        this._showComponentStep(this.stepsCurrentIdx - 1);
      }
    });
    document.getElementById('step-next')!.addEventListener('click', () => {
      if (this.stepsCurrentIdx < steps.length - 1) {
        sliderEl.value = String(this.stepsCurrentIdx + 1);
        this._showComponentStep(this.stepsCurrentIdx + 1);
      }
    });
    document.getElementById('step-tool-toggle')!.addEventListener('change', (e) => {
      this.showStepTool = (e.target as HTMLInputElement).checked;
      this._updateStepToolMesh();
    });
    document.getElementById('steps-exit')!.addEventListener('click', () => this._exitStepsMode());

    panel.querySelectorAll('.step-row').forEach((row) => {
      const rowEl = row as HTMLElement;
      rowEl.addEventListener('click', () => {
        const idx = Number.parseInt(rowEl.dataset.stepIdx!, 10);
        sliderEl.value = String(idx);
        this._showComponentStep(idx);
      });
      rowEl.addEventListener('mouseenter', () => {
        rowEl.style.background = 'rgba(206,217,224,0.5)';
      });
      rowEl.addEventListener('mouseleave', () => {
        if (Number.parseInt(rowEl.dataset.stepIdx!, 10) !== this.stepsCurrentIdx) {
          rowEl.style.background = '';
        }
      });
    });
  }

  async _showComponentStep(idx: number) {
    const steps = this.stepsData.steps;
    if (idx < 0 || idx >= steps.length) return;
    this.stepsCurrentIdx = idx;
    const step = steps[idx];

    const infoEl = document.getElementById('step-info');
    if (infoEl) {
      const opLabel = step.op === 'cut' ? 'subtract' : step.op === 'union' ? 'add' : step.op;
      const scriptLine = step.script
        ? `<div style="font-family:'Input Sans Narrow','SF Mono',monospace;font-size:11px;margin-top:4px;padding:4px 6px;background:var(--bp-dark-gray1,#1C2127);color:#ABB3BF;border-radius:3px;overflow-x:auto;white-space:nowrap">${step.script.replace(/</g, '&lt;')}</div>`
        : '';
      infoEl.innerHTML = `<div class="step-title">${step.label}</div><div class="step-desc">Step ${idx + 1} of ${steps.length} &mdash; <code>${opLabel}</code></div>${scriptLine}`;
    }
    const valueEl = document.getElementById('step-value');
    if (valueEl) valueEl.textContent = `${idx + 1} / ${steps.length}`;

    document.querySelectorAll('.step-row').forEach((row) => {
      (row as HTMLElement).style.background =
        Number.parseInt((row as HTMLElement).dataset.stepIdx!, 10) === idx ? 'rgba(19,124,189,0.15)' : '';
    });

    const hasPrev = idx > 0 && step.has_tool;
    const bodyIdx = hasPrev ? idx - 1 : idx;
    const geometry = await this._loadComponentStepSTL(bodyIdx);
    if (!geometry) return;

    clearGroup(this.stepsGroup);
    const material = new THREE.MeshPhysicalMaterial({
      color: 0xced9e0,
      roughness: 0.5,
      metalness: 0.1,
      clearcoat: 0.1,
    });
    this.stepsGroup.add(new THREE.Mesh(geometry, material));

    const edges = new THREE.EdgesGeometry(geometry, 28);
    const lines = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({
        color: 0x000000,
        transparent: true,
        opacity: 0.6,
      }),
    );
    lines.raycast = () => {};
    this.stepsGroup.add(lines);

    await this._updateStepToolMesh();

    this._prefetchComponentStep(idx - 1);
    this._prefetchComponentStep(idx + 1);

    if (!this._stepsHasFramed) {
      this._stepsFrameCamera(geometry);
      this._stepsHasFramed = true;
    }
  }

  async _updateStepToolMesh() {
    clearGroup(this.stepsToolGroup);
    const step = this.stepsData?.steps[this.stepsCurrentIdx];
    if (!this.showStepTool || !step || !step.has_tool) return;

    const name = this.currentComponent.name;
    const geometry = await this._loadStepSTLGeneric(
      this.stepsToolStlCache,
      this.stepsCurrentIdx,
      `/api/components/${name}/shapescript/${this.stepsCurrentIdx}/tool-stl`,
    );
    if (!geometry) return;

    const isCut = step.op === 'cut';
    const toolColor = isCut ? 0xdb3737 : 0x0f9960;
    const material = new THREE.MeshPhysicalMaterial({
      color: toolColor,
      transparent: true,
      opacity: 0.35,
      roughness: 0.8,
      metalness: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.stepsToolGroup.add(new THREE.Mesh(geometry, material));

    const toolEdges = new THREE.EdgesGeometry(geometry, 28);
    const toolLines = new THREE.LineSegments(
      toolEdges,
      new THREE.LineBasicMaterial({
        color: toolColor,
        transparent: true,
        opacity: 0.4,
      }),
    );
    toolLines.raycast = () => {};
    this.stepsToolGroup.add(toolLines);
  }

  async _loadComponentStepSTL(idx: number) {
    const name = this.currentComponent.name;
    return this._loadStepSTLGeneric(this.stepsStlCache, idx, `/api/components/${name}/shapescript/${idx}/stl`);
  }

  async _loadStepSTLGeneric(cache: Record<number, any>, idx: number, url: string) {
    if (idx < 0) return null;
    if (cache[idx]) return cache[idx];

    return new Promise((resolve) => {
      this.stlLoader.load(
        url,
        (geometry) => {
          geometry.computeVertexNormals();
          cache[idx] = geometry;
          resolve(geometry);
        },
        undefined,
        () => resolve(null),
      );
    });
  }

  _prefetchComponentStep(idx: number) {
    const steps = this.stepsData?.steps;
    if (!steps || idx < 0 || idx >= steps.length) return;
    if (!this.stepsStlCache[idx]) this._loadComponentStepSTL(idx);
    if (steps[idx]?.has_tool && !this.stepsToolStlCache[idx]) {
      const name = this.currentComponent.name;
      this._loadStepSTLGeneric(this.stepsToolStlCache, idx, `/api/components/${name}/shapescript/${idx}/tool-stl`);
    }
  }

  _stepsFrameCamera(geometry: THREE.BufferGeometry) {
    geometry.computeBoundingBox();
    const box = geometry.boundingBox!;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    const preset = VIEW_PRESETS.iso;
    const dist = maxDim * 2;
    this.viewport.camera.position.copy(center).addScaledVector(preset.dir, dist);
    this.viewport.camera.up.copy(preset.up);
    this.viewport.controls.target.copy(center);
    this.viewport.camera.lookAt(center);
    this.viewport.controls.enableRotate = true;

    this.viewport.fitOrthoFrustum(box);
    this.viewport.controls.update();
  }

  // -----------------------------------------------------------------------
  // Axis gizmo — shows X/Y/Z orientation in the corner
  // -----------------------------------------------------------------------

  _setupAxisGizmo() {
    this._gizmoSvg = document.getElementById('axis-gizmo');
    this._gizmoAxes = [
      { dir: new THREE.Vector3(1, 0, 0), color: hexStr(BP.RED3), label: '+X', neg: '-X' },
      { dir: new THREE.Vector3(0, 1, 0), color: hexStr(BP.GREEN3), label: '+Y', neg: '-Y' },
      { dir: new THREE.Vector3(0, 0, 1), color: hexStr(BP.BLUE4), label: '+Z', neg: '-Z' },
    ];

    // Drive gizmo updates from the viewport animation loop
    const origAnimCb = this.viewport.animationCallback;
    this.viewport.animate(() => {
      if (origAnimCb) origAnimCb();
      this._updateAxisGizmo();
    });
  }

  _updateAxisGizmo() {
    const svg = this._gizmoSvg;
    if (!svg) return;

    const cx = 50;
    const cy = 50;
    const len = 32;
    svg.innerHTML = '';

    const camera = this.viewport.camera;
    camera.updateMatrixWorld();
    const invMat = camera.matrixWorld.clone().invert();

    const projected = this._gizmoAxes
      .map((axis: any) => {
        const d = axis.dir.clone().transformDirection(invMat);
        return { ...axis, d, depth: d.z };
      })
      .sort((a: any, b: any) => b.depth - a.depth);

    for (const { d, color, label } of projected) {
      const behind = d.z > 0.2;
      const opacity = behind ? 0.3 : 1.0;
      const lineW = behind ? 1.5 : 2.5;
      const sx = d.x * len;
      const sy = -d.y * len;

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', String(cx));
      line.setAttribute('y1', String(cy));
      line.setAttribute('x2', String(cx + sx));
      line.setAttribute('y2', String(cy + sy));
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', String(lineW));
      line.setAttribute('stroke-linecap', 'round');
      line.setAttribute('opacity', String(opacity));
      svg.appendChild(line);

      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', String(cx + sx));
      dot.setAttribute('cy', String(cy + sy));
      dot.setAttribute('r', String(behind ? 3 : 5));
      dot.setAttribute('fill', color);
      dot.setAttribute('opacity', String(opacity));
      svg.appendChild(dot);

      const labelDist = len + 12;
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', String(cx + d.x * labelDist));
      text.setAttribute('y', String(cy - d.y * labelDist));
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'central');
      text.setAttribute('fill', color);
      text.setAttribute('opacity', String(opacity));
      text.setAttribute('style', 'font: 600 11px system-ui, -apple-system, sans-serif');
      text.textContent = label;
      svg.appendChild(text);
    }

    const centerDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    centerDot.setAttribute('cx', String(cx));
    centerDot.setAttribute('cy', String(cy));
    centerDot.setAttribute('r', String(2));
    centerDot.setAttribute('fill', '#8A9BA8');
    svg.appendChild(centerDot);
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
export async function initComponentBrowser(componentParam: string) {
  const browser = new ComponentBrowser();
  await browser.init();

  if (componentParam && componentParam !== 'catalog') {
    await browser.loadComponent(componentParam);

    const params = new URLSearchParams(window.location.search);
    if (params.get('steps') === 'true') {
      window.location.href = `?cadsteps=component:${encodeURIComponent(componentParam)}`;
    }
  }
}
