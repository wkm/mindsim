/**
 * MindSim Component Browser — Three.js-based component viewer.
 *
 * Uses Viewport3D for all 3D rendering (orthographic camera, section plane
 * with stencil caps, measure tool, view presets). This module manages the
 * component catalog, ComponentTree-based visibility, STL loading, and side panel UI.
 */

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { buildSceneTree } from './build-scene-tree.ts';
import { ComponentTree } from './component-tree.ts';
import { DesignScene, NodeKind } from './design-scene.ts';
import type { ManifestMount, ManifestPart, ViewerManifest } from './manifest-types.ts';
import { addMeshWithEdges, BP, createMaterial, hexStr, tintColor } from './presentation.ts';
import { clearGroup, orientToAxis } from './utils.ts';
import { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const SIDEBAR_WIDTH = 0;
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

const AXIS_INDEX: Record<string, number> = { x: 0, y: 1, z: 2 };

// ---------------------------------------------------------------------------
// ComponentBrowser class
// ---------------------------------------------------------------------------
class ComponentBrowser {
  components: any[];
  currentComponent: any;
  stlLoader: STLLoader;
  stlCache: Record<string, any>;
  _manifest: ViewerManifest | null;
  _designScene: DesignScene | null;
  _componentTree: ComponentTree | null;
  _meshGroups: Record<string, THREE.Group>;
  activePreset: string;
  sectionEnabled: boolean;
  sectionAxis: string;
  sectionFlipped: boolean;
  sectionFraction: number;
  stepsMode: boolean;
  stepsData: any;
  stepsCurrentIdx: number;
  stepsStlCache: Record<number, any>;
  stepsToolStlCache: Record<number, any>;
  stepsGroup: THREE.Group;
  stepsToolGroup: THREE.Group;
  showStepTool: boolean;
  viewport: any;
  scene: any;
  camera: any;
  renderer: any;
  controls: any;
  _markerGroup: THREE.Group | null;
  _markers: Record<string, any[]>;
  _stepsHasFramed: boolean;
  _lastSvgText: string;
  _lastSvgName: string;
  _gizmoSvg: HTMLElement | null;
  _gizmoAxes: any[];

  constructor() {
    this.components = [];
    this.currentComponent = null;
    this.stlLoader = new STLLoader();
    this.stlCache = {};
    this._manifest = null;
    this._designScene = null;
    this._componentTree = null;
    this._meshGroups = {};
    this.activePreset = 'iso';
    this.sectionEnabled = false;
    this.sectionAxis = 'z';
    this.sectionFlipped = false;
    this.sectionFraction = 0.5;
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
    this._markerGroup = null;
    this._markers = {};
    this._stepsHasFramed = false;
    this._lastSvgText = '';
    this._lastSvgName = '';
    this._gizmoSvg = null;
    this._gizmoAxes = [];
  }

  async init() {
    this._setupViewport();
    this._setupViewToolbar();
    this._setupAxisGizmo();
    await this._fetchCatalog();
    this._buildSidePanel();
  }

  // -----------------------------------------------------------------------
  // Viewport3D setup — delegates scene/camera/renderer/controls/lighting
  // -----------------------------------------------------------------------

  _setupViewport() {
    const container = document.getElementById('canvas-container');
    container.style.left = `${SIDEBAR_WIDTH}px`;
    container.style.right = `${SIDE_PANEL_WIDTH}px`;

    this.viewport = new Viewport3D(container, {
      cameraType: 'orthographic',
      grid: true,
    });

    // Expose delegates for convenience
    this.scene = this.viewport.scene;
    this.camera = this.viewport.camera;
    this.renderer = this.viewport.renderer;
    this.controls = this.viewport.controls;

    // ShapeScript step groups (added directly to scene)
    this.scene.add(this.stepsGroup);
    this.scene.add(this.stepsToolGroup);

    // Start animation loop
    this.viewport.animate();

    window.addEventListener('resize', () => this._updateLayout());
  }

  // -----------------------------------------------------------------------
  // View preset toolbar
  // -----------------------------------------------------------------------

  _setupViewToolbar() {
    // Populate view dropdown from VIEW_PRESETS
    const dropdown = document.getElementById('view-dropdown');
    const dropdownBtn = document.getElementById('view-dropdown-btn');
    if (dropdown) {
      dropdown.innerHTML = '';
      for (const [key, preset] of Object.entries(VIEW_PRESETS)) {
        const li = document.createElement('li');
        li.innerHTML = `<button class="dropdown-item" data-view="${key}">
          <span class="text">${preset.label}</span>
          <span class="dropdown-kbd">${preset.key}</span>
        </button>`;
        li.querySelector('button').addEventListener('click', () => {
          this._setViewPreset(key);
          dropdown.style.display = 'none';
        });
        dropdown.appendChild(li);
      }
      if (dropdownBtn) {
        dropdownBtn.addEventListener('click', () => {
          dropdown.style.display = dropdown.style.display === 'none' ? '' : 'none';
        });
        document.addEventListener('click', (e) => {
          if (!dropdownBtn.contains(e.target as Node) && !dropdown.contains(e.target as Node)) {
            dropdown.style.display = 'none';
          }
        });
      }
    }

    // Measure tool toggle
    const measureBtn = document.getElementById('measure-toggle');
    if (measureBtn) {
      measureBtn.addEventListener('click', () => {
        const active = !this.viewport._meas.enabled;
        if (active) {
          this.viewport.enableMeasureTool();
        } else {
          this.viewport.disableMeasureTool();
        }
        measureBtn.classList.toggle('active', active);
        const clearEl = document.getElementById('measure-clear');
        if (clearEl) clearEl.style.display = active ? '' : 'none';
      });
    }

    const clearBtn = document.getElementById('measure-clear');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        this.viewport._meas.clearAll();
      });
    }

    const renderBtn = document.getElementById('render-svg');
    if (renderBtn) {
      renderBtn.addEventListener('click', () => this._requestSVGRender());
    }

    // Steps button
    const stepsBtn = document.getElementById('steps-toggle');
    if (stepsBtn) {
      stepsBtn.addEventListener('click', () => {
        if (this.currentComponent) {
          window.location.href = `?cadsteps=component:${encodeURIComponent(this.currentComponent.name)}`;
        }
      });
    }

    // Section plane controls — drive Viewport3D's section
    const sectionToggle = document.getElementById('section-toggle');
    if (sectionToggle) {
      sectionToggle.addEventListener('click', () => {
        this.sectionEnabled = !this.sectionEnabled;
        sectionToggle.classList.toggle('active', this.sectionEnabled);
        const controls = document.getElementById('section-controls');
        if (controls) controls.style.display = this.sectionEnabled ? 'flex' : 'none';
        this._updateSectionPlane();
      });
    }

    for (const axis of ['x', 'y', 'z']) {
      const btn = document.querySelector(`[data-section-axis="${axis}"]`);
      if (btn) {
        btn.addEventListener('click', () => {
          this.sectionAxis = axis;
          document.querySelectorAll('[data-section-axis]').forEach((b) => {
            b.classList.toggle('active', (b as HTMLElement).dataset.sectionAxis === axis);
          });
          this._updateSectionPlane();
        });
      }
    }

    const slider = document.getElementById('section-slider');
    if (slider) {
      slider.addEventListener('input', () => {
        this.sectionFraction = parseFloat((slider as HTMLInputElement).value) / 100;
        this._updateSectionPlane();
      });
    }

    const flipBtn = document.getElementById('section-flip');
    if (flipBtn) {
      flipBtn.addEventListener('click', () => {
        this.sectionFlipped = !this.sectionFlipped;
        flipBtn.classList.toggle('active', this.sectionFlipped);
        this._updateSectionPlane();
      });
    }

    // Keyboard shortcuts
    const keyMap = {
      '1': 'iso',
      '2': 'front',
      '3': 'back',
      '4': 'top',
      '5': 'bottom',
      '6': 'right',
      '7': 'left',
    };
    document.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return;
      const ctrl = e.ctrlKey || e.metaKey;

      if (!ctrl && keyMap[e.key]) {
        e.preventDefault();
        this._setViewPreset(keyMap[e.key]);
      } else if (e.key === 'm' && !ctrl) {
        e.preventDefault();
        measureBtn?.click();
      } else if (e.key === 's' && !ctrl) {
        e.preventDefault();
        sectionToggle?.click();
      } else if (e.key === 'Escape') {
        const modal = document.getElementById('svg-modal');
        if (modal && modal.style.display !== 'none') {
          modal.style.display = 'none';
        }
      } else if (e.key.startsWith('Arrow')) {
        e.preventDefault();
        this._arrowKeyNav(e.key, e.shiftKey);
      }
    });
  }

  _setViewPreset(key: string) {
    const preset = VIEW_PRESETS[key];
    if (!preset) return;

    this.activePreset = key;
    this.viewport.setViewPreset(key);

    // Lock rotation for axis-aligned views, allow for iso
    this.controls.enableRotate = key === 'iso';

    // Fit ortho frustum to visible geometry
    const box = this._getVisibleBBox();
    if (box) this.viewport._fitOrthoFrustum(box);
    this.controls.update();
    this._updatePresetButtons();
  }

  /** Orbit or pan the camera via arrow keys. */
  _arrowKeyNav(key: string, shift: boolean) {
    const ORBIT_STEP = Math.PI / 12;
    const target = this.controls.target;
    const pos = this.camera.position;
    const offset = pos.clone().sub(target);

    if (shift) {
      const panStep = this.camera.top * 0.15;
      this.camera.updateMatrixWorld();
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      right.setFromMatrixColumn(this.camera.matrixWorld, 0);
      up.setFromMatrixColumn(this.camera.matrixWorld, 1);

      let dx = 0,
        dy = 0;
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
        this.camera.updateMatrixWorld();
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
        offset.applyAxisAngle(right, angle);
        this.camera.up.applyAxisAngle(right, angle);
      }
      pos.copy(target).add(offset);
      this.camera.lookAt(target);
    }

    this.controls.enableRotate = true;
    this.activePreset = null;
    this._updatePresetButtons();
    this.controls.update();
  }

  _updatePresetButtons() {
    const dropdown = document.getElementById('view-dropdown');
    if (dropdown) {
      dropdown.querySelectorAll('[data-view]').forEach((btn) => {
        btn.classList.toggle('active', (btn as HTMLElement).dataset.view === this.activePreset);
      });
    }
  }

  // -----------------------------------------------------------------------
  // Orthographic frustum helpers
  // -----------------------------------------------------------------------

  _getVisibleBBox() {
    const box = new THREE.Box3();
    let hasGeometry = false;

    for (const group of Object.values(this._meshGroups)) {
      if (!group.visible) continue;
      group.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (mesh.isMesh) {
          mesh.geometry.computeBoundingBox();
          const childBox = mesh.geometry.boundingBox.clone();
          childBox.applyMatrix4(mesh.matrixWorld);
          box.union(childBox);
          hasGeometry = true;
        }
      });
    }

    return hasGeometry ? box : null;
  }

  _fitCameraToVisibleMeshes() {
    const box = this._getVisibleBBox();
    if (!box) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    const preset = VIEW_PRESETS[this.activePreset] || VIEW_PRESETS.iso;
    const dist = maxDim * 2;
    this.camera.position.copy(center).addScaledVector(preset.dir, dist);
    this.camera.up.copy(preset.up);
    this.controls.target.copy(center);
    this.camera.lookAt(center);

    this.viewport._fitOrthoFrustum(box);
    this.controls.update();
  }

  _updateLayout() {
    const container = document.getElementById('canvas-container');
    container.style.left = `${SIDEBAR_WIDTH}px`;
    container.style.right = `${SIDE_PANEL_WIDTH}px`;

    requestAnimationFrame(() => {
      this.viewport.resize();
    });
  }

  // -----------------------------------------------------------------------
  // Component catalog + sidebar
  // -----------------------------------------------------------------------

  async _fetchCatalog() {
    try {
      const resp = await fetch('/api/components');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      this.components = await resp.json();
      console.log(`Component catalog: ${this.components.length} components loaded`);
    } catch (err) {
      console.error('Failed to fetch component catalog:', err);
      document.getElementById('side-panel').innerHTML =
        `<p style="color:#ff6666; font-size:13px">Failed to load components: ${err.message}</p>`;
    }
  }

  // -----------------------------------------------------------------------
  // Side panel
  // -----------------------------------------------------------------------

  _buildSidePanel() {
    const panel = document.getElementById('side-panel');
    panel.innerHTML = '<p style="color:var(--bp-gray3); font-size:13px">Select a component</p>';
  }

  _updateSidePanel(comp: any) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${comp.name}</h2>`;

    html += `<h3>General</h3>`;
    html += `<div style="font-size:12px; color:#999; line-height:1.8">`;
    html += `<div>Category: <span style="color:#ccc">${comp.category}</span></div>`;
    html += `<div>Dimensions: <span style="color:#ccc">${comp.dimensions_mm.map((d) => d.toFixed(1)).join(' x ')} mm</span></div>`;
    html += `<div>Mass: <span style="color:#ccc">${comp.mass_g.toFixed(1)} g</span></div>`;
    html += `</div>`;

    if (comp.servo) {
      html += `<h3>Servo</h3>`;
      html += `<div style="font-size:12px; color:#999; line-height:1.8">`;
      html += `<div>Torque: <span style="color:#ccc">${comp.servo.stall_torque_nm.toFixed(2)} N-m</span></div>`;
      html += `<div>Speed: <span style="color:#ccc">${comp.servo.no_load_speed_rpm.toFixed(0)} RPM</span></div>`;
      html += `<div>Voltage: <span style="color:#ccc">${comp.servo.voltage} V</span></div>`;
      html += `<div>Range: <span style="color:#ccc">${comp.servo.range_deg[0]}° to ${comp.servo.range_deg[1]}°</span></div>`;
      html += `<div>Gear Ratio: <span style="color:#ccc">${comp.servo.gear_ratio}:1</span></div>`;
      if (comp.servo.continuous) html += `<div style="color:#8888cc">Continuous rotation</div>`;
      html += `</div>`;
    }

    if (comp.mounting_points.length > 0) {
      html += `<h3>Mounting Points</h3>`;
      html += `<div style="font-size:12px; color:#999; line-height:1.6">`;
      for (let i = 0; i < comp.mounting_points.length; i++) {
        const mp = comp.mounting_points[i];
        html += `<div class="highlight-item" data-marker-type="mp" data-marker-idx="${i}">${mp.label} <span style="color:#666">(${mp.diameter_mm.toFixed(1)}mm)</span></div>`;
      }
      html += `</div>`;
    }

    if (comp.wire_ports.length > 0) {
      html += `<h3>Wire Ports</h3>`;
      html += `<div style="font-size:12px; color:#999; line-height:1.6">`;
      for (let i = 0; i < comp.wire_ports.length; i++) {
        const wp = comp.wire_ports[i];
        html += `<div class="highlight-item" data-marker-type="wp" data-marker-idx="${i}">${wp.label} <span style="color:#666">${wp.bus_type}</span></div>`;
      }
      html += `</div>`;
    }

    panel.innerHTML = html;

    this._createMarkers(comp);

    panel.querySelectorAll('.highlight-item').forEach((el) => {
      const hel = el as HTMLElement;
      hel.addEventListener('mouseenter', () => {
        const type = hel.dataset.markerType!;
        const idx = parseInt(hel.dataset.markerIdx!, 10);
        this._highlightMarker(type, idx, true);
      });
      hel.addEventListener('mouseleave', () => {
        const type = hel.dataset.markerType!;
        const idx = parseInt(hel.dataset.markerIdx!, 10);
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
      this.scene.remove(this._markerGroup);
    }
    this._markerGroup = new THREE.Group();
    this._markerGroup.name = 'markers';
    this.scene.add(this._markerGroup);

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
  // Component loading — manifest-driven with ComponentTree
  // -----------------------------------------------------------------------

  async loadComponent(name: string) {
    const comp = this.components.find((c) => c.name === name);
    if (!comp) return;
    this.currentComponent = comp;

    document.getElementById('bot-name').textContent = name;
    document.getElementById('mode-tabs').style.display = 'none';

    // Clear measurements, section, steps mode
    this.viewport._meas.clearAll();
    this._resetSection();
    if (this.stepsMode) {
      this.stepsMode = false;
      clearGroup(this.stepsGroup);
      clearGroup(this.stepsToolGroup);
      this.stepsGroup.visible = false;
      this.stepsToolGroup.visible = false;
      this.stepsData = null;
      this._stepsHasFramed = false;
      const btn = document.getElementById('steps-toggle');
      if (btn) btn.classList.remove('active');
    }

    // Clear previous mesh groups (Viewport3D has no removeGroup — clear + remove from scene)
    for (const [_id, group] of Object.entries(this._meshGroups)) {
      this.viewport.clearGroup(group);
      this.scene.remove(group);
    }
    this._meshGroups = {};

    // Dispose previous tree
    if (this._componentTree) {
      this._componentTree.dispose();
      this._componentTree = null;
    }

    // Fetch manifest
    const resp = await fetch(`/api/components/${name}/manifest`);
    if (!resp.ok) return;
    this._manifest = (await resp.json()) as ViewerManifest;

    // Build scene tree + design scene
    const sceneTree = buildSceneTree(this._manifest);
    this._designScene = new DesignScene(sceneTree);

    // Create viewport groups for body + mount + parts
    const bodyGroup = this.viewport.addGroup(`body:${name}`);
    this._meshGroups[`body:${name}`] = bodyGroup;

    // Load body mesh (always visible)
    const mount = this._manifest.mounts?.[0];
    if (mount?.meshes && mount.meshes.length > 0) {
      // Multi-material
      await this._loadMultiMaterialMount(name, mount, bodyGroup);
    } else {
      // Single material body
      const entityColor = comp.color
        ? new THREE.Color(comp.color[0], comp.color[1], comp.color[2])
        : new THREE.Color(0.541, 0.608, 0.659);
      await this._addSTLMesh(name, 'body', tintColor(entityColor), {}, bodyGroup);
    }

    // Register body meshes with design scene.
    // buildSceneTree creates: body node as `body:{name}`, mount as `mount:{name}:{name}`.
    // Register under the mount node since that's the component container.
    const mountNodeId = `mount:${name}:${name}`;
    const bodyNodeId = `body:${name}`;
    // Use mount node if it exists (multi-material), body node otherwise
    const regNodeId = this._designScene.tree.getNode(mountNodeId) ? mountNodeId : bodyNodeId;
    for (const child of bodyGroup.children) {
      this._designScene.registerMesh(regNodeId, child as THREE.Mesh);
    }

    // Update side panel (specs section — no layer checkboxes)
    this._updateSidePanel(comp);

    // Build ComponentTree in a container
    const treeContainer = this._getOrCreateTreeContainer();
    this._componentTree = new ComponentTree(
      treeContainer,
      this._manifest,
      (_nodeId) => {
        /* select callback — no-op for now */
      },
      {
        onToggleNodeHidden: (nodeId) => this._onTreeToggle(nodeId),
        onSolo: (nodeId) => this._onTreeSolo(nodeId),
        onUnsolo: () => this._onTreeUnsolo(),
        onCategoryToggle: (category, visible) => this._onCategoryToggle(category, visible),
        onShapeScript: (url) => {
          // Route ShapeScript links to component endpoint instead of bot endpoint
          window.location.href = url.replace(/\?cadsteps=/, `?component=${name}&cadsteps=`);
        },
      },
    );
    this._componentTree.build();

    // Hide the redundant body node — meshes are registered under the mount node
    const bodyRow = treeContainer.querySelector(`[data-node-id="body:${name}"]`);
    if (bodyRow) (bodyRow as HTMLElement).style.display = 'none';

    // Build a set of mount node IDs that are design layers or clearances
    const hiddenMountIds = new Set<string>();
    for (const m of this._manifest.mounts ?? []) {
      if (m.category === 'design_layer' || m.category === 'clearance') {
        hiddenMountIds.add(`mount:${m.body}:${m.label}`);
      }
    }

    // Set fasteners, wires, design layers, and clearances hidden by default
    // (by ID prefix / set membership, not node kind, because horns are also SubPart
    // but should stay visible).
    for (const node of this._designScene.tree.nodes.values()) {
      if (node.id.startsWith('fastener-group:') || node.id.startsWith('wire-group:') || hiddenMountIds.has(node.id)) {
        node.hidden = true;
      }
    }
    this._designScene.syncVisibility();
    this._componentTree.updateFromDesignScene(this._designScene.tree);

    // Section cap color from manifest materials
    this.viewport.setSectionCapColorFn((groupName: string) => {
      if (!this._manifest) return null;

      // Body groups → use component color
      if (groupName.startsWith('body:')) {
        const c = this.currentComponent;
        if (!c) return 0xced9e0;
        return tintColor(new THREE.Color(c.color[0], c.color[1], c.color[2]), 0.6);
      }

      // Fastener groups → gold
      if (groupName.startsWith('fastener')) return tintColor(0xd4a843, 0.6);

      // Wire groups → purple
      if (groupName.startsWith('wire') || groupName.startsWith('connector')) return tintColor(0x9179f2, 0.6);

      // Default
      return tintColor(0xced9e0, 0.6);
    });

    this._fitCameraToVisibleMeshes();
  }

  // -----------------------------------------------------------------------
  // Tree callbacks
  // -----------------------------------------------------------------------

  _getOrCreateTreeContainer(): HTMLElement {
    let container = document.getElementById('component-tree-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'component-tree-container';
      // Insert at the top of the side panel
      const panel = document.getElementById('side-panel');
      panel.insertBefore(container, panel.firstChild);
    }
    container.innerHTML = '';
    return container;
  }

  async _onTreeToggle(nodeId: string) {
    if (!this._designScene) return;
    const node = this._designScene.tree.getNode(nodeId);
    if (!node) return;

    node.hidden = !node.hidden;

    // Lazy-load mesh if this is the first time making it visible
    if (!node.hidden && node.meshIds.length === 0) {
      await this._lazyLoadNodeMesh(nodeId);
    }

    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);

    // Re-fit camera and section
    const box = this._getVisibleBBox();
    if (box) this.viewport._fitOrthoFrustum(box);
    this.controls.update();
    if (this.sectionEnabled) this._updateSectionPlane();
  }

  _onTreeSolo(nodeId: string) {
    if (!this._designScene) return;
    this._designScene.tree.solo(nodeId);
    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);
  }

  _onTreeUnsolo() {
    if (!this._designScene) return;
    this._designScene.tree.unsolo();
    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);
  }

  _onCategoryToggle(category: string, visible: boolean) {
    if (!this._designScene) return;
    // Toggle hidden on all nodes of this category (nodes is a Map)
    for (const node of this._designScene.tree.nodes.values()) {
      const nodeCategory = this._nodeCategoryForFilter(node);
      if (nodeCategory === category) {
        node.hidden = !visible;
      }
    }
    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);
  }

  _nodeCategoryForFilter(node: any): string | null {
    // Map node kinds to filter categories.
    if (node.kind === NodeKind.Body) return 'body';
    if (node.kind === NodeKind.SubPart) {
      if (node.id.startsWith('fastener')) return 'fastener';
    }
    // Wire groups are NodeKind.Component with `wire-group:` prefix
    if (node.id.startsWith('wire-group:')) return 'wire';
    if (node.kind === NodeKind.Component) {
      // Check if this is a design layer or clearance mount
      if (this._manifest) {
        const mounts = this._manifest.mounts ?? [];
        const mount = mounts.find((m: ManifestMount) => `mount:${m.body}:${m.label}` === node.id);
        if (mount?.category === 'design_layer') return 'design_layer';
        if (mount?.category === 'clearance') return 'clearance';
      }
      return 'mount';
    }
    return null;
  }

  // -----------------------------------------------------------------------
  // Lazy-load mesh for tree nodes
  // -----------------------------------------------------------------------

  async _lazyLoadNodeMesh(nodeId: string) {
    if (!this._manifest || !this._designScene) return;
    const compName = this._manifest.bot_name;
    const allParts = this._manifest.parts ?? [];

    // Match node ID to manifest parts.
    // buildSceneTree generates these ID patterns:
    //   fastener-group:mount:{body}:{label}:{groupKey}  (fasteners on a mount)
    //   wire-group:{body}                                (all wires on a body)
    let matchingParts: ManifestPart[];

    if (nodeId.startsWith('wire-group:')) {
      // Wire group: load ALL wire parts for this body
      matchingParts = allParts.filter((p) => p.category === 'wire');
    } else if (nodeId.startsWith('mount:')) {
      // Design layer mount — find the matching mount in manifest
      const mounts = this._manifest.mounts ?? [];
      const mount = mounts.find((m: ManifestMount) => `mount:${m.body}:${m.label}` === nodeId);
      if (!mount) return;

      let group = this._meshGroups[nodeId];
      if (!group) {
        group = this.viewport.addGroup(nodeId);
        this._meshGroups[nodeId] = group;
      }

      const stlUrl = `/api/components/${compName}/stl/${mount.mesh}`;
      try {
        const stlResp = await fetch(stlUrl);
        if (!stlResp.ok) return;
        const buf = await stlResp.arrayBuffer();
        const geometry = this.stlLoader.parse(buf);
        geometry.computeVertexNormals();

        let color: THREE.Color;
        const opts: Record<string, unknown> = {};
        if (mount.color) {
          color = new THREE.Color(mount.color[0], mount.color[1], mount.color[2]);
          if (mount.color[3] !== undefined && mount.color[3] < 1) {
            opts.transparent = true;
            opts.opacity = mount.color[3];
          }
        } else {
          color = new THREE.Color(0.5, 0.5, 0.5);
        }

        addMeshWithEdges(geometry, tintColor(color), group, opts);

        for (const child of group.children) {
          this._designScene.registerMesh(nodeId, child as THREE.Mesh);
        }
      } catch (err) {
        console.warn(`Failed to load mount mesh for ${nodeId}:`, err);
      }
      return;
    } else if (nodeId.startsWith('fastener-group:')) {
      // Fastener group: match by group key (last segment of node ID).
      // groupKey comes from groupFasteners() in utils.ts — it's the part name.
      const segments = nodeId.split(':');
      const groupKey = segments[segments.length - 1];
      matchingParts = allParts.filter((p) => p.category === 'fastener' && p.name === groupKey);
    } else {
      return; // Unknown node type
    }

    if (matchingParts.length === 0) return;

    // Create or find the viewport group for this node
    let group = this._meshGroups[nodeId];
    if (!group) {
      group = this.viewport.addGroup(nodeId);
      this._meshGroups[nodeId] = group;
    }

    // Deduplicate STL URLs, batch-load geometries
    const stlUrlForPart = (part: ManifestPart) => {
      // Meshes starting with '_' are shared synthetic components (_screw_*, _connector_*, _wire_stub)
      const meshName = part.mesh;
      return meshName.startsWith('_')
        ? `/api/components/${meshName}/stl/body`
        : `/api/components/${compName}/stl/${meshName}`;
    };

    const uniqueUrls = [...new Set(matchingParts.map(stlUrlForPart))];
    const geomCache: Record<string, THREE.BufferGeometry> = {};
    await Promise.all(
      uniqueUrls.map(async (url) => {
        try {
          const stlResp = await fetch(url);
          if (!stlResp.ok) return;
          const buf = await stlResp.arrayBuffer();
          const geom = this.stlLoader.parse(buf);
          geom.computeVertexNormals();
          geomCache[url] = geom;
        } catch {
          /* skip */
        }
      }),
    );

    // Place each part with its position/quaternion
    for (const part of matchingParts) {
      const url = stlUrlForPart(part);
      const srcGeom = geomCache[url];
      if (!srcGeom) continue;

      let color: THREE.Color | number;
      if (part.category === 'fastener') {
        color = tintColor(0xd4a843);
      } else if (part.color) {
        color = tintColor(new THREE.Color(part.color[0], part.color[1], part.color[2]));
      } else {
        color = tintColor(0x808080);
      }

      const mat = createMaterial(color);
      const mesh = new THREE.Mesh(srcGeom.clone(), mat);
      mesh.castShadow = true;
      if (part.pos) mesh.position.set(part.pos[0], part.pos[1], part.pos[2]);
      if (part.quat) mesh.quaternion.set(part.quat[1], part.quat[2], part.quat[3], part.quat[0]);
      group.add(mesh);

      this._designScene.registerMesh(nodeId, mesh);
    }
  }

  // -----------------------------------------------------------------------
  // Multi-material mount helper
  // -----------------------------------------------------------------------

  async _loadMultiMaterialMount(compName: string, mount: ManifestMount, group: THREE.Group) {
    if (!mount.meshes) return;
    const materials = this._manifest?.materials ?? {};

    await Promise.all(
      mount.meshes.map(async (meshEntry) => {
        const stlUrl = `/api/components/${compName}/stl/${meshEntry.file}`;
        try {
          const stlResp = await fetch(stlUrl);
          if (!stlResp.ok) return;
          const buf = await stlResp.arrayBuffer();
          const geometry = this.stlLoader.parse(buf);
          geometry.computeVertexNormals();

          const mat = materials[meshEntry.material];
          const color = mat
            ? new THREE.Color(mat.color[0], mat.color[1], mat.color[2])
            : new THREE.Color(0.5, 0.5, 0.5);
          addMeshWithEdges(geometry, tintColor(color), group, {
            metalness: mat?.metallic ?? 0,
            roughness: mat?.roughness ?? 0.8,
          });
        } catch {
          // skip failed material meshes
        }
      }),
    );
  }

  async _addSTLMesh(componentName: string, partName: string, color: any, opts: any, group: THREE.Group) {
    const url = `/api/components/${componentName}/stl/${partName}`;

    let geometry: THREE.BufferGeometry;
    if (this.stlCache[url]) {
      geometry = this.stlCache[url].clone();
    } else {
      try {
        const response = await fetch(url);
        if (!response.ok) return;
        const buffer = await response.arrayBuffer();
        geometry = this.stlLoader.parse(buffer);
        this.stlCache[url] = geometry;
        geometry = geometry.clone();
      } catch {
        return;
      }
    }

    geometry.computeVertexNormals();
    addMeshWithEdges(geometry, color, group, opts);
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
      const resp = await fetch(`/api/components/${this.currentComponent.name}/shapescript`);
      if (!resp.ok) {
        if (resp.status === 404) {
          this._showStepsUnavailable();
        } else {
          console.error('ShapeScript steps fetch failed:', resp.status);
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
      console.error('Failed to fetch ShapeScript steps:', err);
    } finally {
      if (btn) {
        btn.textContent = 'Steps';
        btn.disabled = false;
      }
    }
  }

  _showStepsUnavailable() {
    const panel = document.getElementById('side-panel');
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

    for (const group of Object.values(this._meshGroups)) {
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
    // Restore mesh group visibility
    for (const group of Object.values(this._meshGroups)) {
      group.visible = true;
    }
    // Re-apply design scene visibility (respects hidden nodes)
    if (this._designScene) {
      this._designScene.syncVisibility();
    }

    const comp = this.currentComponent;
    if (comp) this._updateSidePanel(comp);
    this._fitCameraToVisibleMeshes();
  }

  _buildStepsPanel() {
    const panel = document.getElementById('side-panel');
    const comp = this.currentComponent;
    const steps = this.stepsData.steps;

    let html = `<h2>${comp.name}</h2>`;
    html += `<div class="prop-badge" style="margin-bottom:12px">ShapeScript Steps</div>`;

    html += `<div id="step-info" class="step-info"><div class="step-title">Loading...</div></div>`;

    html += `<div class="slider-group">`;
    html += `<div class="slider-label"><span class="name">Step</span><span class="value" id="step-value">0</span></div>`;
    html += `<input type="range" id="steps-slider" min="0" max="${steps.length - 1}" value="${steps.length - 1}">`;
    html += `</div>`;

    html += `<div style="display:flex;gap:8px;margin-bottom:12px">`;
    html += `<button class="btn" id="step-prev">Prev</button>`;
    html += `<button class="btn" id="step-next">Next</button>`;
    html += `</div>`;

    html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">`;
    html += `<input type="checkbox" id="step-tool-toggle" ${this.showStepTool ? 'checked' : ''}>`;
    html += `<label for="step-tool-toggle" style="font-size:12px;color:var(--bp-gray1);cursor:pointer">Show tool solid</label>`;
    html += `</div>`;

    html += `<h3>All Steps</h3>`;
    html += `<div id="steps-list">`;
    const OP_COLORS = { create: '#2B95D6', cut: '#DB3737', union: '#0F9960' };
    for (const step of steps) {
      html += `<div class="step-row" data-step-idx="${step.index}" style="font-size:12px;padding:4px 6px;border-radius:4px;cursor:pointer;margin-bottom:2px;display:flex;align-items:center;gap:6px;transition:background 0.1s">`;
      html += `<span style="width:8px;height:8px;border-radius:50%;flex-shrink:0;background:${OP_COLORS[step.op] || '#5C7080'}"></span>`;
      html += `<span style="color:var(--bp-dark-gray5);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:'Input Sans Narrow','SF Mono',monospace;font-size:11px">${step.script || step.label}</span>`;
      html += `</div>`;
    }
    html += `</div>`;

    html += `<div style="margin-top:16px"><button class="btn" id="steps-exit">Exit Steps</button></div>`;

    panel.innerHTML = html;

    const sliderEl = document.getElementById('steps-slider') as HTMLInputElement;
    sliderEl.addEventListener('input', () => this._showComponentStep(parseInt(sliderEl.value, 10)));

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
        const idx = parseInt(rowEl.dataset.stepIdx!, 10);
        sliderEl.value = String(idx);
        this._showComponentStep(idx);
      });
      rowEl.addEventListener('mouseenter', () => {
        rowEl.style.background = 'rgba(206,217,224,0.5)';
      });
      rowEl.addEventListener('mouseleave', () => {
        if (parseInt(rowEl.dataset.stepIdx!, 10) !== this.stepsCurrentIdx) {
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
        parseInt((row as HTMLElement).dataset.stepIdx!, 10) === idx ? 'rgba(19,124,189,0.15)' : '';
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
    const box = geometry.boundingBox;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    const preset = VIEW_PRESETS.iso;
    const dist = maxDim * 2;
    this.camera.position.copy(center).addScaledVector(preset.dir, dist);
    this.camera.up.copy(preset.up);
    this.controls.target.copy(center);
    this.camera.lookAt(center);
    this.controls.enableRotate = true;

    this.viewport._fitOrthoFrustum(box);
    this.controls.update();
  }

  // -----------------------------------------------------------------------
  // Section plane — delegates to Viewport3D
  // -----------------------------------------------------------------------

  _updateSectionPlane() {
    if (this.sectionEnabled) {
      // Compute plane from our UI state and pass to viewport
      const box = this._getVisibleBBox();
      if (box) {
        const axisIdx = AXIS_INDEX[this.sectionAxis];
        const axisMin = box.min.getComponent(axisIdx);
        const axisMax = box.max.getComponent(axisIdx);
        const pos = axisMin + (axisMax - axisMin) * this.sectionFraction;
        const sign = this.sectionFlipped ? 1 : -1;
        const normal = new THREE.Vector3();
        normal.setComponent(axisIdx, sign);

        // Configure viewport's section state and trigger rebuild
        this.viewport._secOn = true;
        this.viewport._secAxis = this.sectionAxis;
        this.viewport._secFrac = this.sectionFraction;
        this.viewport._secPlane.normal.copy(normal);
        this.viewport._secPlane.constant = pos * -sign;

        // Apply clipping and rebuild caps
        this.viewport._applySection();

        // Update section viz position (the _applySection uses its own logic,
        // but we need to handle flipped normal)
        if (this.viewport._secViz) {
          const center = box.getCenter(new THREE.Vector3());
          const sz = box.getSize(new THREE.Vector3());
          const planeSize = Math.max(sz.x, sz.y, sz.z) * 1.5;
          this.viewport._secViz.scale.set(planeSize, planeSize, 1);
          this.viewport._secViz.visible = true;
          center.setComponent(axisIdx, pos);
          this.viewport._secViz.position.copy(center);
          this.viewport._secViz.rotation.set(0, 0, 0);
          if (axisIdx === 0) this.viewport._secViz.rotation.y = Math.PI / 2;
          else if (axisIdx === 1) this.viewport._secViz.rotation.x = Math.PI / 2;
        }
      }
    } else {
      // Disable section in viewport
      this.viewport._secOn = false;
      if (this.viewport._secViz) this.viewport._secViz.visible = false;
      this.viewport._clearSectionCaps();
      this.viewport.scene.traverse((ch) => {
        if (ch.material) ch.material.clippingPlanes = [];
      });
    }
  }

  _resetSection() {
    if (!this.sectionEnabled) return;
    this.sectionEnabled = false;
    this.sectionFlipped = false;
    const sectionBtn = document.querySelector('#section-toggle');
    if (sectionBtn) sectionBtn.classList.remove('active');
    const flipBtn = document.querySelector('#section-flip');
    if (flipBtn) flipBtn.classList.remove('active');
    const controls = document.querySelector('#section-controls') as HTMLElement | null;
    if (controls) controls.style.display = 'none';
    this._updateSectionPlane();
  }

  // -----------------------------------------------------------------------
  // Server-side SVG render
  // -----------------------------------------------------------------------

  async _requestSVGRender() {
    const comp = this.currentComponent;
    if (!comp) return;

    const layers = Object.keys(this._meshGroups).filter((id) => this._meshGroups[id].visible);
    if (layers.length === 0) return;

    this.camera.updateMatrixWorld();
    const dir = new THREE.Vector3();
    dir.setFromMatrixColumn(this.camera.matrixWorld, 2);
    const up = new THREE.Vector3();
    up.setFromMatrixColumn(this.camera.matrixWorld, 1);

    const preset = VIEW_PRESETS[this.activePreset];
    const viewLabel = preset ? preset.label : 'Custom';

    const body: any = {
      view_dir: [dir.x, dir.y, dir.z],
      view_up: [up.x, up.y, up.z],
      layers,
      annotate: {
        component: comp.name,
        view: viewLabel,
        layers: layers,
        dimensions_mm: comp.dimensions_mm,
        mass_g: comp.mass_g,
      },
    };

    if (this.sectionEnabled) {
      const box = this._getVisibleBBox();
      if (box) {
        const axisIdx = AXIS_INDEX[this.sectionAxis];
        const pos =
          box.min.getComponent(axisIdx) +
          (box.max.getComponent(axisIdx) - box.min.getComponent(axisIdx)) * this.sectionFraction;
        body.section = {
          axis: this.sectionAxis,
          position: pos,
          flipped: this.sectionFlipped,
        };
        body.annotate.section = `Section ${this.sectionAxis.toUpperCase()} @ ${(pos * 1000).toFixed(1)} mm`;
      }
    }

    const btn = document.querySelector('#render-svg') as HTMLButtonElement;
    const origText = btn.textContent;
    btn.textContent = 'Rendering...';
    btn.disabled = true;

    try {
      const resp = await fetch(`/api/components/${comp.name}/render-svg`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (resp.ok) {
        const svgText = await resp.text();
        this._showSVGModal(svgText, comp.name, viewLabel);
      } else {
        console.error('SVG render failed:', await resp.text());
      }
    } catch (err) {
      console.error('SVG render request failed:', err);
    } finally {
      btn.textContent = origText;
      btn.disabled = false;
    }
  }

  _showSVGModal(svgText: string, componentName: string, viewLabel: string) {
    const modal = document.getElementById('svg-modal');
    const title = document.getElementById('svg-modal-title');
    const body = document.getElementById('svg-modal-body');

    title.textContent = `${componentName} — ${viewLabel} view`;
    body.innerHTML = svgText.replace(/\s+width="[^"]*"/, '').replace(/\s+height="[^"]*"/, '');

    this._lastSvgText = svgText;
    this._lastSvgName = `${componentName}_${viewLabel.toLowerCase()}`;

    modal.style.display = 'flex';

    const close = () => {
      modal.style.display = 'none';
    };
    (document.querySelector('#svg-modal .modal-backdrop') as HTMLElement).onclick = close;
    document.getElementById('svg-modal-close')!.onclick = close;

    document.getElementById('svg-modal-download')!.onclick = () => {
      const blob = new Blob([this._lastSvgText], { type: 'image/svg+xml' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${this._lastSvgName}.svg`;
      a.click();
      URL.revokeObjectURL(a.href);
    };
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
    const origAnimCb = this.viewport._animCb;
    this.viewport.animate(() => {
      if (origAnimCb) origAnimCb();
      this._updateAxisGizmo();
    });
  }

  _updateAxisGizmo() {
    const svg = this._gizmoSvg;
    if (!svg) return;

    const cx = 50,
      cy = 50;
    const len = 32;
    svg.innerHTML = '';

    this.camera.updateMatrixWorld();
    const invMat = this.camera.matrixWorld.clone().invert();

    const projected = this._gizmoAxes
      .map((axis) => {
        const d = axis.dir.clone().transformDirection(invMat);
        return { ...axis, d, depth: d.z };
      })
      .sort((a, b) => b.depth - a.depth);

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
