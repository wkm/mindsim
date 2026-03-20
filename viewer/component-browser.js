/**
 * MindSim Component Browser — Three.js-based component viewer.
 *
 * Orthographic camera with view presets (Front/Side/Top/Iso) and a
 * checkbox-based layer system for inspecting component geometry.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { clearGroup, orientToAxis } from './utils.js';
import {
  BP, RENDER_ORDER, SECTION_STENCIL_BASE, SECTION_STENCIL_STRIDE,
  hexStr, tintColor, createMaterial, addMeshWithEdges,
} from './presentation.js';
import { MeasureTool } from './measure-tool.js';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const SIDEBAR_WIDTH = 0;
const SIDE_PANEL_WIDTH = 320;

// ---------------------------------------------------------------------------
// View presets — camera direction (from center) and up vector
// ---------------------------------------------------------------------------
const VIEW_PRESETS = {
  iso:    { dir: new THREE.Vector3(1, -1, 0.8).normalize(), up: new THREE.Vector3(0, 0, 1), label: 'Iso',    key: '1' },
  front:  { dir: new THREE.Vector3(0, -1, 0), up: new THREE.Vector3(0, 0, 1), label: 'Front',  key: '2' },
  back:   { dir: new THREE.Vector3(0, 1, 0),  up: new THREE.Vector3(0, 0, 1), label: 'Back',   key: '3' },
  top:    { dir: new THREE.Vector3(0, 0, 1),  up: new THREE.Vector3(0, 1, 0), label: 'Top',    key: '4' },
  bottom: { dir: new THREE.Vector3(0, 0, -1), up: new THREE.Vector3(0, -1, 0), label: 'Bottom', key: '5' },
  right:  { dir: new THREE.Vector3(1, 0, 0),  up: new THREE.Vector3(0, 0, 1), label: 'Right',  key: '6' },
  left:   { dir: new THREE.Vector3(-1, 0, 0), up: new THREE.Vector3(0, 0, 1), label: 'Left',   key: '7' },
};

// ---------------------------------------------------------------------------
// Layer definitions
// ---------------------------------------------------------------------------
// True entity colors — the actual physical color of each part.
// At render time these are tinted via presentation.tintColor() so
// edges and annotations remain visible against lighter geometry.
const LAYER_META = {
  body:             { label: 'Body',             colorHex: null,     opts: {} },
  servo:            { label: 'Servo',            colorHex: 0x182026, opts: {} },
  horn:             { label: 'Horn',             colorHex: 0xE8E8E8, opts: {} },
  bracket:          { label: 'Bracket',          colorHex: 0xCED9E0, opts: {} },
  cradle:           { label: 'Cradle',           colorHex: 0xCED9E0, opts: {} },
  coupler:          { label: 'Coupler',          colorHex: 0xF55656, opts: {} },
  bracket_envelope: { label: 'Bracket Envelope', colorHex: 0xF55656, opts: { transparent: true, opacity: 0.25 } },
  cradle_envelope:  { label: 'Cradle Envelope',  colorHex: 0xF55656, opts: { transparent: true, opacity: 0.25 } },
  fasteners:        { label: 'Fasteners',        colorHex: 0xD4A843, opts: {} },
};
const ALL_LAYER_IDS = Object.keys(LAYER_META);
const AXIS_INDEX = { x: 0, y: 1, z: 2 };

// ---------------------------------------------------------------------------
// ComponentBrowser class
// ---------------------------------------------------------------------------
class ComponentBrowser {
  constructor() {
    this.components = [];
    this.currentComponent = null;
    this.stlLoader = new STLLoader();
    this.stlCache = {};       // url → BufferGeometry
    this.layerGroups = {};    // layer id → THREE.Group
    this.activePreset = 'iso';

    // Section plane state
    this.sectionEnabled = false;
    this.sectionAxis = 'z';       // 'x', 'y', or 'z'
    this.sectionFlipped = false;
    this.sectionPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.sectionFraction = 0.5;   // slider position (0–1 along bbox)
  }

  async init() {
    this._setupThreeJS();
    this._setupViewToolbar();
    this._setupAxisGizmo();
    await this._fetchCatalog();
    this._buildSidePanel();
    this._animate();
  }

  // -----------------------------------------------------------------------
  // Three.js setup — orthographic camera
  // -----------------------------------------------------------------------

  _setupThreeJS() {
    const container = document.getElementById('canvas-container');
    container.style.left = SIDEBAR_WIDTH + 'px';
    container.style.right = SIDE_PANEL_WIDTH + 'px';

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xF5F8FA);

    const vw = window.innerWidth - SIDEBAR_WIDTH - SIDE_PANEL_WIDTH;
    const vh = window.innerHeight;
    const aspect = vw / vh;

    // Orthographic camera — frustum sized to a typical component (~0.1m)
    const frustum = 0.06;
    this.camera = new THREE.OrthographicCamera(
      -frustum * aspect, frustum * aspect,
      frustum, -frustum,
      0.0001, 10,
    );
    this.camera.position.set(0.06, -0.06, 0.05);
    this.camera.up.set(0, 0, 1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, stencil: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(vw, vh);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.localClippingEnabled = true;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.enableRotate = true;
    this.controls.update();

    // Measurement tool
    this.measureTool = new MeasureTool(this.camera, this.scene, container);

    // Lighting
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
    dirLight.position.set(0.2, 0.4, 0.3);
    dirLight.castShadow = true;
    this.scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0xccccff, 0.5);
    fillLight.position.set(-0.2, 0.1, -0.2);
    this.scene.add(fillLight);

    // Grid (200mm with 5mm divisions)
    this.scene.add(new THREE.GridHelper(0.2, 40, 0xBFCCD6, 0xCED9E0));

    // Create a group per layer
    for (const id of ALL_LAYER_IDS) {
      const g = new THREE.Group();
      g.name = id;
      this.scene.add(g);
      this.layerGroups[id] = g;
    }

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
      // Toggle dropdown
      if (dropdownBtn) {
        dropdownBtn.addEventListener('click', () => {
          dropdown.style.display = dropdown.style.display === 'none' ? '' : 'none';
        });
        // Close on click outside
        document.addEventListener('click', (e) => {
          if (!dropdownBtn.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.style.display = 'none';
          }
        });
      }
    }

    // Measure tool toggle
    const measureBtn = document.getElementById('measure-toggle');
    if (measureBtn) {
      measureBtn.addEventListener('click', () => {
        const active = !this.measureTool.enabled;
        if (active) {
          this.measureTool.enable();
          // Disable orbit when measuring
          this.controls.enabled = false;
        } else {
          this.measureTool.disable();
          this.controls.enabled = true;
        }
        measureBtn.classList.toggle('active', active);
      });
    }

    const clearBtn = document.getElementById('measure-clear');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        this.measureTool.clearAll();
      });
    }

    const renderBtn = document.getElementById('render-svg');
    if (renderBtn) {
      renderBtn.addEventListener('click', () => this._requestSVGRender());
    }

    // Section plane controls
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
          document.querySelectorAll('[data-section-axis]').forEach(b =>
            b.classList.toggle('active', b.dataset.sectionAxis === axis));
          this._updateSectionPlane();
        });
      }
    }

    const slider = document.getElementById('section-slider');
    if (slider) {
      slider.addEventListener('input', () => {
        this.sectionFraction = parseFloat(slider.value) / 100;
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

    // Keyboard shortcuts: 1-6 = views, 0 = iso, M = measure, S = section
    const keyMap = {
      '1': 'iso', '2': 'front', '3': 'back', '4': 'top',
      '5': 'bottom', '6': 'right', '7': 'left',
    };
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
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
      } else if (e.key.startsWith('Arrow')) {
        e.preventDefault();
        this._arrowKeyNav(e.key, e.shiftKey);
      }
    });
  }

  _setViewPreset(key) {
    const preset = VIEW_PRESETS[key];
    if (!preset) return;

    this.activePreset = key;

    // Get bounding box center and size for framing
    const box = this._getVisibleBBox();
    const center = box ? box.getCenter(new THREE.Vector3()) : new THREE.Vector3();
    const size = box ? box.getSize(new THREE.Vector3()) : new THREE.Vector3(0.1, 0.1, 0.1);
    const maxDim = Math.max(size.x, size.y, size.z);

    // Position camera along preset direction
    const dist = maxDim * 2;
    this.camera.position.copy(center).addScaledVector(preset.dir, dist);
    this.camera.up.copy(preset.up);
    this.controls.target.copy(center);
    this.camera.lookAt(center);

    // Lock rotation for axis-aligned views, allow for iso
    this.controls.enableRotate = (key === 'iso');

    this._fitOrthoFrustum(box);
    this.controls.update();
    this._updatePresetButtons();
  }

  /** Orbit or pan the camera via arrow keys. */
  _arrowKeyNav(key, shift) {
    const ORBIT_STEP = Math.PI / 12;  // 15° per keypress
    const target = this.controls.target;
    const pos = this.camera.position;
    const offset = pos.clone().sub(target);

    if (shift) {
      // Shift+Arrow: pan
      const panStep = this.camera.top * 0.15;  // 15% of visible height
      this.camera.updateMatrixWorld();
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      right.setFromMatrixColumn(this.camera.matrixWorld, 0);
      up.setFromMatrixColumn(this.camera.matrixWorld, 1);

      let dx = 0, dy = 0;
      if (key === 'ArrowLeft')  dx = -panStep;
      if (key === 'ArrowRight') dx = panStep;
      if (key === 'ArrowUp')    dy = panStep;
      if (key === 'ArrowDown')  dy = -panStep;

      const move = right.multiplyScalar(dx).add(up.multiplyScalar(dy));
      target.add(move);
      pos.add(move);
    } else {
      // Arrow: orbit
      if (key === 'ArrowLeft' || key === 'ArrowRight') {
        // Horizontal orbit around Z axis
        const angle = key === 'ArrowLeft' ? ORBIT_STEP : -ORBIT_STEP;
        offset.applyAxisAngle(new THREE.Vector3(0, 0, 1), angle);
      } else {
        // Vertical orbit — rotate around camera's local X axis
        const angle = key === 'ArrowUp' ? -ORBIT_STEP : ORBIT_STEP;
        this.camera.updateMatrixWorld();
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
        offset.applyAxisAngle(right, angle);
        // Update up vector to stay coherent
        this.camera.up.applyAxisAngle(right, angle);
      }
      pos.copy(target).add(offset);
      this.camera.lookAt(target);
    }

    // Unlock rotation since we're now in a custom orientation
    this.controls.enableRotate = true;
    this.activePreset = null;
    this._updatePresetButtons();
    this.controls.update();
  }

  _updatePresetButtons() {
    const dropdown = document.getElementById('view-dropdown');
    if (dropdown) {
      dropdown.querySelectorAll('[data-view]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === this.activePreset);
      });
    }
    // Update dropdown button label
    const dropdownBtn = document.getElementById('view-dropdown-btn');
    if (dropdownBtn && this.activePreset) {
      const preset = VIEW_PRESETS[this.activePreset];
      if (preset) {
        dropdownBtn.textContent = `${preset.label} \u25BE`;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Orthographic frustum helpers
  // -----------------------------------------------------------------------

  _getVisibleBBox() {
    const box = new THREE.Box3();
    let hasGeometry = false;

    for (const id of ALL_LAYER_IDS) {
      const group = this.layerGroups[id];
      if (!group.visible) continue;
      group.traverse(child => {
        if (child.isMesh) {
          child.geometry.computeBoundingBox();
          const childBox = child.geometry.boundingBox.clone();
          childBox.applyMatrix4(child.matrixWorld);
          box.union(childBox);
          hasGeometry = true;
        }
      });
    }

    return hasGeometry ? box : null;
  }

  _fitOrthoFrustum(box) {
    if (!box) return;

    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const pad = maxDim * 0.15;  // 15% padding

    const container = document.getElementById('canvas-container');
    const aspect = container.offsetWidth / container.offsetHeight;
    const halfH = (maxDim / 2) + pad;
    const halfW = halfH * aspect;

    this.camera.left = -halfW;
    this.camera.right = halfW;
    this.camera.top = halfH;
    this.camera.bottom = -halfH;
    this.camera.updateProjectionMatrix();
  }

  _fitCameraToVisibleMeshes() {
    const box = this._getVisibleBBox();
    if (!box) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Position along active preset direction
    const preset = VIEW_PRESETS[this.activePreset] || VIEW_PRESETS.iso;
    const dist = maxDim * 2;
    this.camera.position.copy(center).addScaledVector(preset.dir, dist);
    this.camera.up.copy(preset.up);
    this.controls.target.copy(center);
    this.camera.lookAt(center);

    this._fitOrthoFrustum(box);
    this.controls.update();
  }

  _updateLayout() {
    const container = document.getElementById('canvas-container');
    container.style.left = SIDEBAR_WIDTH + 'px';
    container.style.right = SIDE_PANEL_WIDTH + 'px';

    requestAnimationFrame(() => {
      const w = container.offsetWidth;
      const h = container.offsetHeight;
      if (w > 0 && h > 0) {
        // Update ortho frustum aspect ratio
        const aspect = w / h;
        const halfH = this.camera.top;  // preserve current zoom level
        this.camera.left = -halfH * aspect;
        this.camera.right = halfH * aspect;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        // Update Line2 material resolution
        if (this._contourLineMat) {
          this._contourLineMat.resolution.set(w, h);
        }
      }
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

  _buildLayerControls(comp) {
    let html = '<h3>Layers</h3>';
    html += '<div class="layer-controls">';

    for (const id of (comp.layers || [])) {
      const meta = LAYER_META[id];
      if (!meta) continue;

      const checked = this.layerGroups[id].visible ? 'checked' : '';
      const colorSwatch = meta.colorHex !== null
        ? `<span class="layer-swatch" style="background:#${meta.colorHex.toString(16).padStart(6, '0')}"></span>`
        : `<span class="layer-swatch" style="background:rgb(${Math.round(comp.color[0]*255)},${Math.round(comp.color[1]*255)},${Math.round(comp.color[2]*255)})"></span>`;

      html += `<label class="layer-toggle">
        <input type="checkbox" data-layer="${id}" ${checked}>
        ${colorSwatch}
        <span class="layer-label">${meta.label}</span>
      </label>`;
    }

    html += '</div>';
    return html;
  }

  _updateSidePanel(comp) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${comp.name}</h2>`;

    // Layer controls
    html += this._buildLayerControls(comp);

    html += `<h3>General</h3>`;
    html += `<div style="font-size:12px; color:#999; line-height:1.8">`;
    html += `<div>Category: <span style="color:#ccc">${comp.category}</span></div>`;
    html += `<div>Dimensions: <span style="color:#ccc">${comp.dimensions_mm.map(d => d.toFixed(1)).join(' x ')} mm</span></div>`;
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

    // Wire up checkbox listeners
    panel.querySelectorAll('input[data-layer]').forEach(cb => {
      cb.addEventListener('change', () => this._onLayerToggle(cb.dataset.layer, cb.checked));
    });

    // Create 3D markers and wire up hover highlighting
    this._createMarkers(comp);

    panel.querySelectorAll('.highlight-item').forEach(el => {
      el.addEventListener('mouseenter', () => {
        const type = el.dataset.markerType;
        const idx = parseInt(el.dataset.markerIdx);
        this._highlightMarker(type, idx, true);
      });
      el.addEventListener('mouseleave', () => {
        const type = el.dataset.markerType;
        const idx = parseInt(el.dataset.markerIdx);
        this._highlightMarker(type, idx, false);
      });
    });
  }

  // -----------------------------------------------------------------------
  // 3D markers (mounting points, wire ports)
  // -----------------------------------------------------------------------

  _createMarkers(comp) {
    if (this._markerGroup) {
      clearGroup(this._markerGroup);
      this.scene.remove(this._markerGroup);
    }
    this._markerGroup = new THREE.Group();
    this._markerGroup.name = 'markers';
    this.scene.add(this._markerGroup);

    this._markers = { mp: [], wp: [] };

    const markerMat = new THREE.MeshPhysicalMaterial({
      color: 0x4488ff, transparent: true, opacity: 0.4,
      roughness: 0.3, metalness: 0.1,
    });
    const highlightMat = new THREE.MeshPhysicalMaterial({
      color: 0x44ff88, transparent: false, opacity: 1.0,
      roughness: 0.3, metalness: 0.1, emissive: 0x22aa44, emissiveIntensity: 0.5,
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
      color: 0xff8844, transparent: true, opacity: 0.4,
      roughness: 0.3, metalness: 0.1,
    });
    const wpHighlightMat = new THREE.MeshPhysicalMaterial({
      color: 0xffaa22, transparent: false, opacity: 1.0,
      roughness: 0.3, metalness: 0.1, emissive: 0xcc8800, emissiveIntensity: 0.5,
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

  _highlightMarker(type, idx, on) {
    const markers = this._markers?.[type];
    if (!markers || !markers[idx]) return;
    const mesh = markers[idx];
    mesh.material = on ? mesh.userData.highlightMat : mesh.userData.defaultMat;
    const s = on ? 1.8 : 1.0;
    mesh.scale.set(s, s, s);
  }

  // -----------------------------------------------------------------------
  // Layer loading
  // -----------------------------------------------------------------------

  async _onLayerToggle(layerId, enabled) {
    const group = this.layerGroups[layerId];

    if (enabled) {
      if (group.children.length === 0) {
        await this._loadLayer(layerId);
      }
      group.visible = true;
    } else {
      group.visible = false;
    }

    // Re-fit the frustum without changing camera angle
    const box = this._getVisibleBBox();
    this._fitOrthoFrustum(box);
    this.controls.update();

    // Re-apply section plane (clips new geometry, rebuilds caps)
    if (this.sectionEnabled) {
      this._updateSectionPlane();
    }
  }

  async loadComponent(name) {
    const comp = this.components.find(c => c.name === name);
    if (!comp) return;

    this.currentComponent = comp;

    // Update top bar
    document.getElementById('bot-name').textContent = name;
    document.getElementById('mode-tabs').style.display = 'none';

    // Clear measurements, section, and layers
    this.measureTool.clearAll();
    this._resetSection();
    for (const id of ALL_LAYER_IDS) {
      clearGroup(this.layerGroups[id]);
      this.layerGroups[id].visible = false;
    }

    // First layer is the default
    const defaultLayer = (comp.layers && comp.layers[0]) || 'body';

    // Update side panel
    this._updateSidePanel(comp);

    // Load default layer and fit view
    await this._loadLayer(defaultLayer);
    this.layerGroups[defaultLayer].visible = true;
    this._fitCameraToVisibleMeshes();
  }

  async _loadLayer(layerId) {
    const group = this.layerGroups[layerId];
    const comp = this.currentComponent;
    const compName = comp.name;

    if (layerId === 'fasteners') {
      await this._loadFasteners(compName, group);
      return;
    }

    const meta = LAYER_META[layerId] || { colorHex: null, opts: {} };
    const entityColor = meta.colorHex !== null
      ? meta.colorHex
      : new THREE.Color(comp.color[0], comp.color[1], comp.color[2]);

    // Tint: 20% entity color on a light base — keeps edges visible
    const renderColor = tintColor(entityColor);

    await this._addSTLMesh(compName, layerId, renderColor, meta.opts, group);
  }

  async _loadFasteners(compName, group) {
    try {
      const resp = await fetch(`/api/components/${compName}/fasteners`);
      if (!resp.ok) return;
      const data = await resp.json();

      const uniqueUrls = [...new Set(data.fasteners.map(f => f.stl_url))];
      const geomCache = {};
      await Promise.all(uniqueUrls.map(async (url) => {
        const stlResp = await fetch(url);
        if (!stlResp.ok) return;
        const buf = await stlResp.arrayBuffer();
        const geom = this.stlLoader.parse(buf);
        geom.computeVertexNormals();
        geomCache[url] = geom;
      }));

      const fastenerMat = createMaterial(tintColor(0xD4A843));

      for (const f of data.fasteners) {
        const srcGeom = geomCache[f.stl_url];
        if (!srcGeom) continue;

        const mesh = new THREE.Mesh(srcGeom.clone(), fastenerMat);
        mesh.castShadow = true;
        mesh.position.set(f.pos[0], f.pos[1], f.pos[2]);
        if (f.quat) {
          mesh.quaternion.set(f.quat[1], f.quat[2], f.quat[3], f.quat[0]);
        }
        group.add(mesh);
      }
    } catch (err) {
      console.warn('Failed to load fasteners:', err);
    }
  }

  async _addSTLMesh(componentName, partName, color, opts, group) {
    const url = `/api/components/${componentName}/stl/${partName}`;

    let geometry;
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
  // Section plane
  // -----------------------------------------------------------------------

  _updateSectionPlane() {
    const box = this._getVisibleBBox();
    const clips = this.sectionEnabled ? [this.sectionPlane] : [];

    if (this.sectionEnabled && box) {
      // Compute plane position from slider fraction along the chosen axis
      const min = box.min;
      const max = box.max;
      const axisIdx = AXIS_INDEX[this.sectionAxis];
      const axisMin = min.getComponent(axisIdx);
      const axisMax = max.getComponent(axisIdx);
      const pos = axisMin + (axisMax - axisMin) * this.sectionFraction;

      // Plane normal: default clips geometry on the + side; flip reverses
      const sign = this.sectionFlipped ? 1 : -1;
      const normal = new THREE.Vector3();
      normal.setComponent(axisIdx, sign);
      this.sectionPlane.normal.copy(normal);
      this.sectionPlane.constant = pos * -sign;

      this._updateSectionViz(box, axisIdx, pos);
    } else {
      this._removeSectionViz();
    }

    // Apply clipping planes to all meshes AND edge lines in layer groups
    for (const id of ALL_LAYER_IDS) {
      this.layerGroups[id].traverse(child => {
        if (child.material) {
          // Clone shared edge material on first clip so we don't
          // mutate the global instance used by non-clipped viewers
          if (child.isLineSegments && !child.material._clippable) {
            child.material = child.material.clone();
            child.material._clippable = true;
          }
          child.material.clippingPlanes = clips;
          child.material.clipShadows = true;
        }
      });
    }

    // Rebuild stencil-based section caps
    this._rebuildSectionCaps(clips);
  }

  _updateSectionViz(box, axisIdx, pos) {
    if (!this._sectionViz) {
      const geom = new THREE.PlaneGeometry(1, 1);
      const mat = new THREE.MeshBasicMaterial({
        color: 0x2B95D6,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      this._sectionViz = new THREE.Mesh(geom, mat);
      this._sectionViz.renderOrder = RENDER_ORDER.SECTION_VIZ;
      this._sectionViz.raycast = () => {};
      this.scene.add(this._sectionViz);

      // Edge ring around the plane
      const ringGeom = new THREE.EdgesGeometry(geom);
      this._sectionRing = new THREE.LineSegments(ringGeom, new THREE.LineBasicMaterial({
        color: 0x2B95D6, transparent: true, opacity: 0.4,
      }));
      this._sectionRing.raycast = () => {};
      this._sectionViz.add(this._sectionRing);
    }

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const planeSize = Math.max(size.x, size.y, size.z) * 1.5;

    this._sectionViz.scale.set(planeSize, planeSize, 1);
    this._sectionViz.visible = true;

    // Position and orient the plane quad
    center.setComponent(axisIdx, pos);
    this._sectionViz.position.copy(center);

    // Rotate plane to face along the section axis
    this._sectionViz.rotation.set(0, 0, 0);
    if (axisIdx === 0) this._sectionViz.rotation.y = Math.PI / 2;      // X
    else if (axisIdx === 1) this._sectionViz.rotation.x = Math.PI / 2;  // Y
    // Z = default orientation (no rotation needed)
  }

  _resetSection() {
    if (!this.sectionEnabled) return;
    this.sectionEnabled = false;
    this.sectionFlipped = false;
    const sectionBtn = document.querySelector('#section-toggle');
    if (sectionBtn) sectionBtn.classList.remove('active');
    const flipBtn = document.querySelector('#section-flip');
    if (flipBtn) flipBtn.classList.remove('active');
    const controls = document.querySelector('#section-controls');
    if (controls) controls.style.display = 'none';
    this._removeSectionViz();
  }

  _removeSectionViz() {
    if (this._sectionViz) {
      this._sectionViz.visible = false;
    }
    this._clearSectionCaps();
  }

  /**
   * Stencil-based section caps: fills the cross-section interior with
   * a solid color so the cut looks like a real cross-section, not a
   * hollow shell.
   *
   * Technique:
   * 1. For each visible mesh, create two invisible stencil helpers:
   *    - Back-face pass: increments stencil where clipped back faces are visible
   *    - Front-face pass: decrements stencil where clipped front faces are visible
   * 2. A cap plane at the section position renders only where stencil != 0
   *    (i.e., inside the geometry cross-section).
   */
  _rebuildSectionCaps(clips) {
    this._clearSectionCaps();
    if (!clips.length) return;

    this._capGroup = new THREE.Group();
    this._capGroup.name = 'section-caps';
    this.scene.add(this._capGroup);

    const box = this._getVisibleBBox();
    const capSize = box
      ? Math.max(...box.getSize(new THREE.Vector3()).toArray()) * 1.5
      : 0.2;

    const allMeshes = [];
    let layerIndex = 0;

    // Process each layer separately so each gets its own cap color
    for (const id of ALL_LAYER_IDS) {
      const group = this.layerGroups[id];
      if (!group.visible) continue;

      const layerMeshes = [];
      group.traverse(child => {
        if (child.isMesh && child.geometry) {
          layerMeshes.push(child);
        }
      });
      if (layerMeshes.length === 0) continue;

      const ref = layerIndex + 1;  // stencil ref (1-based, 0 = empty)
      const orderBase = SECTION_STENCIL_BASE + layerIndex * SECTION_STENCIL_STRIDE;
      layerIndex++;

      // Stencil helpers for this layer's meshes
      for (const mesh of layerMeshes) {
        this._addStencilHelper(mesh, THREE.BackSide, ref, clips,
          orderBase + RENDER_ORDER.STENCIL_BACK);
        this._addStencilHelper(mesh, THREE.FrontSide, 0, clips,
          orderBase + RENDER_ORDER.STENCIL_FRONT);
      }

      // Cap plane for this layer — darker tint of the layer color
      const meta = LAYER_META[id] || { colorHex: null };
      const entityColor = meta.colorHex !== null
        ? meta.colorHex
        : new THREE.Color(
            this.currentComponent.color[0],
            this.currentComponent.color[1],
            this.currentComponent.color[2],
          );
      // 60% tint for the cap fill (darker than the 50% body tint)
      const capColor = tintColor(entityColor, 0.6);

      const capGeom = new THREE.PlaneGeometry(capSize, capSize);
      const capMat = new THREE.MeshBasicMaterial({
        color: capColor,
        side: THREE.DoubleSide,
        depthWrite: false,
        stencilWrite: true,
        stencilFunc: THREE.EqualStencilFunc,
        stencilRef: ref,
        stencilFail: THREE.KeepStencilOp,
        stencilZFail: THREE.KeepStencilOp,
        stencilZPass: THREE.ZeroStencilOp,  // clear after rendering
      });
      const capPlane = new THREE.Mesh(capGeom, capMat);
      capPlane.raycast = () => {};
      capPlane.renderOrder = orderBase + RENDER_ORDER.STENCIL_CAP;

      if (this._sectionViz) {
        capPlane.position.copy(this._sectionViz.position);
        capPlane.rotation.copy(this._sectionViz.rotation);
      }

      this._capGroup.add(capPlane);
      allMeshes.push(...layerMeshes);
    }

    // Compute cross-section contour lines across all layers
    const contourSegments = [];
    const plane = this.sectionPlane;
    for (const mesh of allMeshes) {
      this._computeMeshPlaneContour(mesh, plane, contourSegments);
    }

    if (contourSegments.length > 0) {
      const lineGeom = new LineSegmentsGeometry();
      lineGeom.setPositions(contourSegments);
      const lineMat = new LineMaterial({
        color: BP.DARK_GRAY3,
        linewidth: 3,
        resolution: new THREE.Vector2(
          this.renderer.domElement.width,
          this.renderer.domElement.height,
        ),
        clippingPlanes: clips,
      });
      const contourLines = new LineSegments2(lineGeom, lineMat);
      contourLines.renderOrder = RENDER_ORDER.SECTION_CONTOUR;
      contourLines.raycast = () => {};
      this._capGroup.add(contourLines);
      this._contourLineMat = lineMat;
    }
  }

  /** Compute line segments where a mesh intersects a plane. */
  _computeMeshPlaneContour(mesh, plane, out) {
    const geom = mesh.geometry;
    const posAttr = geom.getAttribute('position');
    if (!posAttr) return;

    const index = geom.index;
    const matrix = mesh.matrixWorld;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();

    const triCount = index ? index.count / 3 : posAttr.count / 3;

    for (let i = 0; i < triCount; i++) {
      const i0 = index ? index.getX(i * 3) : i * 3;
      const i1 = index ? index.getX(i * 3 + 1) : i * 3 + 1;
      const i2 = index ? index.getX(i * 3 + 2) : i * 3 + 2;

      a.fromBufferAttribute(posAttr, i0).applyMatrix4(matrix);
      b.fromBufferAttribute(posAttr, i1).applyMatrix4(matrix);
      c.fromBufferAttribute(posAttr, i2).applyMatrix4(matrix);

      const da = plane.distanceToPoint(a);
      const db = plane.distanceToPoint(b);
      const dc = plane.distanceToPoint(c);

      // Find edge crossings (sign changes)
      const crossings = [];
      if (da * db < 0) crossings.push(this._planeEdgeIntersect(a, b, da, db));
      if (db * dc < 0) crossings.push(this._planeEdgeIntersect(b, c, db, dc));
      if (dc * da < 0) crossings.push(this._planeEdgeIntersect(c, a, dc, da));

      if (crossings.length === 2) {
        out.push(
          crossings[0].x, crossings[0].y, crossings[0].z,
          crossings[1].x, crossings[1].y, crossings[1].z,
        );
      }
    }
  }

  _planeEdgeIntersect(p1, p2, d1, d2) {
    const t = d1 / (d1 - d2);
    return new THREE.Vector3().lerpVectors(p1, p2, t);
  }

  /** Create an invisible stencil helper mesh that writes to the stencil buffer. */
  _addStencilHelper(sourceMesh, side, stencilRef, clips, renderOrder) {
    const mat = new THREE.MeshBasicMaterial({
      colorWrite: false,
      depthWrite: false,
      side,
      clippingPlanes: clips,
      stencilWrite: true,
      stencilFunc: THREE.AlwaysStencilFunc,
      stencilRef,
      stencilFail: THREE.KeepStencilOp,
      stencilZFail: THREE.KeepStencilOp,
      stencilZPass: THREE.ReplaceStencilOp,
    });
    const mesh = new THREE.Mesh(sourceMesh.geometry, mat);
    mesh.position.copy(sourceMesh.position);
    mesh.quaternion.copy(sourceMesh.quaternion);
    mesh.scale.copy(sourceMesh.scale);
    mesh.renderOrder = renderOrder;
    mesh.raycast = () => {};
    this._capGroup.add(mesh);
  }

  _clearSectionCaps() {
    if (this._capGroup) {
      this._capGroup.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      this.scene.remove(this._capGroup);
      this._capGroup = null;
    }
    this._contourLineMat = null;
  }

  // -----------------------------------------------------------------------
  // Server-side SVG render
  // -----------------------------------------------------------------------

  async _requestSVGRender() {
    const comp = this.currentComponent;
    if (!comp) return;

    const layers = ALL_LAYER_IDS.filter(id => this.layerGroups[id].visible);
    if (layers.length === 0) return;

    // Extract actual view axes from camera world matrix (not camera.up,
    // which is just a lookAt hint and drifts after orbiting)
    this.camera.updateMatrixWorld();
    const dir = new THREE.Vector3();
    dir.setFromMatrixColumn(this.camera.matrixWorld, 2);  // +Z = backward (camera toward scene)
    const up = new THREE.Vector3();
    up.setFromMatrixColumn(this.camera.matrixWorld, 1);   // +Y = screen up

    // Determine view label for annotation
    const preset = VIEW_PRESETS[this.activePreset];
    const viewLabel = preset ? preset.label : 'Custom';

    const body = {
      view_dir: [dir.x, dir.y, dir.z],
      view_up: [up.x, up.y, up.z],
      layers,
      annotate: {
        component: comp.name,
        view: viewLabel,
        layers: layers.map(id => LAYER_META[id]?.label || id),
        dimensions_mm: comp.dimensions_mm,
        mass_g: comp.mass_g,
      },
    };

    if (this.sectionEnabled) {
      const box = this._getVisibleBBox();
      if (box) {
        const axisIdx = AXIS_INDEX[this.sectionAxis];
        const pos = box.min.getComponent(axisIdx) +
          (box.max.getComponent(axisIdx) - box.min.getComponent(axisIdx)) * this.sectionFraction;
        body.section = {
          axis: this.sectionAxis,
          position: pos,
          flipped: this.sectionFlipped,
        };
        body.annotate.section = `Section ${this.sectionAxis.toUpperCase()} @ ${(pos * 1000).toFixed(1)} mm`;
      }
    }

    const btn = document.querySelector('#render-svg');
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

  _showSVGModal(svgText, componentName, viewLabel) {
    const modal = document.getElementById('svg-modal');
    const title = document.getElementById('svg-modal-title');
    const body = document.getElementById('svg-modal-body');

    title.textContent = `${componentName} — ${viewLabel} view`;
    // Strip explicit width/height so SVG scales to fill the modal via CSS
    body.innerHTML = svgText.replace(/\s+width="[^"]*"/, '').replace(/\s+height="[^"]*"/, '');

    // Store for download
    this._lastSvgText = svgText;
    this._lastSvgName = `${componentName}_${viewLabel.toLowerCase()}`;

    modal.style.display = 'flex';

    // Wire close handlers (re-wire each time to keep it simple)
    const close = () => { modal.style.display = 'none'; };
    document.getElementById('svg-modal-backdrop').onclick = close;
    document.getElementById('svg-modal-close').onclick = close;

    document.getElementById('svg-modal-download').onclick = () => {
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
    // Axis definitions: direction in world space, color, label
    this._gizmoAxes = [
      { dir: new THREE.Vector3(1, 0, 0), color: hexStr(BP.RED3),   label: '+X', neg: '-X' },
      { dir: new THREE.Vector3(0, 1, 0), color: hexStr(BP.GREEN3), label: '+Y', neg: '-Y' },
      { dir: new THREE.Vector3(0, 0, 1), color: hexStr(BP.BLUE4),  label: '+Z', neg: '-Z' },
    ];
  }

  _updateAxisGizmo() {
    const svg = this._gizmoSvg;
    if (!svg) return;

    const cx = 50, cy = 50;  // center of the 100×100 SVG
    const len = 32;           // axis line length in px
    svg.innerHTML = '';

    this.camera.updateMatrixWorld();
    const invMat = this.camera.matrixWorld.clone().invert();

    // Sort axes by depth so front-most draws last (on top)
    const projected = this._gizmoAxes.map(axis => {
      const d = axis.dir.clone().transformDirection(invMat);
      return { ...axis, d, depth: d.z };
    }).sort((a, b) => b.depth - a.depth);  // back-to-front

    for (const { d, color, label } of projected) {
      const behind = d.z > 0.2;
      const opacity = behind ? 0.3 : 1.0;
      const lineW = behind ? 1.5 : 2.5;
      const sx = d.x * len;
      const sy = -d.y * len;

      // Axis line
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', cx);
      line.setAttribute('y1', cy);
      line.setAttribute('x2', cx + sx);
      line.setAttribute('y2', cy + sy);
      line.setAttribute('stroke', color);
      line.setAttribute('stroke-width', lineW);
      line.setAttribute('stroke-linecap', 'round');
      line.setAttribute('opacity', opacity);
      svg.appendChild(line);

      // Dot at the tip
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', cx + sx);
      dot.setAttribute('cy', cy + sy);
      dot.setAttribute('r', behind ? 3 : 5);
      dot.setAttribute('fill', color);
      dot.setAttribute('opacity', opacity);
      svg.appendChild(dot);

      // Label next to the dot
      const labelDist = len + 12;
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', cx + d.x * labelDist);
      text.setAttribute('y', cy - d.y * labelDist);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'central');
      text.setAttribute('fill', color);
      text.setAttribute('opacity', opacity);
      text.setAttribute('style', 'font: 600 11px system-ui, -apple-system, sans-serif');
      text.textContent = label;
      svg.appendChild(text);
    }

    // Center dot
    const centerDot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    centerDot.setAttribute('cx', cx);
    centerDot.setAttribute('cy', cy);
    centerDot.setAttribute('r', 2);
    centerDot.setAttribute('fill', '#8A9BA8');
    svg.appendChild(centerDot);
  }

  // -----------------------------------------------------------------------
  // Animation loop
  // -----------------------------------------------------------------------

  _animate() {
    const loop = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this._updateAxisGizmo();
      if (this.measureTool.measurements.length > 0 || this.measureTool._firstPoint) {
        this.measureTool.update();
      }
      requestAnimationFrame(loop);
    };
    loop();
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
export async function initComponentBrowser(componentParam) {
  const browser = new ComponentBrowser();
  await browser.init();

  if (componentParam && componentParam !== 'catalog') {
    await browser.loadComponent(componentParam);
  }
}
