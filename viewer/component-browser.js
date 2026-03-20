/**
 * MindSim Component Browser — Three.js-based component viewer.
 *
 * Orthographic camera with view presets (Front/Side/Top/Iso) and a
 * checkbox-based layer system for inspecting component geometry.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { clearGroup, orientToAxis } from './utils.js';
import { tintColor, createMaterial, addMeshWithEdges } from './presentation.js';
import { MeasureTool } from './measure-tool.js';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const SIDEBAR_WIDTH = 220;
const SIDE_PANEL_WIDTH = 320;

// ---------------------------------------------------------------------------
// View presets — camera direction (from center) and up vector
// ---------------------------------------------------------------------------
const VIEW_PRESETS = {
  front: { dir: new THREE.Vector3(0, -1, 0), up: new THREE.Vector3(0, 0, 1), label: 'Front' },
  side:  { dir: new THREE.Vector3(1, 0, 0),  up: new THREE.Vector3(0, 0, 1), label: 'Side' },
  top:   { dir: new THREE.Vector3(0, 0, 1),  up: new THREE.Vector3(0, 1, 0), label: 'Top' },
  iso:   { dir: new THREE.Vector3(1, -1, 0.8).normalize(), up: new THREE.Vector3(0, 0, 1), label: 'Iso' },
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
    this.sectionPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this.sectionFraction = 0.5;   // slider position (0–1 along bbox)
  }

  async init() {
    this._setupThreeJS();
    this._setupViewToolbar();
    await this._fetchCatalog();
    this._buildSidebar();
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

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
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
    const toolbar = document.getElementById('view-toolbar');
    if (!toolbar) return;

    for (const [key, preset] of Object.entries(VIEW_PRESETS)) {
      const btn = toolbar.querySelector(`[data-view="${key}"]`);
      if (btn) {
        btn.addEventListener('click', () => this._setViewPreset(key));
      }
    }

    // Measure tool toggle
    const measureBtn = toolbar.querySelector('#measure-toggle');
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

    // Clear measurements
    const clearBtn = toolbar.querySelector('#measure-clear');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        this.measureTool.clearAll();
      });
    }

    // Section plane controls
    const sectionToggle = toolbar.querySelector('#section-toggle');
    if (sectionToggle) {
      sectionToggle.addEventListener('click', () => {
        this.sectionEnabled = !this.sectionEnabled;
        sectionToggle.classList.toggle('active', this.sectionEnabled);
        const controls = toolbar.querySelector('#section-controls');
        if (controls) controls.style.display = this.sectionEnabled ? 'flex' : 'none';
        this._updateSectionPlane();
      });
    }

    for (const axis of ['x', 'y', 'z']) {
      const btn = toolbar.querySelector(`[data-section-axis="${axis}"]`);
      if (btn) {
        btn.addEventListener('click', () => {
          this.sectionAxis = axis;
          toolbar.querySelectorAll('[data-section-axis]').forEach(b =>
            b.classList.toggle('active', b.dataset.sectionAxis === axis));
          this._updateSectionPlane();
        });
      }
    }

    const slider = toolbar.querySelector('#section-slider');
    if (slider) {
      slider.addEventListener('input', () => {
        this.sectionFraction = parseFloat(slider.value) / 100;
        this._updateSectionPlane();
      });
    }
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

  _updatePresetButtons() {
    const toolbar = document.getElementById('view-toolbar');
    if (!toolbar) return;
    toolbar.querySelectorAll('[data-view]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === this.activePreset);
    });
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

  _buildSidebar() {
    const sidebar = document.getElementById('component-sidebar');
    sidebar.innerHTML = '';

    const groups = {};
    for (const comp of this.components) {
      const cat = comp.category;
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(comp);
    }

    for (const [category, comps] of Object.entries(groups)) {
      const title = document.createElement('div');
      title.className = 'sidebar-title';
      title.textContent = category;
      sidebar.appendChild(title);

      for (const comp of comps) {
        const btn = document.createElement('div');
        btn.className = 'component-item';
        btn.innerHTML = `${comp.name}<span class="item-category">${comp.dimensions_mm.map(d => d.toFixed(1)).join(' x ')} mm</span>`;
        btn.addEventListener('click', () => this.loadComponent(comp.name));
        btn.dataset.name = comp.name;
        sidebar.appendChild(btn);
      }
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

    this._fitCameraToVisibleMeshes();
  }

  async loadComponent(name) {
    const comp = this.components.find(c => c.name === name);
    if (!comp) return;

    this.currentComponent = comp;

    // Update sidebar selection
    document.querySelectorAll('.component-item').forEach(el => {
      el.classList.toggle('active', el.dataset.name === name);
    });

    // Update top bar
    document.getElementById('bot-name').textContent = name;
    document.getElementById('mode-tabs').style.display = 'none';

    // Clear measurements, section, and layers
    this.measureTool.clearAll();
    if (this.sectionEnabled) {
      this.sectionEnabled = false;
      const sectionBtn = document.querySelector('#section-toggle');
      if (sectionBtn) sectionBtn.classList.remove('active');
      const controls = document.querySelector('#section-controls');
      if (controls) controls.style.display = 'none';
      this._removeSectionViz();
    }
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
      const axisIdx = { x: 0, y: 1, z: 2 }[this.sectionAxis];
      const axisMin = [min.x, min.y, min.z][axisIdx];
      const axisMax = [max.x, max.y, max.z][axisIdx];
      const pos = axisMin + (axisMax - axisMin) * this.sectionFraction;

      // Plane normal points in -axis direction (clips geometry on the + side)
      const normal = new THREE.Vector3();
      normal.setComponent(axisIdx, -1);
      this.sectionPlane.normal.copy(normal);
      this.sectionPlane.constant = pos;

      this._updateSectionViz(box, axisIdx, pos);
    } else {
      this._removeSectionViz();
    }

    // Apply clipping planes to all mesh materials in layer groups
    for (const id of ALL_LAYER_IDS) {
      this.layerGroups[id].traverse(child => {
        if (child.isMesh && child.material) {
          child.material.clippingPlanes = clips;
          child.material.clipShadows = true;
        }
      });
    }
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
      this._sectionViz.renderOrder = 999;
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

  _removeSectionViz() {
    if (this._sectionViz) {
      this._sectionViz.visible = false;
    }
  }

  // -----------------------------------------------------------------------
  // Animation loop
  // -----------------------------------------------------------------------

  _animate() {
    const loop = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
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
