/**
 * MindSim Component Browser — Three.js-based component 3D viewer.
 *
 * Checkbox-based layer system: each part (body, bracket, cradle, coupler,
 * envelopes, fasteners) is an independently toggleable layer. Multiple
 * layers can be visible simultaneously for debugging component geometry.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------
const SIDEBAR_WIDTH = 220;
const SIDE_PANEL_WIDTH = 320;

// ---------------------------------------------------------------------------
// Layer definitions
// ---------------------------------------------------------------------------
const LAYERS = [
  { id: 'body',             label: 'Body',             servoOnly: false, colorHex: null,     opts: {} },
  { id: 'servo',            label: 'Servo',            servoOnly: true,  colorHex: 0x182026, opts: {} },
  { id: 'bracket',          label: 'Bracket',          servoOnly: true,  colorHex: 0xCED9E0, opts: {} },
  { id: 'cradle',           label: 'Cradle',           servoOnly: true,  colorHex: 0xCED9E0, opts: {} },
  { id: 'coupler',          label: 'Coupler',          servoOnly: true,  colorHex: 0xF55656, opts: {} },
  { id: 'bracket_envelope', label: 'Bracket Envelope', servoOnly: true,  colorHex: 0xF55656, opts: { transparent: true, opacity: 0.25 } },
  { id: 'cradle_envelope',  label: 'Cradle Envelope',  servoOnly: true,  colorHex: 0xF55656, opts: { transparent: true, opacity: 0.25 } },
  { id: 'fasteners',        label: 'Fasteners',        servoOnly: false, colorHex: 0xD4A843, opts: {} },
];

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
  }

  async init() {
    this._setupThreeJS();
    await this._fetchCatalog();
    this._buildSidebar();
    this._buildSidePanel();
    this._animate();
  }

  _setupThreeJS() {
    const container = document.getElementById('canvas-container');
    container.style.left = SIDEBAR_WIDTH + 'px';
    container.style.right = SIDE_PANEL_WIDTH + 'px';

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xF5F8FA);

    const viewWidth = window.innerWidth - SIDEBAR_WIDTH - SIDE_PANEL_WIDTH;
    this.camera = new THREE.PerspectiveCamera(45, viewWidth / window.innerHeight, 0.0001, 10);
    this.camera.position.set(0.06, 0.06, 0.08);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(viewWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.update();

    // Lighting — match bot viewer (bright, with fill light)
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.6);
    dirLight.position.set(0.2, 0.4, 0.3);
    dirLight.castShadow = true;
    this.scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0xccccff, 0.5);
    fillLight.position.set(-0.2, 0.1, -0.2);
    this.scene.add(fillLight);

    // Grid (200mm with 5mm divisions) — light theme colors
    this.scene.add(new THREE.GridHelper(0.2, 40, 0xBFCCD6, 0xCED9E0));

    // Create a group per layer
    for (const layer of LAYERS) {
      const g = new THREE.Group();
      g.name = layer.id;
      this.scene.add(g);
      this.layerGroups[layer.id] = g;
    }

    window.addEventListener('resize', () => {
      const vw = window.innerWidth - SIDEBAR_WIDTH - SIDE_PANEL_WIDTH;
      this.camera.aspect = vw / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(vw, window.innerHeight);
    });
  }

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

  _buildSidePanel() {
    const panel = document.getElementById('side-panel');
    panel.innerHTML = '<p style="color:var(--bp-gray3); font-size:13px">Select a component</p>';
  }

  _buildLayerControls(comp) {
    let html = '<h3>Layers</h3>';
    html += '<div class="layer-controls">';

    for (const layer of LAYERS) {
      if (layer.servoOnly && !comp.is_servo) continue;

      const checked = this.layerGroups[layer.id].visible ? 'checked' : '';
      const colorSwatch = layer.colorHex !== null
        ? `<span class="layer-swatch" style="background:#${layer.colorHex.toString(16).padStart(6, '0')}"></span>`
        : `<span class="layer-swatch" style="background:rgb(${Math.round(comp.color[0]*255)},${Math.round(comp.color[1]*255)},${Math.round(comp.color[2]*255)})"></span>`;

      html += `<label class="layer-toggle">
        <input type="checkbox" data-layer="${layer.id}" ${checked}>
        ${colorSwatch}
        <span class="layer-label">${layer.label}</span>
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

  _createMarkers(comp) {
    // Remove old markers
    if (this._markerGroup) {
      this._clearGroup(this._markerGroup);
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

    // Mounting point markers — small cylinders oriented along mount axis
    for (const mp of comp.mounting_points) {
      const r = (mp.diameter_mm / 1000) * 0.8; // slightly larger than hole
      const h = 0.002;
      const geom = new THREE.CylinderGeometry(r, r, h, 12);
      // CylinderGeometry is Y-up; rotate to align with mount axis
      const mesh = new THREE.Mesh(geom, markerMat.clone());
      mesh.position.set(mp.pos[0], mp.pos[1], mp.pos[2]);

      // Orient cylinder axis to mount axis
      const axis = new THREE.Vector3(mp.axis[0], mp.axis[1], mp.axis[2]).normalize();
      const up = new THREE.Vector3(0, 1, 0);
      const quat = new THREE.Quaternion().setFromUnitVectors(up, axis);
      mesh.quaternion.copy(quat);

      mesh.userData.defaultMat = mesh.material;
      mesh.userData.highlightMat = highlightMat;
      this._markerGroup.add(mesh);
      this._markers.mp.push(mesh);
    }

    // Wire port markers — small spheres
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
    // Scale up when highlighted
    const s = on ? 1.8 : 1.0;
    mesh.scale.set(s, s, s);
  }

  async _onLayerToggle(layerId, enabled) {
    const group = this.layerGroups[layerId];

    if (enabled) {
      // Load if empty
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

    // Update top bar — hide mode tabs, show component name
    document.getElementById('bot-name').textContent = name;
    document.getElementById('mode-tabs').style.display = 'none';

    // Clear all layers
    for (const layer of LAYERS) {
      this._clearGroup(this.layerGroups[layer.id]);
      this.layerGroups[layer.id].visible = false;
    }

    // Body enabled by default (set visible after load below)

    // Update side panel (includes layer checkboxes)
    this._updateSidePanel(comp);

    // Load body
    await this._loadLayer('body');
    this.layerGroups['body'].visible = true;
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

    const layerDef = LAYERS.find(l => l.id === layerId);
    const color = layerDef.colorHex !== null
      ? layerDef.colorHex
      : new THREE.Color(comp.color[0], comp.color[1], comp.color[2]);

    await this._addSTLMesh(compName, layerId, color, layerDef.opts, group);
  }

  async _loadFasteners(compName, group) {
    try {
      const resp = await fetch(`/api/components/${compName}/fasteners`);
      if (!resp.ok) return;
      const data = await resp.json();

      // Fetch unique screw STLs in parallel
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

      // Shared material for all fasteners
      const fastenerMat = new THREE.MeshPhysicalMaterial({
        color: 0xD4A843, roughness: 0.4, metalness: 0.3,
      });

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

    const matOpts = {
      color: color,
      roughness: 0.6,
      metalness: 0.1,
    };

    if (opts.transparent) {
      matOpts.transparent = true;
      matOpts.opacity = opts.opacity || 0.3;
      matOpts.side = THREE.DoubleSide;
    }
    if (opts.wireframe) {
      matOpts.wireframe = true;
    }

    const material = new THREE.MeshPhysicalMaterial(matOpts);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = !opts.wireframe;
    mesh.receiveShadow = !opts.wireframe;
    group.add(mesh);

    // Edge outlines (matching bot viewer style) — skip for wireframe/transparent
    if (!opts.wireframe && !opts.transparent) {
      const edges = new THREE.EdgesGeometry(geometry, 28);
      const lines = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
        color: 0x000000, transparent: true, opacity: 0.6,
      }));
      lines.raycast = () => {};
      group.add(lines);
    }
  }

  _clearGroup(group) {
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }
  }

  _fitCameraToVisibleMeshes() {
    const box = new THREE.Box3();
    let hasGeometry = false;

    for (const layer of LAYERS) {
      const group = this.layerGroups[layer.id];
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

    if (!hasGeometry) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this.camera.fov * (Math.PI / 180);
    const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

    this.controls.target.copy(center);
    this.camera.position.set(
      center.x + dist * 0.6,
      center.y + dist * 0.5,
      center.z + dist * 0.7,
    );
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _animate() {
    const loop = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
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
