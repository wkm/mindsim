/**
 * Viewport3D — reusable self-contained 3D viewport with overlay controls.
 *
 * Scene, camera, renderer, OrbitControls, standard lighting, post-processing
 * edge detection, view presets (keyboard 1-7), measure tool, and section plane.
 *
 * Overlay: Shapr3D-style orientation cube (top-right) + vertical tool strip
 * (left edge). All overlay elements created programmatically — no external CSS.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createEdgeComposer } from './presentation.js';
import { MeasureTool } from './measure-tool.js';

const VIEW_PRESETS = {
  iso:    { dir: new THREE.Vector3(1, -1, 0.8).normalize(), up: new THREE.Vector3(0, 0, 1), label: 'Iso',    key: '1' },
  front:  { dir: new THREE.Vector3(0, -1, 0),               up: new THREE.Vector3(0, 0, 1), label: 'Front',  key: '2' },
  top:    { dir: new THREE.Vector3(0, 0, 1),                 up: new THREE.Vector3(0, 1, 0), label: 'Top',    key: '3' },
  right:  { dir: new THREE.Vector3(1, 0, 0),                 up: new THREE.Vector3(0, 0, 1), label: 'Right',  key: '4' },
  back:   { dir: new THREE.Vector3(0, 1, 0),                 up: new THREE.Vector3(0, 0, 1), label: 'Back',   key: '5' },
  bottom: { dir: new THREE.Vector3(0, 0, -1),                up: new THREE.Vector3(0, -1, 0), label: 'Bottom', key: '6' },
  left:   { dir: new THREE.Vector3(-1, 0, 0),                up: new THREE.Vector3(0, 0, 1), label: 'Left',   key: '7' },
};
const KEY_TO_PRESET = {};
for (const [name, p] of Object.entries(VIEW_PRESETS)) KEY_TO_PRESET[p.key] = name;
const AXIS_IDX = { x: 0, y: 1, z: 2 };

// ── SVG icons for tool strip ──
const ICONS = {
  select: `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2">
    <path d="M5 3l14 9-6 2-4 6z"/>
  </svg>`,
  measure: `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2">
    <path d="M21 6H3M21 18H3M12 6v12M7 6v4M17 6v4M7 18v-4M17 18v-4"/>
  </svg>`,
  section: `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <line x1="3" y1="12" x2="21" y2="12"/>
  </svg>`,
};

// ── Cube face mapping: face index → preset name ──
// THREE.BoxGeometry face order: +X, -X, +Y, -Y, +Z, -Z
// We map them to view presets based on the outward normal direction.
const CUBE_FACE_MAP = [
  { preset: 'right',  label: 'RIGHT',  key: '4' },  // +X
  { preset: 'left',   label: 'LEFT',   key: '7' },  // -X
  { preset: 'back',   label: 'BACK',   key: '5' },  // +Y
  { preset: 'front',  label: 'FRONT',  key: '2' },  // -Y
  { preset: 'top',    label: 'TOP',    key: '3' },  // +Z
  { preset: 'bottom', label: 'BOTTOM', key: '6' },  // -Z
];

// Blueprint palette grays for cube faces
const CUBE_FACE_COLORS = [
  '#D8DEE4', // +X right  — medium
  '#D8DEE4', // -X left   — medium
  '#D8DEE4', // +Y back   — medium
  '#D8DEE4', // -Y front  — medium
  '#E8EDF0', // +Z top    — lighter
  '#C5CDD4', // -Z bottom — darker
];

export class Viewport3D {
  /** @param {HTMLElement} container  @param {Object} [options] */
  constructor(container, options = {}) {
    this._container = container;
    this._groups = {};
    this._animCb = null;
    this._animating = false;
    this._disposed = false;
    this._activeTool = null; // 'select' | 'measure' | 'section' | null
    // Lerp
    this._lerpActive = false;
    this._lerpS = { pos: new THREE.Vector3(), tgt: new THREE.Vector3(), up: new THREE.Vector3() };
    this._lerpE = { pos: new THREE.Vector3(), tgt: new THREE.Vector3(), up: new THREE.Vector3() };
    this._lerpDur = 300;
    // Section
    this._secOn = false;
    this._secAxis = 'z';
    this._secFrac = 0.5;
    this._secPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this._secViz = null;
    // Follow mode
    this._followMode = false;
    this._followTarget = null;  // Vector3 — center of followed geometry
    this._followRadius = 0;     // bounding sphere radius for exit threshold
    this._followBadge = null;   // DOM element for "Following" indicator
    this._onFollowChange = null; // callback when follow mode changes

    this._initScene(options);
    this._initOverlay();
    this._initKeys();
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
  }

  get scene()    { return this._scene; }
  get camera()   { return this._cam; }
  get controls() { return this._ctrl; }
  get renderer() { return this._ren; }

  addGroup(name) {
    const g = new THREE.Group(); g.name = name;
    this._scene.add(g); this._groups[name] = g; return g;
  }
  getGroup(name) { return this._groups[name] || null; }

  frameOnGeometry(geometry) {
    geometry.computeBoundingBox();
    this.frameOnBox(geometry.boundingBox);
  }
  frameOnBox(box3, animate = true) {
    const center = new THREE.Vector3(), size = new THREE.Vector3();
    box3.getCenter(center); box3.getSize(size);
    const d = Math.max(size.x, size.y, size.z) * 2.5;
    const pos = new THREE.Vector3(center.x + d * 0.6, center.y - d * 0.6, center.z + d * 0.8);
    const up = new THREE.Vector3(0, 0, 1);
    if (animate && this._animating) { this._startLerp(pos, center, up); }
    else { this._cam.position.copy(pos); this._cam.up.copy(up); this._ctrl.target.copy(center); this._ctrl.update(); }
  }
  setViewPreset(name) {
    const p = VIEW_PRESETS[name]; if (!p) return;
    const box = this._bbox();
    const c = box ? box.getCenter(new THREE.Vector3()) : new THREE.Vector3();
    const sz = box ? box.getSize(new THREE.Vector3()) : new THREE.Vector3(0.1, 0.1, 0.1);
    const d = Math.max(sz.x, sz.y, sz.z) * 2.5;
    // In follow mode, maintain focus on the followed target instead of bbox center
    const target = (this._followMode && this._followTarget) ? this._followTarget.clone() : c;
    const pos = target.clone().addScaledVector(p.dir, d);
    if (this._animating) this._startLerp(pos, target, p.up.clone());
    else { this._cam.position.copy(pos); this._cam.up.copy(p.up); this._ctrl.target.copy(target); this._cam.lookAt(target); this._ctrl.update(); }
  }

  // ── Follow mode ──

  setFollowMode(enabled) {
    this._followMode = enabled;
    if (!enabled) {
      this._followTarget = null;
      this._followRadius = 0;
    }
    this._updateFollowBadge();
    if (this._onFollowChange) this._onFollowChange(enabled);
  }

  isFollowMode() { return this._followMode; }

  onFollowChange(cb) { this._onFollowChange = cb; }

  /**
   * Called by the step debugger when the step changes while in follow mode.
   * Smoothly frames on the given geometry and updates the follow target.
   */
  frameIfFollowing(geometry) {
    if (!this._followMode || !geometry) return;
    this.frameOnGeometry(geometry);
    // Update follow target/radius for drift detection
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    this._followTarget = new THREE.Vector3();
    box.getCenter(this._followTarget);
    const size = new THREE.Vector3();
    box.getSize(size);
    this._followRadius = Math.max(size.x, size.y, size.z) * 0.5;
  }

  _updateFollowBadge() {
    if (this._followMode) {
      if (!this._followBadge) {
        this._followBadge = document.createElement('div');
        this._followBadge.style.cssText = 'position:absolute;top:8px;left:50%;transform:translateX(-50%);background:rgba(145,121,242,0.8);color:white;font:500 11px system-ui,-apple-system,sans-serif;padding:2px 10px;border-radius:10px;pointer-events:none;z-index:60;transition:opacity 0.2s;';
        this._followBadge.textContent = 'Following';
        this._overlay.appendChild(this._followBadge);
      }
      this._followBadge.style.opacity = '1';
      this._followBadge.style.display = '';
    } else if (this._followBadge) {
      this._followBadge.style.opacity = '0';
      setTimeout(() => { if (this._followBadge && !this._followMode) this._followBadge.style.display = 'none'; }, 200);
    }
  }

  enableMeasureTool()  {
    this._meas.enable(); this._ctrl.enabled = false;
    this._setActiveTool('measure');
  }
  disableMeasureTool() {
    this._meas.disable(); this._ctrl.enabled = true;
    if (this._activeTool === 'measure') this._setActiveTool(null);
  }

  enableSectionPlane(axis = 'z', frac = 0.5) {
    this._secOn = true; this._secAxis = axis; this._secFrac = frac;
    this._setActiveTool('section');
    this._secPopover.style.display = '';
    this._hiliteAxis(); this._secSlider.value = String(Math.round(frac * 100));
    this._applySection();
  }
  disableSectionPlane() {
    this._secOn = false;
    if (this._activeTool === 'section') this._setActiveTool(null);
    this._secPopover.style.display = 'none';
    if (this._secViz) this._secViz.visible = false;
    this._scene.traverse(ch => { if (ch.material) ch.material.clippingPlanes = []; });
  }

  animate(cb) { this._animCb = cb || null; if (!this._animating) { this._animating = true; this._tick(); } }
  resize() {
    const w = this._container.clientWidth, h = this._container.clientHeight;
    if (!w || !h) return;
    this._cam.aspect = w / h; this._cam.updateProjectionMatrix();
    this._ren.setSize(w, h); if (this._edgeC) this._edgeC.resize(w, h);
  }
  dispose() {
    this._disposed = true;
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('keydown', this._onKey);
    this._meas.disable(); this._ctrl.dispose(); this._ren.dispose();
    if (this._cubeRen) this._cubeRen.dispose();
    if (this._overlay) this._overlay.remove();
    if (this._cubeCanvas) this._cubeCanvas.remove();
  }

  // ── Scene init ──
  _initScene(opts) {
    const c = this._container, w = c.clientWidth || 800, h = c.clientHeight || 600;
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0xF5F8FA);
    this._cam = new THREE.PerspectiveCamera(45, w / h, 0.0001, 10);
    this._cam.position.set(0.08, -0.08, 0.1); this._cam.up.set(0, 0, 1);
    this._ren = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true, stencil: true });
    this._ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._ren.setSize(w, h); this._ren.shadowMap.enabled = true;
    this._ren.shadowMap.type = THREE.PCFShadowMap; this._ren.localClippingEnabled = true;
    c.appendChild(this._ren.domElement);
    this._ctrl = new OrbitControls(this._cam, this._ren.domElement);
    this._ctrl.enableDamping = true; this._ctrl.dampingFactor = 0.1;
    // Trackpad: two-finger drag = rotate (not pan), pinch = zoom
    this._ctrl.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_ROTATE };
    this._ctrl.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this._ctrl.update();
    // Lighting
    this._scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6); dir.position.set(0.3, 0.5, 0.4); this._scene.add(dir);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-0.3, -0.2, -0.4); this._scene.add(fill);
    if (opts.grid) { const g = new THREE.GridHelper(0.3, 30, 0xCED9E0, 0xE8EDF0); g.rotation.x = Math.PI / 2; this._scene.add(g); }
    if (opts.edges) this._edgeC = createEdgeComposer(this._ren, this._scene, this._cam);
    this._meas = new MeasureTool(this._cam, this._scene, c);
  }

  // ── Overlay controls ──
  _initOverlay() {
    const ov = document.createElement('div');
    ov.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:50;';
    this._container.appendChild(ov); this._overlay = ov;

    this._initOrientationCube();
    this._initToolStrip();
  }

  // ── Orientation Cube (top-right) ──
  _initOrientationCube() {
    const SIZE = 100;
    const PIXEL_RATIO = Math.min(window.devicePixelRatio, 2);

    // Canvas element
    const canvas = document.createElement('canvas');
    canvas.width = SIZE * PIXEL_RATIO;
    canvas.height = SIZE * PIXEL_RATIO;
    canvas.style.cssText = `position:absolute;top:8px;right:8px;width:${SIZE}px;height:${SIZE}px;pointer-events:auto;cursor:pointer;border-radius:6px;`;
    this._overlay.appendChild(canvas);
    this._cubeCanvas = canvas;

    // Separate renderer
    this._cubeRen = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this._cubeRen.setPixelRatio(PIXEL_RATIO);
    this._cubeRen.setSize(SIZE, SIZE);
    this._cubeRen.setClearColor(0x000000, 0);

    // Scene + camera
    this._cubeScene = new THREE.Scene();
    this._cubeCam = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    this._cubeCam.position.set(0, 0, 4);

    // Lighting for cube
    this._cubeScene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const cubeDir = new THREE.DirectionalLight(0xffffff, 0.6);
    cubeDir.position.set(2, 3, 4);
    this._cubeScene.add(cubeDir);

    // Create face materials with canvas textures
    const materials = CUBE_FACE_MAP.map((face, i) => {
      const texCanvas = document.createElement('canvas');
      texCanvas.width = 128; texCanvas.height = 128;
      const ctx = texCanvas.getContext('2d');
      ctx.fillStyle = CUBE_FACE_COLORS[i];
      ctx.fillRect(0, 0, 128, 128);
      ctx.fillStyle = '#5C7080';
      ctx.font = 'bold 22px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(face.label, 64, 64);
      const tex = new THREE.CanvasTexture(texCanvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      return new THREE.MeshStandardMaterial({ map: tex, roughness: 0.7, metalness: 0 });
    });

    // Store base textures for hover highlighting
    this._cubeFaceTextures = materials.map((_, i) => ({
      baseColor: CUBE_FACE_COLORS[i],
      label: CUBE_FACE_MAP[i].label,
    }));

    const geo = new THREE.BoxGeometry(1, 1, 1);
    this._cubeMesh = new THREE.Mesh(geo, materials);
    this._cubeScene.add(this._cubeMesh);

    // Edge wireframe
    const edges = new THREE.EdgesGeometry(geo);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x394B59, linewidth: 1 });
    const wireframe = new THREE.LineSegments(edges, lineMat);
    this._cubeMesh.add(wireframe);

    // Raycaster for click/hover detection
    this._cubeRaycaster = new THREE.Raycaster();
    this._cubeHoveredFace = -1;

    // Tooltip
    this._cubeTooltip = document.createElement('div');
    this._cubeTooltip.style.cssText = 'position:absolute;pointer-events:none;background:rgba(28,33,39,0.9);color:#E8EDF0;font:500 12px system-ui,-apple-system,sans-serif;padding:4px 8px;border-radius:4px;white-space:nowrap;display:none;z-index:60;';
    this._overlay.appendChild(this._cubeTooltip);

    // Events
    canvas.addEventListener('mousemove', (e) => this._onCubeHover(e));
    canvas.addEventListener('mouseleave', () => this._onCubeLeave());
    canvas.addEventListener('click', (e) => this._onCubeClick(e));
  }

  _getCubeFaceAtMouse(e) {
    const rect = this._cubeCanvas.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    this._cubeRaycaster.setFromCamera(mouse, this._cubeCam);
    const hits = this._cubeRaycaster.intersectObject(this._cubeMesh);
    if (hits.length === 0) return -1;
    // faceIndex is the triangle index; each cube face has 2 triangles
    return Math.floor(hits[0].faceIndex / 2);
  }

  _onCubeHover(e) {
    const fi = this._getCubeFaceAtMouse(e);
    if (fi === this._cubeHoveredFace) return;

    // Restore previous face
    if (this._cubeHoveredFace >= 0) this._setCubeFaceHighlight(this._cubeHoveredFace, false);
    this._cubeHoveredFace = fi;
    if (fi >= 0) {
      this._setCubeFaceHighlight(fi, true);
      const face = CUBE_FACE_MAP[fi];
      this._cubeTooltip.textContent = `${face.label.charAt(0) + face.label.slice(1).toLowerCase()} view (${face.key})`;
      // Position tooltip to the left of the cube
      const rect = this._cubeCanvas.getBoundingClientRect();
      const containerRect = this._container.getBoundingClientRect();
      this._cubeTooltip.style.display = '';
      this._cubeTooltip.style.top = `${e.clientY - containerRect.top - 12}px`;
      this._cubeTooltip.style.right = `${containerRect.right - rect.left + 6}px`;
    } else {
      this._cubeTooltip.style.display = 'none';
    }
    this._cubeCanvas.style.cursor = fi >= 0 ? 'pointer' : 'default';
  }

  _onCubeLeave() {
    if (this._cubeHoveredFace >= 0) this._setCubeFaceHighlight(this._cubeHoveredFace, false);
    this._cubeHoveredFace = -1;
    this._cubeTooltip.style.display = 'none';
  }

  _onCubeClick(e) {
    const fi = this._getCubeFaceAtMouse(e);
    if (fi < 0) return;
    this.setViewPreset(CUBE_FACE_MAP[fi].preset);
  }

  _setCubeFaceHighlight(faceIdx, highlight) {
    const info = this._cubeFaceTextures[faceIdx];
    const mat = this._cubeMesh.material[faceIdx];
    const texCanvas = mat.map.image;
    const ctx = texCanvas.getContext('2d');
    ctx.fillStyle = highlight ? '#BCC7CF' : info.baseColor;
    ctx.fillRect(0, 0, 128, 128);
    ctx.fillStyle = highlight ? '#182026' : '#5C7080';
    ctx.font = 'bold 22px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(info.label, 64, 64);
    mat.map.needsUpdate = true;
  }

  _syncOrientationCube() {
    // Mirror main camera orientation onto cube camera.
    // The cube camera looks at the origin from a fixed distance;
    // we rotate the *cube mesh* to match the inverse of the camera direction.
    const dir = new THREE.Vector3();
    this._cam.getWorldDirection(dir);
    const dist = 4;
    this._cubeCam.position.copy(dir.multiplyScalar(-dist));
    this._cubeCam.up.copy(this._cam.up);
    this._cubeCam.lookAt(0, 0, 0);
    this._cubeRen.render(this._cubeScene, this._cubeCam);
  }

  // ── Vertical Tool Strip (left edge) ──
  _initToolStrip() {
    const strip = document.createElement('div');
    strip.style.cssText = 'position:absolute;left:8px;top:50%;transform:translateY(-50%);pointer-events:auto;background:rgba(28,33,39,0.85);border-radius:8px;padding:4px;display:flex;flex-direction:column;gap:2px;z-index:51;';
    this._overlay.appendChild(strip);
    this._toolStrip = strip;

    const toolBtnCSS = 'width:36px;height:36px;border:none;border-radius:6px;background:transparent;color:#CED9E0;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.12s,color 0.12s;position:relative;';

    // Select tool
    this._selectBtn = document.createElement('button');
    this._selectBtn.style.cssText = toolBtnCSS;
    this._selectBtn.innerHTML = ICONS.select;
    this._selectBtn.addEventListener('click', () => {
      if (this._activeTool === 'measure') this.disableMeasureTool();
      if (this._activeTool === 'section') this.disableSectionPlane();
      this._setActiveTool(null);
    });
    strip.appendChild(this._selectBtn);
    this._addToolTooltip(this._selectBtn, 'Select (Esc)');

    // Divider
    const divider = document.createElement('div');
    divider.style.cssText = 'height:1px;margin:2px 4px;background:rgba(206,217,224,0.2);';
    strip.appendChild(divider);

    // Measure tool
    this._measBtn = document.createElement('button');
    this._measBtn.style.cssText = toolBtnCSS;
    this._measBtn.innerHTML = ICONS.measure;
    this._measBtn.addEventListener('click', () => {
      if (this._meas.enabled) this.disableMeasureTool();
      else {
        if (this._secOn) this.disableSectionPlane();
        this.enableMeasureTool();
      }
    });
    strip.appendChild(this._measBtn);
    this._addToolTooltip(this._measBtn, 'Measure (M)');

    // Section tool
    this._secBtn = document.createElement('button');
    this._secBtn.style.cssText = toolBtnCSS;
    this._secBtn.innerHTML = ICONS.section;
    this._secBtn.addEventListener('click', () => {
      if (this._secOn) this.disableSectionPlane();
      else {
        if (this._meas.enabled) this.disableMeasureTool();
        this.enableSectionPlane();
      }
    });
    strip.appendChild(this._secBtn);
    this._addToolTooltip(this._secBtn, 'Section (S)');

    // Section popover (appears to the right of tool strip)
    this._secPopover = document.createElement('div');
    this._secPopover.style.cssText = 'position:absolute;left:52px;top:50%;transform:translateY(-50%);pointer-events:auto;background:rgba(28,33,39,0.92);border-radius:8px;padding:10px;display:none;box-shadow:0 4px 12px rgba(0,0,0,0.25);font:12px system-ui,-apple-system,sans-serif;color:#CED9E0;min-width:120px;z-index:52;';

    // Axis selector
    const axLabel = document.createElement('div');
    axLabel.style.cssText = 'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:#8A9BA8;margin-bottom:6px;';
    axLabel.textContent = 'Axis';
    this._secPopover.appendChild(axLabel);

    const axRow = document.createElement('div');
    axRow.style.cssText = 'display:flex;gap:3px;margin-bottom:8px;';
    this._secAxisBtns = {};
    const axisBtnCSS = 'width:32px;height:26px;border:1px solid rgba(206,217,224,0.2);border-radius:4px;background:transparent;color:#CED9E0;font:600 11px system-ui,-apple-system,sans-serif;cursor:pointer;transition:all 0.12s;';
    for (const ax of ['x', 'y', 'z']) {
      const b = document.createElement('button');
      b.style.cssText = axisBtnCSS;
      b.textContent = ax.toUpperCase();
      b.addEventListener('click', () => { this._secAxis = ax; this._hiliteAxis(); this._applySection(); });
      axRow.appendChild(b); this._secAxisBtns[ax] = b;
    }
    this._secPopover.appendChild(axRow);

    // Slider
    const sliderLabel = document.createElement('div');
    sliderLabel.style.cssText = 'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:#8A9BA8;margin-bottom:4px;';
    sliderLabel.textContent = 'Position';
    this._secPopover.appendChild(sliderLabel);

    this._secSlider = document.createElement('input');
    Object.assign(this._secSlider, { type: 'range', min: '0', max: '100', value: '50' });
    this._secSlider.style.cssText = 'width:100%;margin:0;cursor:pointer;accent-color:#137CBD;';
    this._secSlider.addEventListener('input', () => { this._secFrac = parseFloat(this._secSlider.value) / 100; this._applySection(); });
    this._secPopover.appendChild(this._secSlider);

    // Attach popover next to strip
    strip.style.position = 'absolute'; // ensure positioning context
    this._overlay.appendChild(this._secPopover);

    // Keep popover positioned next to the strip
    this._positionSecPopover();
  }

  _positionSecPopover() {
    // We anchor the popover relative to the tool strip
    const strip = this._toolStrip;
    const update = () => {
      if (this._disposed) return;
      const sr = strip.getBoundingClientRect();
      const cr = this._container.getBoundingClientRect();
      this._secPopover.style.left = `${sr.right - cr.left + 6}px`;
      this._secPopover.style.top = `${sr.top - cr.top + sr.height / 2}px`;
      this._secPopover.style.transform = 'translateY(-50%)';
    };
    // Update on every show
    const origDisplay = Object.getOwnPropertyDescriptor(CSSStyleDeclaration.prototype, 'display');
    const panel = this._secPopover;
    this._updateSecPopoverPos = update;
  }

  _addToolTooltip(btn, text) {
    const tip = document.createElement('div');
    tip.style.cssText = 'position:absolute;left:calc(100% + 8px);top:50%;transform:translateY(-50%);pointer-events:none;background:rgba(28,33,39,0.9);color:#E8EDF0;font:500 12px system-ui,-apple-system,sans-serif;padding:4px 8px;border-radius:4px;white-space:nowrap;display:none;z-index:60;';
    tip.textContent = text;
    btn.appendChild(tip);
    btn.addEventListener('mouseenter', () => { tip.style.display = ''; });
    btn.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
  }

  _setActiveTool(tool) {
    this._activeTool = tool;
    const activeCSS = 'background:rgba(19,124,189,0.3);color:#2B95D6;';
    const inactiveCSS = 'background:transparent;color:#CED9E0;';

    // Update button styles (preserve base styles)
    const base = 'width:36px;height:36px;border:none;border-radius:6px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.12s,color 0.12s;position:relative;';
    this._selectBtn.style.cssText = base + (tool === null ? activeCSS : inactiveCSS);
    this._measBtn.style.cssText = base + (tool === 'measure' ? activeCSS : inactiveCSS);
    this._secBtn.style.cssText = base + (tool === 'section' ? activeCSS : inactiveCSS);
  }

  _initKeys() {
    this._onKey = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (e.ctrlKey || e.metaKey) return;
      if (KEY_TO_PRESET[e.key]) { e.preventDefault(); this.setViewPreset(KEY_TO_PRESET[e.key]); }
      else if (e.key === 'f') { e.preventDefault(); this.setFollowMode(!this._followMode); }
      else if (e.key === 'm') { e.preventDefault(); this._measBtn.click(); }
      else if (e.key === 's') { e.preventDefault(); this._secBtn.click(); }
      else if (e.key === 'Escape') {
        e.preventDefault();
        if (this._meas.enabled) this.disableMeasureTool();
        if (this._secOn) this.disableSectionPlane();
        this._setActiveTool(null);
      }
    };
    window.addEventListener('keydown', this._onKey);
  }

  _hiliteAxis() {
    for (const [a, b] of Object.entries(this._secAxisBtns)) {
      b.style.background = a === this._secAxis ? 'rgba(19,124,189,0.3)' : 'transparent';
      b.style.color = a === this._secAxis ? '#2B95D6' : '#CED9E0';
      b.style.borderColor = a === this._secAxis ? 'rgba(43,149,214,0.4)' : 'rgba(206,217,224,0.2)';
    }
  }

  // ── Content bounding box ──
  _bbox() {
    const box = new THREE.Box3(); let has = false;
    for (const g of Object.values(this._groups)) {
      if (!g.visible) continue;
      g.traverse(ch => {
        if (ch.isMesh && ch.geometry) {
          ch.geometry.computeBoundingBox();
          const b = ch.geometry.boundingBox.clone(); b.applyMatrix4(ch.matrixWorld);
          box.union(b); has = true;
        }
      });
    }
    return has ? box : null;
  }

  // ── Camera lerp ──
  _startLerp(pos, tgt, up) {
    this._lerpS.pos.copy(this._cam.position); this._lerpS.tgt.copy(this._ctrl.target); this._lerpS.up.copy(this._cam.up);
    this._lerpE.pos.copy(pos); this._lerpE.tgt.copy(tgt); this._lerpE.up.copy(up);
    this._lerpT0 = performance.now(); this._lerpActive = true;
  }
  _tickLerp() {
    if (!this._lerpActive) return;
    let t = Math.min((performance.now() - this._lerpT0) / this._lerpDur, 1);
    t = 1 - Math.pow(1 - t, 3); // ease-out cubic
    this._cam.position.lerpVectors(this._lerpS.pos, this._lerpE.pos, t);
    this._ctrl.target.lerpVectors(this._lerpS.tgt, this._lerpE.tgt, t);
    this._cam.up.lerpVectors(this._lerpS.up, this._lerpE.up, t).normalize();
    this._cam.lookAt(this._ctrl.target); this._ctrl.update();
    if (t >= 1) this._lerpActive = false;
  }

  // ── Section plane ──
  _applySection() {
    const box = this._bbox();
    if (box) {
      const ai = AXIS_IDX[this._secAxis];
      const pos = box.min.getComponent(ai) + (box.max.getComponent(ai) - box.min.getComponent(ai)) * this._secFrac;
      const n = new THREE.Vector3(); n.setComponent(ai, -1);
      this._secPlane.normal.copy(n); this._secPlane.constant = pos;
      this._showSecViz(box, ai, pos);
    }
    const clips = [this._secPlane];
    this._scene.traverse(ch => {
      if (ch.material && !ch.userData._vpSec) {
        if (ch.isLineSegments && !ch.material._vpClip) { ch.material = ch.material.clone(); ch.material._vpClip = true; }
        ch.material.clippingPlanes = clips; ch.material.clipShadows = true;
      }
    });
  }
  _showSecViz(box, ai, pos) {
    if (!this._secViz) {
      const mat = new THREE.MeshBasicMaterial({ color: 0x2B95D6, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false });
      this._secViz = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
      this._secViz.userData._vpSec = true; this._secViz.raycast = () => {}; this._scene.add(this._secViz);
    }
    const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
    this._secViz.scale.set(Math.max(sz.x, sz.y, sz.z) * 1.5, Math.max(sz.x, sz.y, sz.z) * 1.5, 1);
    this._secViz.visible = true; c.setComponent(ai, pos); this._secViz.position.copy(c);
    this._secViz.rotation.set(0, 0, 0);
    if (ai === 0) this._secViz.rotation.y = Math.PI / 2;
    else if (ai === 1) this._secViz.rotation.x = Math.PI / 2;
  }

  // ── Render loop ──
  _tick() {
    if (this._disposed) return;
    requestAnimationFrame(() => this._tick());
    this._tickLerp(); this._ctrl.update();
    // Follow mode drift detection — exit if user pans/zooms significantly
    if (this._followMode && this._followTarget && !this._lerpActive) {
      const drift = this._ctrl.target.distanceTo(this._followTarget);
      const threshold = this._followRadius > 0 ? this._followRadius * 0.2 : 0.001;
      if (drift > threshold) {
        this.setFollowMode(false);
      }
    }
    if (this._animCb) this._animCb();
    if (this._edgeC) this._edgeC.render(); else this._ren.render(this._scene, this._cam);
    if (this._meas.enabled || this._meas.measurements.length > 0) this._meas.update();
    // Sync orientation cube
    this._syncOrientationCube();
    // Keep section popover positioned
    if (this._secOn && this._updateSecPopoverPos) this._updateSecPopoverPos();
  }
}
