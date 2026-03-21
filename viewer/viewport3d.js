/**
 * Viewport3D — reusable self-contained 3D viewport with overlay controls.
 *
 * Scene, camera, renderer, OrbitControls, standard lighting, post-processing
 * edge detection, view presets (keyboard 1-7 + clickable buttons), measure
 * tool, and section plane — all embedded in a single container element.
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

const BTN_CSS = `pointer-events:auto;cursor:pointer;border:none;background:rgba(255,255,255,0.85);color:#394B59;font:600 12px system-ui,-apple-system,sans-serif;width:28px;height:28px;border-radius:4px;display:flex;align-items:center;justify-content:center;box-shadow:0 1px 3px rgba(0,0,0,0.12);transition:background 0.12s;`;

export class Viewport3D {
  /** @param {HTMLElement} container  @param {Object} [options] */
  constructor(container, options = {}) {
    this._container = container;
    this._groups = {};
    this._animCb = null;
    this._animating = false;
    this._disposed = false;
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
    const pos = c.clone().addScaledVector(p.dir, d);
    if (this._animating) this._startLerp(pos, c, p.up.clone());
    else { this._cam.position.copy(pos); this._cam.up.copy(p.up); this._ctrl.target.copy(c); this._cam.lookAt(c); this._ctrl.update(); }
    this._hilitePreset(name);
  }

  enableMeasureTool()  { this._meas.enable();  this._ctrl.enabled = false; this._measBtn.classList.add('vp-active'); }
  disableMeasureTool() { this._meas.disable(); this._ctrl.enabled = true;  this._measBtn.classList.remove('vp-active'); }

  enableSectionPlane(axis = 'z', frac = 0.5) {
    this._secOn = true; this._secAxis = axis; this._secFrac = frac;
    this._secBtn.classList.add('vp-active'); this._secPanel.style.display = '';
    this._hiliteAxis(); this._secSlider.value = String(Math.round(frac * 100));
    this._applySection();
  }
  disableSectionPlane() {
    this._secOn = false; this._secBtn.classList.remove('vp-active'); this._secPanel.style.display = 'none';
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
    if (this._overlay) this._overlay.remove();
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
    this._ctrl.enableDamping = true; this._ctrl.dampingFactor = 0.1; this._ctrl.update();
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

    // View preset buttons (bottom-left)
    const bar = document.createElement('div');
    bar.style.cssText = 'position:absolute;bottom:8px;left:8px;display:flex;gap:3px;pointer-events:auto;';
    this._presetBtns = {};
    for (const [name, pr] of Object.entries(VIEW_PRESETS)) {
      const b = document.createElement('button'); b.style.cssText = BTN_CSS;
      b.textContent = pr.key; b.title = `${pr.label} (${pr.key})`;
      b.addEventListener('click', () => this.setViewPreset(name));
      bar.appendChild(b); this._presetBtns[name] = b;
    }
    ov.appendChild(bar);

    // Tool buttons (top-right)
    const tools = document.createElement('div');
    tools.style.cssText = 'position:absolute;top:8px;right:8px;display:flex;gap:3px;pointer-events:auto;';
    this._measBtn = this._mkBtn('\u{1F4CF}', 'Measure (M)', () => {
      if (this._meas.enabled) this.disableMeasureTool(); else this.enableMeasureTool();
    });
    tools.appendChild(this._measBtn);
    this._secBtn = this._mkBtn('\u2702', 'Section plane (S)', () => {
      if (this._secOn) this.disableSectionPlane(); else this.enableSectionPlane();
    });
    tools.appendChild(this._secBtn);
    ov.appendChild(tools);

    // Section panel
    this._secPanel = document.createElement('div');
    this._secPanel.style.cssText = 'position:absolute;top:44px;right:8px;pointer-events:auto;background:rgba(255,255,255,0.92);border-radius:6px;padding:8px;display:none;box-shadow:0 2px 8px rgba(0,0,0,0.12);font:12px system-ui,-apple-system,sans-serif;color:#394B59;';
    const axR = document.createElement('div'); axR.style.cssText = 'display:flex;gap:3px;margin-bottom:6px;';
    this._secAxisBtns = {};
    for (const ax of ['x', 'y', 'z']) {
      const b = document.createElement('button'); b.style.cssText = BTN_CSS + 'width:32px;font-size:11px;';
      b.textContent = ax.toUpperCase();
      b.addEventListener('click', () => { this._secAxis = ax; this._hiliteAxis(); this._applySection(); });
      axR.appendChild(b); this._secAxisBtns[ax] = b;
    }
    this._secPanel.appendChild(axR);
    this._secSlider = document.createElement('input');
    Object.assign(this._secSlider, { type: 'range', min: '0', max: '100', value: '50' });
    this._secSlider.style.cssText = 'width:100%;margin:0;cursor:pointer;';
    this._secSlider.addEventListener('input', () => { this._secFrac = parseFloat(this._secSlider.value) / 100; this._applySection(); });
    this._secPanel.appendChild(this._secSlider);
    ov.appendChild(this._secPanel);
  }

  _mkBtn(text, title, onClick) {
    const b = document.createElement('button'); b.style.cssText = BTN_CSS;
    b.textContent = text; b.title = title; b.addEventListener('click', onClick); return b;
  }

  _initKeys() {
    this._onKey = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (e.ctrlKey || e.metaKey) return;
      if (KEY_TO_PRESET[e.key]) { e.preventDefault(); this.setViewPreset(KEY_TO_PRESET[e.key]); }
      else if (e.key === 'm') { e.preventDefault(); this._measBtn.click(); }
      else if (e.key === 's') { e.preventDefault(); this._secBtn.click(); }
    };
    window.addEventListener('keydown', this._onKey);
  }

  _hilitePreset(active) {
    for (const [n, b] of Object.entries(this._presetBtns)) {
      b.style.color = n === active ? '#137CBD' : '#394B59';
      b.style.background = n === active ? 'rgba(43,149,214,0.15)' : 'rgba(255,255,255,0.85)';
    }
  }
  _hiliteAxis() {
    for (const [a, b] of Object.entries(this._secAxisBtns)) {
      b.style.color = a === this._secAxis ? '#137CBD' : '#394B59';
      b.style.background = a === this._secAxis ? 'rgba(43,149,214,0.15)' : 'rgba(255,255,255,0.85)';
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
    if (this._animCb) this._animCb();
    if (this._edgeC) this._edgeC.render(); else this._ren.render(this._scene, this._cam);
    if (this._meas.enabled || this._meas.measurements.length > 0) this._meas.update();
  }
}
