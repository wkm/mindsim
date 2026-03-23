/**
 * Viewport3D — reusable self-contained 3D viewport with overlay controls.
 *
 * Scene, camera, renderer, OrbitControls, standard lighting, post-processing
 * edge detection, view presets (keyboard 1-7), measure tool, and section plane
 * with geometry-based caps and contour lines.
 *
 * Overlay: polished orientation cube (top-right) + vertical tool strip
 * (left edge). All overlay elements created programmatically — no external CSS.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { Earcut } from 'three/src/extras/Earcut.js';
import { MeasureTool } from './measure-tool.ts';
import { BP, createEdgeComposer, RENDER_ORDER } from './presentation.ts';

const VIEW_PRESETS = {
  iso: { dir: new THREE.Vector3(1, -1, 0.8).normalize(), up: new THREE.Vector3(0, 0, 1), label: 'Iso', key: '1' },
  front: { dir: new THREE.Vector3(0, -1, 0), up: new THREE.Vector3(0, 0, 1), label: 'Front', key: '2' },
  top: { dir: new THREE.Vector3(0, 0, 1), up: new THREE.Vector3(0, 1, 0), label: 'Top', key: '3' },
  right: { dir: new THREE.Vector3(1, 0, 0), up: new THREE.Vector3(0, 0, 1), label: 'Right', key: '4' },
  back: { dir: new THREE.Vector3(0, 1, 0), up: new THREE.Vector3(0, 0, 1), label: 'Back', key: '5' },
  bottom: { dir: new THREE.Vector3(0, 0, -1), up: new THREE.Vector3(0, -1, 0), label: 'Bottom', key: '6' },
  left: { dir: new THREE.Vector3(-1, 0, 0), up: new THREE.Vector3(0, 0, 1), label: 'Left', key: '7' },
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
  settings: `<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" fill="none" stroke-width="2">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
  </svg>`,
};

// ── Cube face mapping: face index → preset name ──
// THREE.BoxGeometry face order: +X, -X, +Y, -Y, +Z, -Z
const CUBE_FACE_MAP = [
  { preset: 'right', label: 'Right', key: '4', normal: new THREE.Vector3(1, 0, 0) }, // +X
  { preset: 'left', label: 'Left', key: '7', normal: new THREE.Vector3(-1, 0, 0) }, // -X
  { preset: 'back', label: 'Back', key: '5', normal: new THREE.Vector3(0, 1, 0) }, // +Y
  { preset: 'front', label: 'Front', key: '2', normal: new THREE.Vector3(0, -1, 0) }, // -Y
  { preset: 'top', label: 'Top', key: '3', normal: new THREE.Vector3(0, 0, 1) }, // +Z
  { preset: 'bottom', label: 'Bottom', key: '6', normal: new THREE.Vector3(0, 0, -1) }, // -Z
];

// Blueprint palette grays for cube faces — subtle gradient top-to-bottom
const CUBE_FACE_COLORS = [
  '#D8DEE4', // +X right  — medium
  '#D8DEE4', // -X left   — medium
  '#D8DEE4', // +Y back   — medium
  '#D8DEE4', // -Y front  — medium
  '#E8EDF0', // +Z top    — lightest
  '#C5CDD4', // -Z bottom — darkest
];

// ── Edge/corner click zones for the orientation cube ──
// Edges: average of two adjacent face normals → 45-degree view
const _CUBE_EDGES = [
  // Top edges
  { n: new THREE.Vector3(1, 0, 1).normalize(), label: 'Right-Top' },
  { n: new THREE.Vector3(-1, 0, 1).normalize(), label: 'Left-Top' },
  { n: new THREE.Vector3(0, 1, 1).normalize(), label: 'Back-Top' },
  { n: new THREE.Vector3(0, -1, 1).normalize(), label: 'Front-Top' },
  // Bottom edges
  { n: new THREE.Vector3(1, 0, -1).normalize(), label: 'Right-Bottom' },
  { n: new THREE.Vector3(-1, 0, -1).normalize(), label: 'Left-Bottom' },
  { n: new THREE.Vector3(0, 1, -1).normalize(), label: 'Back-Bottom' },
  { n: new THREE.Vector3(0, -1, -1).normalize(), label: 'Front-Bottom' },
  // Horizontal edges
  { n: new THREE.Vector3(1, -1, 0).normalize(), label: 'Right-Front' },
  { n: new THREE.Vector3(-1, -1, 0).normalize(), label: 'Left-Front' },
  { n: new THREE.Vector3(1, 1, 0).normalize(), label: 'Right-Back' },
  { n: new THREE.Vector3(-1, 1, 0).normalize(), label: 'Left-Back' },
];

// Corners: isometric views with Z-bias matching VIEW_PRESETS.iso (1, -1, 0.8)
// The 0.8 Z factor gives a slightly top-down perspective that matches the `1` key preset.
const _CUBE_CORNERS = [
  { n: new THREE.Vector3(1, -1, 0.8).normalize(), label: 'Iso' },
  { n: new THREE.Vector3(-1, -1, 0.8).normalize(), label: 'Iso' },
  { n: new THREE.Vector3(1, 1, 0.8).normalize(), label: 'Iso' },
  { n: new THREE.Vector3(-1, 1, 0.8).normalize(), label: 'Iso' },
  { n: new THREE.Vector3(1, -1, -0.8).normalize(), label: 'Iso' },
  { n: new THREE.Vector3(-1, -1, -0.8).normalize(), label: 'Iso' },
  { n: new THREE.Vector3(1, 1, -0.8).normalize(), label: 'Iso' },
  { n: new THREE.Vector3(-1, 1, -0.8).normalize(), label: 'Iso' },
];

export class Viewport3D {
  _container: any;
  _groups: any;
  _animCb: any;
  _animating: boolean;
  _disposed: boolean;
  _activeTool: any;
  _cameraType: string;
  _lerpActive: boolean;
  _lerpS: any;
  _lerpE: any;
  _lerpDur: number;
  _lerpT0: number;
  _secOn: boolean;
  _secAxis: string;
  _secFrac: number;
  _secFlipped: boolean;
  _secPlane: any;
  _secViz: any;
  _capGroup: any;
  _contourLineMat: any;
  _ghostedMeshes: Map<any, any>;
  _followMode: boolean;
  _followTarget: any;
  _followRadius: number;
  _followBadge: any;
  _onFollowChange: any;
  _sectionCapColorFn: any;
  _scene: any;
  _cam: any;
  _ctrl: any;
  _ren: any;
  _edgeC: any;
  _meas: any;
  _gridHelper: any;
  _overlay: any;
  _cubeCanvas: any;
  _cubeRen: any;
  _cubeScene: any;
  _cubeCam: any;
  _cubeMesh: any;
  _cubeFaceTextures: any;
  _cubeContainer: any;
  _cubeRaycaster: any;
  _cubeHoveredFace: number;
  _cubeTooltip: any;
  _cubeHoverInfo: any;
  _toolStrip: any;
  _selectBtn: any;
  _measBtn: any;
  _secBtn: any;
  _secPopover: any;
  _secAxisBtns: any;
  _secSlider: any;
  _settingsBtn: any;
  _settingsPopover: any;
  _perspBtn: any;
  _orthoBtn: any;
  _edgeCb: any;
  _gridCb: any;
  _onResize: any;
  _onKey: any;
  _updateSecPopoverPos: any;
  _updateSettingsPopoverPos: any;
  _hatchTextures: any;

  /** @param {HTMLElement} container  @param {Object} [options] */
  constructor(container: any, options: any = {}) {
    this._container = container;
    this._groups = {};
    this._animCb = null;
    this._animating = false;
    this._disposed = false;
    this._activeTool = null; // 'select' | 'measure' | 'section' | null
    this._cameraType = options.cameraType || 'orthographic';
    // Lerp
    this._lerpActive = false;
    this._lerpS = { pos: new THREE.Vector3(), tgt: new THREE.Vector3(), up: new THREE.Vector3() };
    this._lerpE = { pos: new THREE.Vector3(), tgt: new THREE.Vector3(), up: new THREE.Vector3() };
    this._lerpDur = 300;
    // Section
    this._secOn = false;
    this._secAxis = 'z';
    this._secFrac = 0.5;
    this._secFlipped = false;
    this._secPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
    this._secViz = null;
    this._capGroup = null;
    this._contourLineMat = null;
    // Ghosting
    this._ghostedMeshes = new Map(); // mesh → { opacity, transparent }
    // Follow mode
    this._followMode = false;
    this._followTarget = null; // Vector3 — center of followed geometry
    this._followRadius = 0; // bounding sphere radius for exit threshold
    this._followBadge = null; // DOM element for "Following" indicator
    this._onFollowChange = null; // callback when follow mode changes
    // Section cap callback — lets external code provide per-group cap colors
    this._sectionCapColorFn = null;

    this._initScene(options);
    this._initOverlay();
    this._initKeys();
    this._onResize = () => this.resize();
    window.addEventListener('resize', this._onResize);
  }

  get scene() {
    return this._scene;
  }
  get camera() {
    return this._cam;
  }
  get controls() {
    return this._ctrl;
  }
  get renderer() {
    return this._ren;
  }

  addGroup(name) {
    const g = new THREE.Group();
    g.name = name;
    this._scene.add(g);
    this._groups[name] = g;
    return g;
  }
  getGroup(name) {
    return this._groups[name] || null;
  }

  /**
   * Set a callback that returns a cap color for a given group name.
   * Signature: (groupName: string) => THREE.Color | number | null
   * If null is returned, a default darker tint is used.
   */
  setSectionCapColorFn(fn) {
    this._sectionCapColorFn = fn;
  }

  /** Zoom to fit all visible geometry (F key). */
  zoomToFit() {
    const box = this._bbox();
    if (box) this.frameOnBox(box);
  }

  frameOnGeometry(geometry) {
    geometry.computeBoundingBox();
    this.frameOnBox(geometry.boundingBox);
  }
  frameOnBox(box3, animate = true) {
    const center = new THREE.Vector3(),
      size = new THREE.Vector3();
    box3.getCenter(center);
    box3.getSize(size);
    const d = Math.max(size.x, size.y, size.z) * 2.5;
    const pos = new THREE.Vector3(center.x + d * 0.6, center.y - d * 0.6, center.z + d * 0.8);
    const up = new THREE.Vector3(0, 0, 1);
    if (this._cameraType === 'orthographic') {
      this._fitOrthoFrustum(box3);
    }
    if (animate && this._animating) {
      this._startLerp(pos, center, up);
    } else {
      this._cam.position.copy(pos);
      this._cam.up.copy(up);
      this._ctrl.target.copy(center);
      this._ctrl.update();
    }
  }
  setViewPreset(name) {
    const p = VIEW_PRESETS[name];
    if (!p) return;

    // If already at this angle, zoom to fit instead of re-animating
    const currentDir = new THREE.Vector3();
    currentDir.subVectors(this._cam.position, this._ctrl.target).normalize();
    const presetDir = p.dir.clone().normalize();
    if (currentDir.dot(presetDir) > 0.99) {
      this.zoomToFit();
      return;
    }

    const box = this._bbox();
    const c = box ? box.getCenter(new THREE.Vector3()) : new THREE.Vector3();
    const sz = box ? box.getSize(new THREE.Vector3()) : new THREE.Vector3(0.1, 0.1, 0.1);
    const d = Math.max(sz.x, sz.y, sz.z) * 2.5;
    // In follow mode, maintain focus on the followed target instead of bbox center
    const target = this._followMode && this._followTarget ? this._followTarget.clone() : c;
    const pos = target.clone().addScaledVector(p.dir, d);
    if (this._cameraType === 'orthographic' && box) {
      this._fitOrthoFrustum(box);
    }
    if (this._animating) this._startLerp(pos, target, p.up.clone());
    else {
      this._cam.position.copy(pos);
      this._cam.up.copy(p.up);
      this._ctrl.target.copy(target);
      this._cam.lookAt(target);
      this._ctrl.update();
    }
  }

  /**
   * Set the camera to look from a given direction (used for edge/corner clicks).
   * @param {THREE.Vector3} dir — normalized direction the camera looks FROM
   * @param {THREE.Vector3} [up] — up vector (defaults to Z-up or Y-up for top/bottom)
   */
  setViewFromDirection(dir: any, up?: any) {
    const box = this._bbox();
    const c = box ? box.getCenter(new THREE.Vector3()) : new THREE.Vector3();
    const sz = box ? box.getSize(new THREE.Vector3()) : new THREE.Vector3(0.1, 0.1, 0.1);
    const d = Math.max(sz.x, sz.y, sz.z) * 2.5;
    const target = this._followMode && this._followTarget ? this._followTarget.clone() : c;
    const pos = target.clone().addScaledVector(dir, d);
    // Pick a sensible up vector: Z-up unless we're looking straight along Z
    if (!up) {
      up = new THREE.Vector3(0, 0, 1);
      if (Math.abs(dir.dot(up)) > 0.9) up = new THREE.Vector3(0, 1, 0);
    }
    if (this._cameraType === 'orthographic' && box) {
      this._fitOrthoFrustum(box);
    }
    if (this._animating) this._startLerp(pos, target, up.clone());
    else {
      this._cam.position.copy(pos);
      this._cam.up.copy(up);
      this._ctrl.target.copy(target);
      this._cam.lookAt(target);
      this._ctrl.update();
    }
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

  isFollowMode() {
    return this._followMode;
  }

  onFollowChange(cb) {
    this._onFollowChange = cb;
  }

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
        this._followBadge.style.cssText =
          'position:absolute;top:8px;left:50%;transform:translateX(-50%);background:rgba(145,121,242,0.8);color:white;font:500 11px system-ui,-apple-system,sans-serif;padding:2px 10px;border-radius:10px;pointer-events:none;z-index:60;transition:opacity 0.2s;';
        this._followBadge.textContent = 'Following';
        this._overlay.appendChild(this._followBadge);
      }
      this._followBadge.style.opacity = '1';
      this._followBadge.style.display = '';
    } else if (this._followBadge) {
      this._followBadge.style.opacity = '0';
      setTimeout(() => {
        if (this._followBadge && !this._followMode) this._followBadge.style.display = 'none';
      }, 200);
    }
  }

  // ── Ghosting ──

  /**
   * Ghost a mesh — save original material state and make very transparent.
   * @param {THREE.Mesh} mesh
   */
  ghostMesh(mesh) {
    if (!mesh || !mesh.material || this._ghostedMeshes.has(mesh)) return;
    this._ghostedMeshes.set(mesh, {
      opacity: mesh.material.opacity,
      transparent: mesh.material.transparent,
    });
    mesh.material.opacity = 0.06;
    mesh.material.transparent = true;
  }

  /**
   * Restore a ghosted mesh to its original material state.
   * @param {THREE.Mesh} mesh
   */
  unghostMesh(mesh) {
    const saved = this._ghostedMeshes.get(mesh);
    if (!saved) return;
    mesh.material.opacity = saved.opacity;
    mesh.material.transparent = saved.transparent;
    this._ghostedMeshes.delete(mesh);
  }

  /**
   * Unghost all currently ghosted meshes.
   */
  unghostAll() {
    for (const [mesh, saved] of this._ghostedMeshes) {
      if (mesh.material) {
        mesh.material.opacity = saved.opacity;
        mesh.material.transparent = saved.transparent;
      }
    }
    this._ghostedMeshes.clear();
  }

  // ── Geometry disposal ──

  /**
   * Dispose all children of a group, cleaning up geometry and materials.
   * @param {THREE.Group} group
   */
  clearGroup(group) {
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      child.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
      });
    }
  }

  enableMeasureTool() {
    this._meas.enable();
    this._ctrl.enabled = false;
    this._setActiveTool('measure');
  }
  disableMeasureTool() {
    this._meas.disable();
    this._ctrl.enabled = true;
    if (this._activeTool === 'measure') this._setActiveTool(null);
    // Clear all measurements when deactivating the tool
    if (this._meas.clearAll) this._meas.clearAll();
    else if (this._meas.measurements) this._meas.measurements.length = 0;
  }

  enableSectionPlane(axis = 'z', frac = 0.5) {
    this._secOn = true;
    this._secAxis = axis;
    this._secFrac = frac;
    this._setActiveTool('section');
    this._secPopover.style.display = '';
    this._hiliteAxis();
    this._secSlider.value = String(Math.round(frac * 100));
    this._applySection();
  }
  disableSectionPlane() {
    this._secOn = false;
    if (this._activeTool === 'section') this._setActiveTool(null);
    this._secPopover.style.display = 'none';
    if (this._secViz) this._secViz.visible = false;
    this._clearSectionCaps();
    this._scene.traverse((ch) => {
      if (ch.material) ch.material.clippingPlanes = [];
    });
  }

  /** Whether section plane is currently active. */
  get sectionEnabled() {
    return this._secOn;
  }
  /** The current section THREE.Plane (read-only). */
  get sectionPlane() {
    return this._secPlane;
  }

  /**
   * Trigger a section plane rebuild (e.g. after layer visibility changes).
   */
  updateSection() {
    if (this._secOn) this._applySection();
  }

  animate(cb) {
    this._animCb = cb || null;
    if (!this._animating) {
      this._animating = true;
      this._tick();
    }
  }
  resize() {
    const w = this._container.clientWidth,
      h = this._container.clientHeight;
    if (!w || !h) return;
    if (this._cameraType === 'orthographic') {
      // Maintain vertical extent, adjust horizontal by aspect
      const halfH = this._cam.top;
      const aspect = w / h;
      this._cam.left = -halfH * aspect;
      this._cam.right = halfH * aspect;
      this._cam.updateProjectionMatrix();
    } else {
      this._cam.aspect = w / h;
      this._cam.updateProjectionMatrix();
    }
    this._ren.setSize(w, h);
    if (this._edgeC) this._edgeC.resize(w, h);
    if (this._contourLineMat) {
      this._contourLineMat.resolution.set(w, h);
    }
  }
  dispose() {
    this._disposed = true;
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('keydown', this._onKey);
    this._meas.disable();
    this._ctrl.dispose();
    this._ren.dispose();
    if (this._cubeRen) this._cubeRen.dispose();
    if (this._overlay) this._overlay.remove();
    if (this._cubeCanvas) this._cubeCanvas.remove();
    this._clearSectionCaps();
    this.unghostAll();
  }

  // ── Ortho frustum helpers ──

  _fitOrthoFrustum(box) {
    if (!box || this._cameraType !== 'orthographic') return;
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const pad = maxDim * 0.15;
    const w = this._container.clientWidth || 1,
      h = this._container.clientHeight || 1;
    const aspect = w / h;
    const halfH = maxDim / 2 + pad;
    const halfW = halfH * aspect;
    this._cam.left = -halfW;
    this._cam.right = halfW;
    this._cam.top = halfH;
    this._cam.bottom = -halfH;
    this._cam.updateProjectionMatrix();
  }

  /**
   * Switch between perspective and orthographic cameras at runtime,
   * preserving the current view direction and target.
   */
  _switchCamera(type) {
    if (type === this._cameraType) return;
    const oldPos = this._cam.position.clone();
    const oldUp = this._cam.up.clone();
    const target = this._ctrl.target.clone();
    const w = this._container.clientWidth || 1,
      h = this._container.clientHeight || 1;

    if (type === 'orthographic') {
      // Compute ortho frustum from perspective view distance
      const dist = oldPos.distanceTo(target);
      const halfH = dist * Math.tan(THREE.MathUtils.degToRad(this._cam.fov / 2));
      const aspect = w / h;
      this._cam = new THREE.OrthographicCamera(-halfH * aspect, halfH * aspect, halfH, -halfH, 0.0001, 10);
    } else {
      // Compute perspective position from ortho frustum
      const halfH = this._cam.top;
      const fov = 45;
      const dist = halfH / Math.tan(THREE.MathUtils.degToRad(fov / 2));
      this._cam = new THREE.PerspectiveCamera(fov, w / h, 0.0001, 10);
      // Adjust position to match the ortho view distance
      const dir = new THREE.Vector3().subVectors(oldPos, target).normalize();
      oldPos.copy(target).addScaledVector(dir, dist);
    }
    this._cameraType = type;
    this._cam.position.copy(oldPos);
    this._cam.up.copy(oldUp);
    this._cam.lookAt(target);
    this._cam.updateProjectionMatrix();

    // Reconnect controls
    this._ctrl.dispose();
    this._ctrl = new OrbitControls(this._cam, this._ren.domElement);
    this._ctrl.enableDamping = true;
    this._ctrl.dampingFactor = 0.1;
    this._ctrl.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_ROTATE };
    this._ctrl.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this._ctrl.target.copy(target);
    this._ctrl.update();

    // Reconnect edge composer if active
    if (this._edgeC) {
      this._edgeC = createEdgeComposer(this._ren, this._scene, this._cam);
    }
    // Reconnect measure tool camera reference
    this._meas._camera = this._cam;
  }

  // ── Scene init ──
  _initScene(opts) {
    const c = this._container,
      w = c.clientWidth || 800,
      h = c.clientHeight || 600;
    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0xf5f8fa);

    if (this._cameraType === 'orthographic') {
      const aspect = w / h;
      const frustum = 0.06;
      this._cam = new THREE.OrthographicCamera(-frustum * aspect, frustum * aspect, frustum, -frustum, 0.0001, 10);
      this._cam.position.set(0.06, -0.06, 0.05);
    } else {
      this._cam = new THREE.PerspectiveCamera(45, w / h, 0.0001, 10);
      this._cam.position.set(0.08, -0.08, 0.1);
    }
    this._cam.up.set(0, 0, 1);

    const rendererOpts: any = { antialias: true };
    if (this._cameraType !== 'orthographic') rendererOpts.logarithmicDepthBuffer = true;
    this._ren = new THREE.WebGLRenderer(rendererOpts);
    this._ren.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._ren.setSize(w, h);
    this._ren.shadowMap.enabled = true;
    this._ren.shadowMap.type = THREE.PCFShadowMap;
    this._ren.localClippingEnabled = true;
    c.appendChild(this._ren.domElement);
    this._ctrl = new OrbitControls(this._cam, this._ren.domElement);
    this._ctrl.enableDamping = true;
    this._ctrl.dampingFactor = 0.1;
    // Trackpad: two-finger drag = rotate (not pan), pinch = zoom
    this._ctrl.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_ROTATE };
    this._ctrl.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    this._ctrl.update();
    // Lighting
    this._scene.add(new THREE.AmbientLight(0xffffff, 1.0));
    const dir = new THREE.DirectionalLight(0xffffff, 1.6);
    dir.position.set(0.3, 0.5, 0.4);
    this._scene.add(dir);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-0.3, -0.2, -0.4);
    this._scene.add(fill);
    this._gridHelper = null;
    if (opts.grid) {
      this._gridHelper = new THREE.GridHelper(0.3, 30, 0xced9e0, 0xe8edf0);
      this._gridHelper.rotation.x = Math.PI / 2;
      this._scene.add(this._gridHelper);
    }
    if (opts.edges) this._edgeC = createEdgeComposer(this._ren, this._scene, this._cam);
    this._meas = new MeasureTool(this._cam, this._scene, c);
  }

  // ── Overlay controls ──
  _initOverlay() {
    const ov = document.createElement('div');
    ov.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:50;';
    this._container.appendChild(ov);
    this._overlay = ov;

    this._initOrientationCube();
    this._initToolStrip();
  }

  // ── Orientation Cube (top-right) — polished with beveled edges ──
  _initOrientationCube() {
    const SIZE = 110;
    const PIXEL_RATIO = Math.min(window.devicePixelRatio, 2);

    // Container with subtle shadow
    const cubeContainer = document.createElement('div');
    cubeContainer.style.cssText = `position:absolute;top:8px;right:8px;width:${SIZE}px;height:${SIZE}px;pointer-events:auto;border-radius:8px;filter:drop-shadow(0 2px 6px rgba(0,0,0,0.15));`;
    this._overlay.appendChild(cubeContainer);
    this._cubeContainer = cubeContainer;

    // Canvas element
    const canvas = document.createElement('canvas');
    canvas.width = SIZE * PIXEL_RATIO;
    canvas.height = SIZE * PIXEL_RATIO;
    canvas.style.cssText = `width:${SIZE}px;height:${SIZE}px;cursor:pointer;border-radius:8px;`;
    cubeContainer.appendChild(canvas);
    this._cubeCanvas = canvas;

    // Separate renderer with antialiasing
    this._cubeRen = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    this._cubeRen.setPixelRatio(PIXEL_RATIO);
    this._cubeRen.setSize(SIZE, SIZE);
    this._cubeRen.setClearColor(0x000000, 0);

    // Scene + camera
    this._cubeScene = new THREE.Scene();
    this._cubeCam = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    this._cubeCam.position.set(0, 0, 4);

    // Lighting for cube — multi-directional for gradient effect
    this._cubeScene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const cubeDir = new THREE.DirectionalLight(0xffffff, 0.7);
    cubeDir.position.set(2, 3, 4);
    this._cubeScene.add(cubeDir);
    const cubeFill = new THREE.DirectionalLight(0xffffff, 0.25);
    cubeFill.position.set(-2, -1, -2);
    this._cubeScene.add(cubeFill);

    // Per-face canvas rotation to ensure text reads correctly from each viewing direction.
    // THREE.BoxGeometry UV layout in our Z-up coordinate system requires compensation:
    // Rotation values in radians applied via ctx.rotate around canvas center.
    const CUBE_FACE_ROTATION = [
      -Math.PI / 2, // +X (Right): UV text appears rotated 90° CW, counter-rotate
      Math.PI / 2, // -X (Left):  UV text appears rotated 90° CCW, rotate CW
      Math.PI, // +Y (Back):  UV text appears upside down, rotate 180°
      0, // -Y (Front): text reads correctly
      0, // +Z (Top):   text reads correctly from above
      Math.PI, // -Z (Bottom): text appears upside down, rotate 180°
    ];

    // Create face materials with canvas textures — cleaner labels
    const materials = CUBE_FACE_MAP.map((face, i) => {
      const texCanvas = document.createElement('canvas');
      texCanvas.width = 128;
      texCanvas.height = 128;
      const ctx = texCanvas.getContext('2d');
      // Fill with face color
      ctx.fillStyle = CUBE_FACE_COLORS[i];
      ctx.fillRect(0, 0, 128, 128);
      // Subtle inner border for bevel feel
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 2;
      ctx.strokeRect(3, 3, 122, 122);
      // Apply per-face rotation for correct text orientation
      const rot = CUBE_FACE_ROTATION[i];
      if (rot) {
        ctx.save();
        ctx.translate(64, 64);
        ctx.rotate(rot);
        ctx.translate(-64, -64);
      }
      // Crisp smaller text
      ctx.fillStyle = '#5C7080';
      ctx.font = '500 18px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(face.label, 64, 64);
      if (rot) ctx.restore();
      const tex = new THREE.CanvasTexture(texCanvas);
      tex.colorSpace = THREE.SRGBColorSpace;
      return new THREE.MeshStandardMaterial({ map: tex, roughness: 0.65, metalness: 0.02 });
    });

    // Store base textures for hover highlighting with smooth transition
    this._cubeFaceTextures = materials.map((_, i) => ({
      baseColor: CUBE_FACE_COLORS[i],
      label: CUBE_FACE_MAP[i].label,
      rotation: CUBE_FACE_ROTATION[i],
      brightness: 0, // 0 = base, 1 = hover highlight (lerped)
      targetBrightness: 0,
    }));

    // Use a slightly beveled box (rounded corners via segments)
    const geo = this._createRoundedBoxGeometry(1, 1, 1, 0.08, 2);
    this._cubeMesh = new THREE.Mesh(geo, materials);
    this._cubeScene.add(this._cubeMesh);

    // Soft edge wireframe
    const edges = new THREE.EdgesGeometry(geo, 20);
    const lineMat = new THREE.LineBasicMaterial({ color: 0x394b59, linewidth: 1, transparent: true, opacity: 0.4 });
    const wireframe = new THREE.LineSegments(edges, lineMat);
    this._cubeMesh.add(wireframe);

    // Axis arrows at the bottom-front-left corner of the cube
    this._buildCubeAxisArrows();

    // Raycaster for click/hover detection
    this._cubeRaycaster = new THREE.Raycaster();
    this._cubeHoveredFace = -1;

    // Tooltip
    this._cubeTooltip = document.createElement('div');
    this._cubeTooltip.style.cssText =
      'position:absolute;pointer-events:none;background:rgba(28,33,39,0.9);color:#E8EDF0;font:500 12px system-ui,-apple-system,sans-serif;padding:4px 8px;border-radius:4px;white-space:nowrap;display:none;z-index:60;';
    this._overlay.appendChild(this._cubeTooltip);

    // Events
    canvas.addEventListener('mousemove', (e) => this._onCubeHover(e));
    canvas.addEventListener('mouseleave', () => this._onCubeLeave());
    canvas.addEventListener('click', (e) => this._onCubeClick(e));
  }

  /**
   * Create a rounded box geometry by subdividing faces.
   * Simple approach: create a box and adjust vertex positions to soften edges.
   */
  _createRoundedBoxGeometry(w, h, d, radius, segments) {
    // Use standard BoxGeometry with enough segments for rounding
    const geo = new THREE.BoxGeometry(w, h, d, segments + 1, segments + 1, segments + 1);
    const pos = geo.attributes.position;
    const v = new THREE.Vector3();
    const halfW = w / 2,
      halfH = h / 2,
      halfD = d / 2;

    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      // Compute how far each component is from the interior box
      const dx = Math.max(0, Math.abs(v.x) - (halfW - radius));
      const dy = Math.max(0, Math.abs(v.y) - (halfH - radius));
      const dz = Math.max(0, Math.abs(v.z) - (halfD - radius));
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist > 0) {
        const scale = radius / dist;
        if (scale < 1) {
          // Pull vertex inward to round the corner/edge
          if (Math.abs(v.x) > halfW - radius) v.x = Math.sign(v.x) * (halfW - radius + dx * scale);
          if (Math.abs(v.y) > halfH - radius) v.y = Math.sign(v.y) * (halfH - radius + dy * scale);
          if (Math.abs(v.z) > halfD - radius) v.z = Math.sign(v.z) * (halfD - radius + dz * scale);
        }
      }
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geo.computeVertexNormals();
    return geo;
  }

  _getCubeHitInfo(e) {
    const rect = this._cubeCanvas.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this._cubeRaycaster.setFromCamera(mouse, this._cubeCam);
    const hits = this._cubeRaycaster.intersectObject(this._cubeMesh);
    if (hits.length === 0) return { type: 'none', faceIndex: -1 };

    const hit = hits[0];
    const point = hit.point;

    // Determine if the hit is on a face center, edge, or corner
    // by checking how close the point is to the cube axes
    const abs = new THREE.Vector3(Math.abs(point.x), Math.abs(point.y), Math.abs(point.z));
    const sorted = [abs.x, abs.y, abs.z].sort((a, b) => b - a);

    // Thresholds for edge/corner detection on the rounded cube
    const edgeThreshold = 0.32;
    const cornerThreshold = 0.32;

    // How many axes are close to the surface (above threshold)?
    const nearSurface = [abs.x, abs.y, abs.z].filter((v) => v > edgeThreshold).length;

    if (nearSurface >= 3) {
      // Corner: all three axes are significant — use preset corner directions
      // that match VIEW_PRESETS.iso Z-bias (0.8) for consistent views
      const sx = Math.sign(point.x),
        sy = Math.sign(point.y),
        sz = Math.sign(point.z);
      const dir = new THREE.Vector3(sx, sy, sz * 0.8).normalize();
      return { type: 'corner', dir };
    } else if (nearSurface >= 2 && sorted[1] > cornerThreshold) {
      // Edge: two axes are significant
      const dir = new THREE.Vector3(
        abs.x > edgeThreshold ? Math.sign(point.x) : 0,
        abs.y > edgeThreshold ? Math.sign(point.y) : 0,
        abs.z > edgeThreshold ? Math.sign(point.z) : 0,
      ).normalize();
      return { type: 'edge', dir };
    } else {
      // Face center
      const _fi = Math.floor(
        hit.faceIndex / ((this._cubeMesh.geometry.index ? this._cubeMesh.geometry.index.count / 3 : 1) / 6),
      );
      // Use a simpler approach: find the dominant axis
      let maxAxis = 0;
      if (abs.y > abs.x && abs.y > abs.z) maxAxis = 1;
      if (abs.z > abs.x && abs.z > abs.y) maxAxis = 2;
      const sign = [point.x, point.y, point.z][maxAxis] > 0 ? 1 : -1;
      // Map to face index: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
      const faceIndex = maxAxis * 2 + (sign > 0 ? 0 : 1);
      return { type: 'face', faceIndex };
    }
  }

  _onCubeHover(e) {
    const info = this._getCubeHitInfo(e);

    // Restore all faces first
    for (let i = 0; i < CUBE_FACE_MAP.length; i++) {
      this._cubeFaceTextures[i].targetBrightness = 0;
    }

    let tooltipText = null;

    if (info.type === 'face' && info.faceIndex >= 0 && info.faceIndex < CUBE_FACE_MAP.length) {
      this._cubeFaceTextures[info.faceIndex].targetBrightness = 1;
      const face = CUBE_FACE_MAP[info.faceIndex];
      tooltipText = `${face.label} view (${face.key})`;
    } else if (info.type === 'edge') {
      tooltipText = 'Edge view';
    } else if (info.type === 'corner') {
      tooltipText = 'Corner view';
    }

    // Update hover brightness with smooth transition
    this._updateCubeFaceBrightness();

    if (tooltipText) {
      this._cubeTooltip.textContent = tooltipText;
      const rect = this._cubeCanvas.getBoundingClientRect();
      const containerRect = this._container.getBoundingClientRect();
      this._cubeTooltip.style.display = '';
      this._cubeTooltip.style.top = `${e.clientY - containerRect.top - 12}px`;
      this._cubeTooltip.style.right = `${containerRect.right - rect.left + 6}px`;
    } else {
      this._cubeTooltip.style.display = 'none';
    }

    this._cubeCanvas.style.cursor = info.type !== 'none' ? 'pointer' : 'default';
    this._cubeHoverInfo = info;
  }

  _onCubeLeave() {
    for (let i = 0; i < CUBE_FACE_MAP.length; i++) {
      this._cubeFaceTextures[i].targetBrightness = 0;
    }
    this._updateCubeFaceBrightness();
    this._cubeTooltip.style.display = 'none';
    this._cubeHoverInfo = null;
  }

  _onCubeClick(e) {
    const info = this._getCubeHitInfo(e);
    if (info.type === 'face' && info.faceIndex >= 0 && info.faceIndex < CUBE_FACE_MAP.length) {
      this.setViewPreset(CUBE_FACE_MAP[info.faceIndex].preset);
    } else if (info.type === 'edge' || info.type === 'corner') {
      this.setViewFromDirection(info.dir);
    }
  }

  _updateCubeFaceBrightness() {
    // Smoothly interpolate brightness toward target
    const speed = 0.25;
    let needsUpdate = false;

    for (let i = 0; i < CUBE_FACE_MAP.length; i++) {
      const ft = this._cubeFaceTextures[i];
      const prev = ft.brightness;
      ft.brightness += (ft.targetBrightness - ft.brightness) * speed;
      // Snap when close
      if (Math.abs(ft.brightness - ft.targetBrightness) < 0.01) {
        ft.brightness = ft.targetBrightness;
      }
      if (ft.brightness !== prev) {
        this._redrawCubeFace(i);
        needsUpdate = true;
      }
    }
    return needsUpdate;
  }

  _redrawCubeFace(faceIdx) {
    const info = this._cubeFaceTextures[faceIdx];
    const mat = this._cubeMesh.material[faceIdx];
    const texCanvas = mat.map.image;
    const ctx = texCanvas.getContext('2d');
    const t = info.brightness;

    // Interpolate between base color and hover color
    const baseR = parseInt(info.baseColor.slice(1, 3), 16);
    const baseG = parseInt(info.baseColor.slice(3, 5), 16);
    const baseB = parseInt(info.baseColor.slice(5, 7), 16);
    const hoverR = 0xbc,
      hoverG = 0xc7,
      hoverB = 0xcf;
    const r = Math.round(baseR + (hoverR - baseR) * t);
    const g = Math.round(baseG + (hoverG - baseG) * t);
    const b = Math.round(baseB + (hoverB - baseB) * t);

    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, 0, 128, 128);
    // Subtle inner bevel
    ctx.strokeStyle = `rgba(255,255,255,${0.25 + t * 0.15})`;
    ctx.lineWidth = 2;
    ctx.strokeRect(3, 3, 122, 122);
    // Apply per-face rotation for correct text orientation
    const rot = info.rotation;
    if (rot) {
      ctx.save();
      ctx.translate(64, 64);
      ctx.rotate(rot);
      ctx.translate(-64, -64);
    }
    // Text — darker on hover for better contrast
    const textR = Math.round(0x5c + (0x18 - 0x5c) * t);
    const textG = Math.round(0x70 + (0x20 - 0x70) * t);
    const textB = Math.round(0x80 + (0x26 - 0x80) * t);
    ctx.fillStyle = `rgb(${textR},${textG},${textB})`;
    ctx.font = '500 18px system-ui, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(info.label, 64, 64);
    if (rot) ctx.restore();
    mat.map.needsUpdate = true;
  }

  _syncOrientationCube() {
    // Smooth face brightness transitions
    this._updateCubeFaceBrightness();

    // Mirror main camera orientation onto cube camera
    const dir = new THREE.Vector3();
    this._cam.getWorldDirection(dir);
    const dist = 4;
    this._cubeCam.position.copy(dir.multiplyScalar(-dist));
    this._cubeCam.up.copy(this._cam.up);
    this._cubeCam.lookAt(0, 0, 0);
    this._cubeRen.render(this._cubeScene, this._cubeCam);
  }

  /**
   * Build colored axis arrows (X=red, Y=green, Z=blue) anchored at the
   * bottom-front-left corner of the orientation cube.  Each arrow is a
   * short line + a sprite label at the tip, all added to the cube scene
   * so they rotate with the camera.
   */
  _buildCubeAxisArrows() {
    const AXIS_LEN = 0.95;
    const AXIS_CFG = [
      { dir: [1, 0, 0], color: 0xdb3737, label: 'X' },
      { dir: [0, 1, 0], color: 0x0f9960, label: 'Y' },
      { dir: [0, 0, 1], color: 0x2b95d6, label: 'Z' },
    ];
    // Anchor point: corner of the cube
    const origin = new THREE.Vector3(-0.55, -0.55, -0.55);
    const axisGroup = new THREE.Group();
    axisGroup.position.copy(origin);

    for (const { dir, color, label } of AXIS_CFG) {
      // Line from origin to tip
      const pts = new Float32Array([0, 0, 0, dir[0] * AXIS_LEN, dir[1] * AXIS_LEN, dir[2] * AXIS_LEN]);
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
      const lineMat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      axisGroup.add(new THREE.Line(lineGeo, lineMat));

      // Cone arrowhead at tip
      const coneGeo = new THREE.ConeGeometry(0.06, 0.15, 8);
      const coneMat = new THREE.MeshBasicMaterial({ color });
      const cone = new THREE.Mesh(coneGeo, coneMat);
      const tipPos = new THREE.Vector3(dir[0] * AXIS_LEN, dir[1] * AXIS_LEN, dir[2] * AXIS_LEN);
      cone.position.copy(tipPos);
      // Orient cone along axis direction
      const up = new THREE.Vector3(0, 1, 0);
      const axDir = new THREE.Vector3(...dir);
      if (Math.abs(axDir.dot(up)) < 0.999) {
        cone.quaternion.setFromUnitVectors(up, axDir);
      } else {
        // axis is along Y — rotate to match
        if (axDir.y < 0) cone.rotation.z = Math.PI;
      }
      axisGroup.add(cone);

      // Sprite label just past the arrowhead tip
      const labelCanvas = document.createElement('canvas');
      labelCanvas.width = 64;
      labelCanvas.height = 64;
      const ctx = labelCanvas.getContext('2d');
      ctx.clearRect(0, 0, 64, 64);
      ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
      ctx.font = 'bold 42px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, 32, 32);
      const spriteTex = new THREE.CanvasTexture(labelCanvas);
      spriteTex.colorSpace = THREE.SRGBColorSpace;
      const spriteMat = new THREE.SpriteMaterial({ map: spriteTex, transparent: true, depthTest: false });
      const sprite = new THREE.Sprite(spriteMat);
      const labelOffset = 0.22;
      sprite.position.set(
        dir[0] * (AXIS_LEN + labelOffset),
        dir[1] * (AXIS_LEN + labelOffset),
        dir[2] * (AXIS_LEN + labelOffset),
      );
      sprite.scale.set(0.28, 0.28, 1);
      axisGroup.add(sprite);
    }

    this._cubeScene.add(axisGroup);
  }

  // ── Vertical Tool Strip (left edge) ──
  _initToolStrip() {
    const strip = document.createElement('div');
    strip.style.cssText =
      'position:absolute;left:8px;top:50%;transform:translateY(-50%);pointer-events:auto;background:rgba(28,33,39,0.85);border-radius:8px;padding:4px;display:flex;flex-direction:column;gap:2px;z-index:51;';
    this._overlay.appendChild(strip);
    this._toolStrip = strip;

    const toolBtnCSS =
      'width:36px;height:36px;border:none;border-radius:6px;background:transparent;color:#CED9E0;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.12s,color 0.12s;position:relative;';

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
    this._secPopover.style.cssText =
      'position:absolute;left:52px;top:50%;transform:translateY(-50%);pointer-events:auto;background:rgba(28,33,39,0.92);border-radius:8px;padding:10px;display:none;box-shadow:0 4px 12px rgba(0,0,0,0.25);font:12px system-ui,-apple-system,sans-serif;color:#CED9E0;min-width:120px;z-index:52;';

    // Axis selector
    const axLabel = document.createElement('div');
    axLabel.style.cssText =
      'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:#8A9BA8;margin-bottom:6px;';
    axLabel.textContent = 'Axis';
    this._secPopover.appendChild(axLabel);

    const axRow = document.createElement('div');
    axRow.style.cssText = 'display:flex;gap:3px;margin-bottom:8px;';
    this._secAxisBtns = {};
    const axisBtnCSS =
      'width:32px;height:26px;border:1px solid rgba(206,217,224,0.2);border-radius:4px;background:transparent;color:#CED9E0;font:600 11px system-ui,-apple-system,sans-serif;cursor:pointer;transition:all 0.12s;';
    for (const ax of ['x', 'y', 'z']) {
      const b = document.createElement('button');
      b.style.cssText = axisBtnCSS;
      b.textContent = ax.toUpperCase();
      b.addEventListener('click', () => {
        this._secAxis = ax;
        this._hiliteAxis();
        this._applySection();
      });
      axRow.appendChild(b);
      this._secAxisBtns[ax] = b;
    }
    // Flip button
    const flipBtn = document.createElement('button');
    flipBtn.style.cssText = `${axisBtnCSS}width:auto;padding:0 8px;`;
    flipBtn.textContent = 'Flip';
    flipBtn.addEventListener('click', () => {
      this._secFlipped = !this._secFlipped;
      this._applySection();
    });
    axRow.appendChild(flipBtn);

    this._secPopover.appendChild(axRow);

    // Slider
    const sliderLabel = document.createElement('div');
    sliderLabel.style.cssText =
      'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:#8A9BA8;margin-bottom:4px;';
    sliderLabel.textContent = 'Position';
    this._secPopover.appendChild(sliderLabel);

    this._secSlider = document.createElement('input');
    Object.assign(this._secSlider, { type: 'range', min: '0', max: '100', value: '50' });
    this._secSlider.style.cssText = 'width:100%;margin:0;cursor:pointer;accent-color:#137CBD;';
    this._secSlider.addEventListener('input', () => {
      this._secFrac = parseFloat(this._secSlider.value) / 100;
      this._applySection();
    });
    this._secPopover.appendChild(this._secSlider);

    // Attach popover next to strip
    strip.style.position = 'absolute'; // ensure positioning context
    this._overlay.appendChild(this._secPopover);

    // Keep popover positioned next to the strip
    this._positionSecPopover();

    // ── Settings button (bottom of strip) ──
    const settingsDivider = document.createElement('div');
    settingsDivider.style.cssText = 'height:1px;margin:2px 4px;background:rgba(206,217,224,0.2);';
    strip.appendChild(settingsDivider);

    this._settingsBtn = document.createElement('button');
    this._settingsBtn.style.cssText = toolBtnCSS;
    this._settingsBtn.innerHTML = ICONS.settings;
    this._settingsBtn.addEventListener('click', () => {
      const vis = this._settingsPopover.style.display === 'none';
      this._settingsPopover.style.display = vis ? '' : 'none';
      if (vis && this._updateSettingsPopoverPos) this._updateSettingsPopoverPos();
    });
    strip.appendChild(this._settingsBtn);
    this._addToolTooltip(this._settingsBtn, 'Settings');

    // Settings popover
    this._settingsPopover = document.createElement('div');
    this._settingsPopover.style.cssText =
      'position:absolute;left:52px;top:50%;transform:translateY(-50%);pointer-events:auto;background:rgba(28,33,39,0.92);border-radius:8px;padding:10px;display:none;box-shadow:0 4px 12px rgba(0,0,0,0.25);font:12px system-ui,-apple-system,sans-serif;color:#CED9E0;min-width:140px;z-index:52;';

    const settingsTitle = document.createElement('div');
    settingsTitle.style.cssText =
      'font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:#8A9BA8;margin-bottom:8px;';
    settingsTitle.textContent = 'Settings';
    this._settingsPopover.appendChild(settingsTitle);

    // Camera type toggle
    const camRow = document.createElement('div');
    camRow.style.cssText = 'display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;';
    const camLabel = document.createElement('span');
    camLabel.textContent = 'Camera';
    camLabel.style.cssText = 'color:#CED9E0;font-size:12px;';
    camRow.appendChild(camLabel);

    const camToggle = document.createElement('div');
    camToggle.style.cssText = 'display:flex;border:1px solid rgba(206,217,224,0.2);border-radius:4px;overflow:hidden;';
    const camBtnCSS =
      'border:none;padding:2px 8px;font:500 10px system-ui,-apple-system,sans-serif;cursor:pointer;transition:all 0.12s;';
    this._perspBtn = document.createElement('button');
    this._perspBtn.textContent = 'Persp';
    this._perspBtn.style.cssText = camBtnCSS;
    this._orthoBtn = document.createElement('button');
    this._orthoBtn.textContent = 'Ortho';
    this._orthoBtn.style.cssText = camBtnCSS;

    const updateCamBtns = () => {
      const isPersp = this._cameraType === 'perspective';
      this._perspBtn.style.background = isPersp ? '#30363D' : 'transparent';
      this._perspBtn.style.color = isPersp ? '#E8EDF0' : '#738694';
      this._orthoBtn.style.background = !isPersp ? '#30363D' : 'transparent';
      this._orthoBtn.style.color = !isPersp ? '#E8EDF0' : '#738694';
    };

    this._perspBtn.addEventListener('click', () => {
      if (this._cameraType !== 'perspective') this._switchCamera('perspective');
      updateCamBtns();
    });
    this._orthoBtn.addEventListener('click', () => {
      if (this._cameraType !== 'orthographic') this._switchCamera('orthographic');
      updateCamBtns();
    });
    camToggle.appendChild(this._perspBtn);
    camToggle.appendChild(this._orthoBtn);
    camRow.appendChild(camToggle);
    this._settingsPopover.appendChild(camRow);
    updateCamBtns();

    // Edge rendering checkbox
    const edgeRow = document.createElement('label');
    edgeRow.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:6px;cursor:pointer;';
    this._edgeCb = document.createElement('input');
    this._edgeCb.type = 'checkbox';
    this._edgeCb.checked = !!this._edgeC;
    this._edgeCb.style.cssText = 'width:12px;height:12px;accent-color:#137CBD;cursor:pointer;';
    this._edgeCb.addEventListener('change', () => {
      if (this._edgeCb.checked) {
        if (!this._edgeC) this._edgeC = createEdgeComposer(this._ren, this._scene, this._cam);
      } else {
        if (this._edgeC) {
          this._edgeC = null;
        }
      }
    });
    edgeRow.appendChild(this._edgeCb);
    const edgeLabel = document.createElement('span');
    edgeLabel.textContent = 'Edge detection';
    edgeLabel.style.cssText = 'color:#CED9E0;font-size:12px;';
    edgeRow.appendChild(edgeLabel);
    this._settingsPopover.appendChild(edgeRow);

    // Grid checkbox
    const gridRow = document.createElement('label');
    gridRow.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;';
    this._gridCb = document.createElement('input');
    this._gridCb.type = 'checkbox';
    // Grid reference is set during _initScene if opts.grid was true
    this._gridCb.checked = !!this._gridHelper?.visible;
    this._gridCb.style.cssText = 'width:12px;height:12px;accent-color:#137CBD;cursor:pointer;';
    this._gridCb.addEventListener('change', () => {
      if (this._gridCb.checked) {
        if (!this._gridHelper) {
          this._gridHelper = new THREE.GridHelper(0.3, 30, 0xced9e0, 0xe8edf0);
          this._gridHelper.rotation.x = Math.PI / 2;
          this._scene.add(this._gridHelper);
        }
        this._gridHelper.visible = true;
      } else if (this._gridHelper) {
        this._gridHelper.visible = false;
      }
    });
    gridRow.appendChild(this._gridCb);
    const gridLabel = document.createElement('span');
    gridLabel.textContent = 'Grid';
    gridLabel.style.cssText = 'color:#CED9E0;font-size:12px;';
    gridRow.appendChild(gridLabel);
    this._settingsPopover.appendChild(gridRow);

    this._overlay.appendChild(this._settingsPopover);

    // Position settings popover near bottom of strip
    this._updateSettingsPopoverPos = () => {
      if (this._disposed) return;
      const sr = strip.getBoundingClientRect();
      const cr = this._container.getBoundingClientRect();
      this._settingsPopover.style.left = `${sr.right - cr.left + 6}px`;
      this._settingsPopover.style.top = `${sr.bottom - cr.top - 40}px`;
      this._settingsPopover.style.transform = 'none';
    };
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
    this._updateSecPopoverPos = update;
  }

  _addToolTooltip(btn, text) {
    const tip = document.createElement('div');
    tip.style.cssText =
      'position:absolute;left:calc(100% + 8px);top:50%;transform:translateY(-50%);pointer-events:none;background:rgba(28,33,39,0.9);color:#E8EDF0;font:500 12px system-ui,-apple-system,sans-serif;padding:4px 8px;border-radius:4px;white-space:nowrap;display:none;z-index:60;';
    tip.textContent = text;
    btn.appendChild(tip);
    btn.addEventListener('mouseenter', () => {
      tip.style.display = '';
    });
    btn.addEventListener('mouseleave', () => {
      tip.style.display = 'none';
    });
  }

  _setActiveTool(tool) {
    this._activeTool = tool;
    const activeCSS = 'background:rgba(19,124,189,0.3);color:#2B95D6;';
    const inactiveCSS = 'background:transparent;color:#CED9E0;';

    // Update button styles (preserve base styles)
    const base =
      'width:36px;height:36px;border:none;border-radius:6px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:background 0.12s,color 0.12s;position:relative;';
    this._selectBtn.style.cssText = base + (tool === null ? activeCSS : inactiveCSS);
    this._measBtn.style.cssText = base + (tool === 'measure' ? activeCSS : inactiveCSS);
    this._secBtn.style.cssText = base + (tool === 'section' ? activeCSS : inactiveCSS);
  }

  _initKeys() {
    this._onKey = (e) => {
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
      if (e.ctrlKey || e.metaKey) return;
      if (KEY_TO_PRESET[e.key]) {
        e.preventDefault();
        this.setViewPreset(KEY_TO_PRESET[e.key]);
      } else if (e.key === 'f') {
        e.preventDefault();
        this.zoomToFit();
      } else if (e.key === 'm') {
        e.preventDefault();
        this._measBtn.click();
      } else if (e.key === 's') {
        e.preventDefault();
        this._secBtn.click();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        if (this._meas.enabled) this.disableMeasureTool();
        else if (this._meas.measurements && this._meas.measurements.length > 0) {
          // Escape clears measurements even when tool is not active
          this._meas.clearAll();
        }
        if (this._secOn) this.disableSectionPlane();
        if (this._settingsPopover && this._settingsPopover.style.display !== 'none') {
          this._settingsPopover.style.display = 'none';
        }
        this._setActiveTool(null);
      }
    };
    window.addEventListener('keydown', this._onKey);
  }

  _hiliteAxis() {
    for (const [a, b] of Object.entries(this._secAxisBtns) as [string, any][]) {
      b.style.background = a === this._secAxis ? 'rgba(19,124,189,0.3)' : 'transparent';
      b.style.color = a === this._secAxis ? '#2B95D6' : '#CED9E0';
      b.style.borderColor = a === this._secAxis ? 'rgba(43,149,214,0.4)' : 'rgba(206,217,224,0.2)';
    }
  }

  // ── Content bounding box ──
  _bbox() {
    this._scene.updateMatrixWorld(true);
    const box = new THREE.Box3();
    let has = false;
    // Scan named groups
    for (const g of Object.values(this._groups) as any[]) {
      if (!g.visible) continue;
      g.traverse((ch: any) => {
        if (ch.isMesh && ch.geometry) {
          ch.geometry.computeBoundingBox();
          const b = ch.geometry.boundingBox.clone();
          b.applyMatrix4(ch.matrixWorld);
          box.union(b);
          has = true;
        }
      });
    }
    // Also scan direct scene children that aren't named groups or viewport internals
    this._scene.children.forEach((child) => {
      if (
        child.isGroup &&
        !this._groups[child.name] &&
        child.visible &&
        !child.userData._vpSec &&
        !child.userData._vpCap &&
        child.name !== 'section-caps'
      ) {
        child.traverse((ch) => {
          if (ch.isMesh && ch.geometry) {
            ch.geometry.computeBoundingBox();
            const b = ch.geometry.boundingBox.clone();
            b.applyMatrix4(ch.matrixWorld);
            box.union(b);
            has = true;
          }
        });
      }
    });
    return has ? box : null;
  }

  // ── Camera lerp ──
  _startLerp(pos, tgt, up) {
    this._lerpS.pos.copy(this._cam.position);
    this._lerpS.tgt.copy(this._ctrl.target);
    this._lerpS.up.copy(this._cam.up);
    this._lerpE.pos.copy(pos);
    this._lerpE.tgt.copy(tgt);
    this._lerpE.up.copy(up);
    this._lerpT0 = performance.now();
    this._lerpActive = true;
  }
  _tickLerp() {
    if (!this._lerpActive) return;
    let t = Math.min((performance.now() - this._lerpT0) / this._lerpDur, 1);
    t = 1 - (1 - t) ** 3; // ease-out cubic
    this._cam.position.lerpVectors(this._lerpS.pos, this._lerpE.pos, t);
    this._ctrl.target.lerpVectors(this._lerpS.tgt, this._lerpE.tgt, t);
    this._cam.up.lerpVectors(this._lerpS.up, this._lerpE.up, t).normalize();
    this._cam.lookAt(this._ctrl.target);
    this._ctrl.update();
    if (t >= 1) this._lerpActive = false;
  }

  // ── Section plane with geometry caps and contour lines ──
  _applySection() {
    const box = this._bbox();
    if (box) {
      const ai = AXIS_IDX[this._secAxis];
      const sign = this._secFlipped ? 1 : -1;
      const pos = box.min.getComponent(ai) + (box.max.getComponent(ai) - box.min.getComponent(ai)) * this._secFrac;
      const n = new THREE.Vector3();
      n.setComponent(ai, sign);
      this._secPlane.normal.copy(n);
      this._secPlane.constant = -sign * pos;
      this._showSecViz(box, ai, pos);
    }
    const clips = [this._secPlane];
    this._scene.traverse((ch) => {
      if (ch.material && !ch.userData._vpSec && !ch.userData._vpCap) {
        if (ch.isLineSegments && !ch.material._vpClip) {
          ch.material = ch.material.clone();
          ch.material._vpClip = true;
        }
        ch.material.clippingPlanes = clips;
        ch.material.clipShadows = true;
        // Polygon offset on solid meshes to prevent z-fighting with contour lines
        if (ch.isMesh && !ch.isLineSegments) {
          ch.material.polygonOffset = true;
          ch.material.polygonOffsetFactor = 1;
          ch.material.polygonOffsetUnits = 1;
        }
      }
    });

    // Rebuild geometry caps and contour lines
    this._rebuildSectionCaps(clips);
  }

  _showSecViz(box, ai, pos) {
    if (!this._secViz) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0x2b95d6,
        transparent: true,
        opacity: 0.06,
        side: THREE.DoubleSide,
        depthWrite: false,
        depthTest: true,
      });
      this._secViz = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat);
      this._secViz.renderOrder = 9999; // render after all body geometry
      this._secViz.userData._vpSec = true;
      this._secViz.raycast = () => {};
      this._scene.add(this._secViz);
    }
    const c = box.getCenter(new THREE.Vector3()),
      sz = box.getSize(new THREE.Vector3());
    this._secViz.scale.set(Math.max(sz.x, sz.y, sz.z) * 1.5, Math.max(sz.x, sz.y, sz.z) * 1.5, 1);
    this._secViz.visible = true;
    c.setComponent(ai, pos);
    this._secViz.position.copy(c);
    this._secViz.rotation.set(0, 0, 0);
    if (ai === 0) this._secViz.rotation.y = Math.PI / 2;
    else if (ai === 1) this._secViz.rotation.x = Math.PI / 2;
  }

  /**
   * Geometry-based section caps with contour lines.
   *
   * For each visible group, computes the mesh-plane intersection contour,
   * chains segments into closed polygons, triangulates them, and creates
   * opaque cap meshes with hatch texture. No stencil buffer needed.
   */
  _rebuildSectionCaps(clips) {
    this._clearSectionCaps();
    if (!clips.length) return;

    this._capGroup = new THREE.Group();
    this._capGroup.name = 'section-caps';
    this._capGroup.userData._vpCap = true;
    this._scene.add(this._capGroup);

    // Collect all visible meshes grouped by layer
    const allMeshes = [];
    const layerMeshes = {};

    // Collect meshes from named groups AND from ungrouped scene children
    const sources = [];
    for (const [groupName, group] of Object.entries(this._groups) as [string, any][]) {
      if (!group.visible) continue;
      sources.push({ name: groupName, node: group });
    }
    // Also scan direct scene children that aren't groups or viewport internals
    this._scene.children.forEach((child) => {
      if (
        child.isGroup &&
        !this._groups[child.name] &&
        child.visible &&
        !child.userData._vpSec &&
        !child.userData._vpCap &&
        child.name !== 'section-caps'
      ) {
        sources.push({ name: child.name || 'scene', node: child });
      }
    });

    for (const { name: groupName, node } of sources) {
      node.traverse((child) => {
        if (child.isMesh && child.geometry && !child.userData._vpSec && !child.userData._vpCap) {
          allMeshes.push({ mesh: child, groupName });
          if (!layerMeshes[groupName]) layerMeshes[groupName] = [];
          layerMeshes[groupName].push(child);
        }
      });
    }

    const plane = this._secPlane;

    // DEBUG: collect ALL segments across all meshes, log counts
    const allSegments = [];
    for (const { mesh } of allMeshes) {
      this._computeMeshPlaneContour(mesh, plane, allSegments);
    }

    // Chain into closed polygons
    const polygons = this._chainSegments(allSegments);

    // Triangulate and create hatched cap meshes
    if (polygons.length > 0) {
      const capGeom = this._triangulateCapsOnPlane(polygons, plane);
      if (capGeom) {
        // Hide the section viz plane — caps show the cut position
        if (this._secViz) this._secViz.visible = false;

        // Cap color — darker than the body for clear contrast
        let capColor;
        if (this._sectionCapColorFn) {
          const firstGroupName = Object.keys(layerMeshes)[0];
          capColor = this._sectionCapColorFn(firstGroupName);
        }
        if (capColor == null) {
          const firstMat = allMeshes[0]?.mesh?.material;
          const baseHex = firstMat?.color ? firstMat.color.getHex() : 0xced9e0;
          // Darken the body color (blend toward dark gray, not white)
          const base = new THREE.Color(baseHex);
          capColor = base.clone().lerp(new THREE.Color(0x394b59), 0.5);
        }

        const hatchTex = this._createHatchTexture(capColor);
        const capMat = new THREE.MeshBasicMaterial({
          map: hatchTex,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        });
        const capMesh = new THREE.Mesh(capGeom, capMat);
        capMesh.raycast = () => {};
        capMesh.userData._vpCap = true;
        this._capGroup.add(capMesh);

        // Track hatch textures for zoom-dependent repeat update
        if (!this._hatchTextures) this._hatchTextures = [];
        this._hatchTextures.push(hatchTex);
      }
    }

    // Contour lines (keep for now)
    if (allSegments.length > 0) {
      const lineGeom = new LineSegmentsGeometry();
      lineGeom.setPositions(allSegments);
      const w = this._ren.domElement.width,
        h = this._ren.domElement.height;
      const lineMat = new LineMaterial({
        color: BP.DARK_GRAY3,
        linewidth: 3,
        resolution: new THREE.Vector2(w, h),
        clippingPlanes: clips,
      });
      const contourLines = new LineSegments2(lineGeom, lineMat);
      contourLines.renderOrder = RENDER_ORDER.SECTION_CONTOUR;
      contourLines.raycast = () => {};
      contourLines.userData._vpCap = true;
      // Offset contour lines slightly toward the camera along the section
      // plane normal so they sit in front of both the cap and clipped body
      const offset = this._secPlane.normal.clone().multiplyScalar(0.0001);
      contourLines.position.copy(offset);
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
    const a = new THREE.Vector3(),
      b = new THREE.Vector3(),
      c = new THREE.Vector3();

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

      const crossings = [];
      if (da * db < 0) crossings.push(this._planeEdgeIntersect(a, b, da, db));
      if (db * dc < 0) crossings.push(this._planeEdgeIntersect(b, c, db, dc));
      if (dc * da < 0) crossings.push(this._planeEdgeIntersect(c, a, dc, da));

      if (crossings.length === 2) {
        out.push(crossings[0].x, crossings[0].y, crossings[0].z, crossings[1].x, crossings[1].y, crossings[1].z);
      }
    }
  }

  _planeEdgeIntersect(p1, p2, d1, d2) {
    const t = d1 / (d1 - d2);
    return new THREE.Vector3().lerpVectors(p1, p2, t);
  }

  /**
   * Chain flat contour segments into closed polygons.
   * @param {number[]} flatSegments — [x0,y0,z0, x1,y1,z1, ...] pairs of endpoints
   * @returns {THREE.Vector3[][]} — array of closed polygon vertex loops
   */
  _chainSegments(flatSegments) {
    const segs = [];
    for (let i = 0; i < flatSegments.length; i += 6) {
      segs.push([
        new THREE.Vector3(flatSegments[i], flatSegments[i + 1], flatSegments[i + 2]),
        new THREE.Vector3(flatSegments[i + 3], flatSegments[i + 4], flatSegments[i + 5]),
      ]);
    }

    const EPS = 1e-6;
    const used = new Set();
    const polygons = [];

    for (let startIdx = 0; startIdx < segs.length; startIdx++) {
      if (used.has(startIdx)) continue;

      const chain = [segs[startIdx][0], segs[startIdx][1]];
      used.add(startIdx);

      // Extend the chain by finding segments that connect at the tail
      let changed = true;
      while (changed) {
        changed = false;
        const tail = chain[chain.length - 1];

        for (let i = 0; i < segs.length; i++) {
          if (used.has(i)) continue;
          const [a, b] = segs[i];

          if (tail.distanceTo(a) < EPS) {
            chain.push(b);
            used.add(i);
            changed = true;
            break;
          } else if (tail.distanceTo(b) < EPS) {
            chain.push(a);
            used.add(i);
            changed = true;
            break;
          }
        }
      }

      // Check if closed (head meets tail)
      if (chain.length >= 3 && chain[0].distanceTo(chain[chain.length - 1]) < EPS) {
        chain.pop(); // remove duplicate closing point
        polygons.push(chain);
      }
    }

    return polygons;
  }

  /**
   * Triangulate closed polygons that lie on a cutting plane.
   * Projects to 2D, triangulates with earcut, maps back to 3D.
   * @param {THREE.Vector3[][]} polygons
   * @param {THREE.Plane} plane
   * @returns {THREE.BufferGeometry|null}
   */
  _triangulateCapsOnPlane(polygons, plane) {
    // Build a local 2D coordinate system on the plane
    const normal = plane.normal.clone().normalize();

    let tangent = new THREE.Vector3(1, 0, 0);
    if (Math.abs(normal.dot(tangent)) > 0.9) tangent = new THREE.Vector3(0, 1, 0);

    const u = new THREE.Vector3().crossVectors(normal, tangent).normalize();
    const v = new THREE.Vector3().crossVectors(normal, u).normalize();

    // A point on the plane
    const planePoint = normal.clone().multiplyScalar(-plane.constant);

    // Project all polygons to 2D, deduplicate near-coincident vertices,
    // compute signed area to classify outer vs holes
    const DEDUP_EPS = 1e-7;
    const projected = polygons
      .map((polygon) => {
        const pts2d = polygon.map((p) => {
          const rel = p.clone().sub(planePoint);
          return [rel.dot(u), rel.dot(v)];
        });
        // Remove near-duplicate consecutive vertices
        const cleaned2d = [pts2d[0]];
        const cleaned3d = [polygon[0]];
        for (let i = 1; i < pts2d.length; i++) {
          const prev = cleaned2d[cleaned2d.length - 1];
          const dx = pts2d[i][0] - prev[0],
            dy = pts2d[i][1] - prev[1];
          if (dx * dx + dy * dy > DEDUP_EPS * DEDUP_EPS) {
            cleaned2d.push(pts2d[i]);
            cleaned3d.push(polygon[i]);
          }
        }
        if (cleaned2d.length < 3) return null;

        // Signed area (shoelace formula)
        let area = 0;
        for (let i = 0; i < cleaned2d.length; i++) {
          const j = (i + 1) % cleaned2d.length;
          area += cleaned2d[i][0] * cleaned2d[j][1];
          area -= cleaned2d[j][0] * cleaned2d[i][1];
        }
        area /= 2;
        return { polygon: cleaned3d, pts2d: cleaned2d, area };
      })
      .filter(Boolean);

    if (projected.length === 0) return null;

    // Sort by absolute area descending
    projected.sort((a, b) => Math.abs(b.area) - Math.abs(a.area));

    // Build containment tree: each polygon is either a top-level outer,
    // a hole inside an outer, or an island inside a hole.
    // We use a simple approach: for each polygon (sorted large→small),
    // count how many larger polygons contain it.
    // Even nesting depth (0, 2, 4...) = outer. Odd (1, 3...) = hole.
    const roles = projected.map((poly, i) => {
      let depth = 0;
      for (let j = 0; j < i; j++) {
        // only check larger polygons
        if (this._pointInPolygon2D(poly.pts2d[0], projected[j].pts2d)) {
          depth++;
        }
      }
      return { ...poly, depth, isOuter: depth % 2 === 0 };
    });

    // Group: each outer gets the holes at depth = outer.depth + 1 that are inside it
    const outers = roles.filter((r) => r.isOuter);
    const allHoles = roles.filter((r) => !r.isOuter);

    // Triangulate each outer with its holes
    const positions = [];
    const indices = [];
    let vertexOffset = 0;

    for (const outer of outers) {
      // Ensure CCW winding for outer
      if (outer.area < 0) {
        outer.pts2d.reverse();
        outer.polygon.reverse();
      }

      // Find holes directly inside this outer (depth == outer.depth + 1, contained by this outer)
      const myHoles = allHoles.filter(
        (h) => h.depth === outer.depth + 1 && this._pointInPolygon2D(h.pts2d[0], outer.pts2d),
      );

      // Ensure CW winding for holes
      for (const h of myHoles) {
        if (h.area > 0) {
          h.pts2d.reverse();
          h.polygon.reverse();
        }
      }

      // Build earcut input
      const flatCoords = [];
      for (const [x, y] of outer.pts2d) flatCoords.push(x, y);

      const holeIndices = [];
      for (const hole of myHoles) {
        holeIndices.push(flatCoords.length / 2);
        for (const [x, y] of hole.pts2d) flatCoords.push(x, y);
      }

      const triIndices = Earcut.triangulate(flatCoords, holeIndices, 2);

      // Add 3D vertices (outer + holes, same order as flatCoords)
      for (const p of outer.polygon) positions.push(p.x, p.y, p.z);
      for (const hole of myHoles) {
        for (const p of hole.polygon) positions.push(p.x, p.y, p.z);
      }

      // Add indices
      for (const idx of triIndices) indices.push(idx + vertexOffset);

      vertexOffset += outer.polygon.length;
      for (const hole of myHoles) vertexOffset += hole.polygon.length;
    }

    if (positions.length === 0) return null;

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();

    // UVs in world-space (meters). texture.repeat is updated each frame
    // based on camera zoom so hatching has consistent screen-space density.
    const uvs = [];
    for (let i = 0; i < positions.length; i += 3) {
      const p = new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]);
      const rel = p.clone().sub(planePoint);
      uvs.push(rel.dot(u), rel.dot(v));
    }
    geom.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));

    return geom;
  }

  /** Point-in-polygon test (2D, ray casting). */
  _pointInPolygon2D(point, polygon) {
    const [px, py] = point;
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i];
      const [xj, yj] = polygon[j];
      if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
    return inside;
  }

  /**
   * Create a repeating diagonal-line hatch texture for section caps.
   * @param {number|THREE.Color} color — hatch line color (hex number or THREE.Color)
   * @param {number} [lineWidth=2] — stroke width of diagonal lines
   * @param {number} [spacing=8] — pixel spacing between lines
   * @returns {THREE.CanvasTexture}
   */
  _createHatchTexture(color, lineWidth = 1.5, spacing = 8) {
    // Tile size must be a multiple of spacing for seamless tiling
    const size = spacing * 8; // e.g., 64 for spacing=8
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // Solid fill in the body color (darker tint)
    const c = color instanceof THREE.Color ? color : new THREE.Color(color);
    const cssColor = `#${c.getHexString()}`;
    ctx.fillStyle = cssColor;
    ctx.fillRect(0, 0, size, size);

    // Diagonal hatch lines — lighter for contrast against the dark fill
    const lighter = c.clone().lerp(new THREE.Color(0xffffff), 0.65);
    ctx.strokeStyle = `#${lighter.getHexString()}`;
    ctx.lineWidth = lineWidth;

    // Draw 45-degree lines that tile seamlessly.
    // For a 45-degree line, a line entering at (0, y) exits at (size, y+size).
    // To tile: draw lines at offsets 0, spacing, 2*spacing, ...
    // Each line wraps around the tile boundaries.
    ctx.beginPath();
    for (let offset = 0; offset < size; offset += spacing) {
      // Line from bottom-left to top-right, offset by `offset`
      ctx.moveTo(offset, 0);
      ctx.lineTo(offset + size, size);
      // Wrap: the portion that exits the right side re-enters from the left
      ctx.moveTo(offset - size, 0);
      ctx.lineTo(offset, size);
    }
    ctx.stroke();

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(4, 4);
    return texture;
  }

  /** Update hatch texture repeat so pattern has ~constant screen-space density. */
  _updateHatchRepeat() {
    if (!this._hatchTextures || this._hatchTextures.length === 0) return;

    // Compute world-units-per-pixel from camera
    let worldPerPx;
    if (this._cameraType === 'orthographic') {
      // Ortho: frustum height / viewport height
      worldPerPx = (this._cam.top - this._cam.bottom) / this._ren.domElement.clientHeight;
    } else {
      // Perspective: approximate from distance to target
      const dist = this._cam.position.distanceTo(this._ctrl.target);
      const vFov = (this._cam.fov * Math.PI) / 180;
      worldPerPx = (2 * dist * Math.tan(vFov / 2)) / this._ren.domElement.clientHeight;
    }

    // We want hatch lines spaced ~8 pixels apart on screen.
    // The texture is 64px with lines every `spacing` (6px in texture space).
    // texture.repeat = N means the texture tiles N times per 1 world unit.
    // So line spacing in world = 1 / (N * linesPerTile)
    // We want: lineSpacingWorld = 8 * worldPerPx
    // → 1 / (N * linesPerTile) = 8 * worldPerPx
    // → N = 1 / (8 * worldPerPx * linesPerTile)
    const SCREEN_SPACING = 8; // pixels between hatch lines on screen
    const LINES_PER_TILE = 8; // 8 lines per tile (spacing=8, size=64)
    const repeat = 1 / (SCREEN_SPACING * worldPerPx * LINES_PER_TILE);

    for (const tex of this._hatchTextures) {
      tex.repeat.set(repeat, repeat);
    }
  }

  _clearSectionCaps() {
    if (this._capGroup) {
      this._capGroup.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (child.material.map) child.material.map.dispose();
          child.material.dispose();
        }
      });
      this._scene.remove(this._capGroup);
      this._capGroup = null;
    }
    this._contourLineMat = null;
    this._hatchTextures = null;
  }

  // ── Render loop ──
  _tick() {
    if (this._disposed) return;
    requestAnimationFrame(() => this._tick());
    this._tickLerp();
    this._ctrl.update();
    // Follow mode drift detection — exit if user pans/zooms significantly
    if (this._followMode && this._followTarget && !this._lerpActive) {
      const drift = this._ctrl.target.distanceTo(this._followTarget);
      const threshold = this._followRadius > 0 ? this._followRadius * 0.2 : 0.001;
      if (drift > threshold) {
        this.setFollowMode(false);
      }
    }
    if (this._animCb) this._animCb();
    // Update hatch texture repeat based on zoom so hatching has
    // consistent screen-space density regardless of camera distance
    this._updateHatchRepeat();
    // Edge detection works with section caps (no stencil dependency)
    if (this._edgeC) {
      this._edgeC.render();
    } else {
      this._ren.render(this._scene, this._cam);
    }
    if (this._meas.enabled || this._meas.measurements.length > 0) this._meas.update();
    // Sync orientation cube
    this._syncOrientationCube();
    // Keep section popover positioned
    if (this._secOn && this._updateSecPopoverPos) this._updateSecPopoverPos();
    if (this._settingsPopover && this._settingsPopover.style.display !== 'none' && this._updateSettingsPopoverPos)
      this._updateSettingsPopoverPos();
  }
}
