/**
 * SectionCutter — reusable section plane with stencil-based cross-section caps.
 *
 * Provides section plane state, UI wiring, clipping, stencil caps with
 * per-mesh cap colors, section plane visualization, and contour lines.
 *
 * Usage:
 *   const cutter = new SectionCutter(scene, renderer);
 *   cutter.setMeshProvider(() => collectVisibleMeshes());
 *   cutter.setCapColorFn((mesh) => mesh.material.color.clone());
 *   cutter.bindUI({ toggle, controls, axisButtons, slider, flipBtn, keyTarget });
 */

import * as THREE from 'three';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { BP, RENDER_ORDER, SECTION_STENCIL_BASE, SECTION_STENCIL_STRIDE, tintColor } from './presentation.ts';

const AXIS_INDEX = { x: 0, y: 1, z: 2 };

export class SectionCutter {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  enabled: boolean;
  axis: string;
  flipped: boolean;
  fraction: number;
  plane: THREE.Plane;
  _meshProvider: () => any[];
  _capColorFn: ((mesh: any) => any) | null;
  _capGroup: THREE.Group | null;
  _sectionViz: THREE.Mesh | null;
  _sectionRing: THREE.LineSegments | null;
  _contourLineMat: any;
  _ui: any;

  constructor(scene: THREE.Scene, renderer: THREE.WebGLRenderer) {
    this.scene = scene;
    this.renderer = renderer;

    // Section plane state
    this.enabled = false;
    this.axis = 'y'; // default: horizontal cut (Y-up scenes)
    this.flipped = false;
    this.fraction = 0.5; // slider position (0–1 along bbox)
    this.plane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);

    // Callbacks
    this._meshProvider = () => [];
    this._capColorFn = null; // (mesh) => THREE.Color

    // Internal state
    this._capGroup = null;
    this._sectionViz = null;
    this._sectionRing = null;
    this._contourLineMat = null;
  }

  /**
   * Set a function that returns the meshes to section.
   * @param {() => THREE.Mesh[]} fn
   */
  setMeshProvider(fn) {
    this._meshProvider = fn;
  }

  /**
   * Set a function that returns the cap color for a given mesh.
   * If not set, caps use a default gray.
   * @param {(mesh: THREE.Mesh) => THREE.Color} fn
   */
  setCapColorFn(fn) {
    this._capColorFn = fn;
  }

  /**
   * Bind to UI elements for section controls.
   *
   * @param {Object} ui
   * @param {HTMLElement} ui.toggle — Section on/off button
   * @param {HTMLElement} ui.controls — container to show/hide
   * @param {NodeList|HTMLElement[]} ui.axisButtons — buttons with data-section-axis
   * @param {HTMLInputElement} ui.slider — range input (0–100)
   * @param {HTMLElement} ui.flipBtn — flip direction button
   * @param {HTMLElement} [ui.keyTarget] — element to listen for 'S' key (default: window)
   */
  bindUI(ui) {
    this._ui = ui;

    if (ui.toggle) {
      ui.toggle.addEventListener('click', () => this.toggle());
    }

    if (ui.axisButtons) {
      for (const btn of ui.axisButtons) {
        btn.addEventListener('click', () => {
          this.axis = btn.dataset.sectionAxis;
          for (const b of ui.axisButtons) {
            b.classList.toggle('active', b.dataset.sectionAxis === this.axis);
          }
          this._update();
        });
      }
    }

    if (ui.slider) {
      ui.slider.addEventListener('input', () => {
        this.fraction = parseFloat(ui.slider.value) / 100;
        this._update();
      });
    }

    if (ui.flipBtn) {
      ui.flipBtn.addEventListener('click', () => {
        this.flipped = !this.flipped;
        ui.flipBtn.classList.toggle('active', this.flipped);
        this._update();
      });
    }

    const keyTarget = ui.keyTarget || window;
    keyTarget.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 's' || e.key === 'S') {
        this.toggle();
      }
    });
  }

  /** Toggle section on/off. */
  toggle() {
    this.enabled = !this.enabled;
    if (this._ui) {
      if (this._ui.toggle) this._ui.toggle.classList.toggle('active', this.enabled);
      if (this._ui.controls) this._ui.controls.style.display = this.enabled ? 'flex' : 'none';
    }
    this._update();
  }

  /** Force a rebuild (call after scene changes). */
  rebuild() {
    if (this.enabled) this._update();
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _update() {
    const meshes = this._meshProvider();
    const box = this._computeBBox(meshes);
    const clips = this.enabled ? [this.plane] : [];

    if (this.enabled && box) {
      const axisIdx = AXIS_INDEX[this.axis];
      const axisMin = box.min.getComponent(axisIdx);
      const axisMax = box.max.getComponent(axisIdx);
      const pos = axisMin + (axisMax - axisMin) * this.fraction;

      const sign = this.flipped ? 1 : -1;
      const normal = new THREE.Vector3();
      normal.setComponent(axisIdx, sign);
      this.plane.normal.copy(normal);
      this.plane.constant = pos * -sign;

      this._updateViz(box, axisIdx, pos);
    } else {
      this._removeViz();
    }

    // Apply clipping to all meshes
    for (const mesh of meshes) {
      if (mesh.material) {
        // Clone shared materials on first clip to avoid mutating shared instances
        if (!mesh.material._sectionClippable) {
          mesh.material = mesh.material.clone();
          mesh.material._sectionClippable = true;
        }
        mesh.material.clippingPlanes = clips;
        mesh.material.clipShadows = true;
      }
    }

    // Also clip edge lines (LineSegments children alongside meshes)
    const visited = new Set();
    for (const mesh of meshes) {
      const parent = mesh.parent;
      if (!parent || visited.has(parent.uuid)) continue;
      visited.add(parent.uuid);
      parent.traverse((child) => {
        if (child.isLineSegments && child.material) {
          if (!child.material._sectionClippable) {
            child.material = child.material.clone();
            child.material._sectionClippable = true;
          }
          child.material.clippingPlanes = clips;
          child.material.clipShadows = true;
        }
      });
    }

    this._rebuildCaps(meshes, clips);
  }

  _computeBBox(meshes) {
    const box = new THREE.Box3();
    let any = false;
    for (const mesh of meshes) {
      if (!mesh.geometry) continue;
      mesh.geometry.computeBoundingBox();
      const childBox = mesh.geometry.boundingBox.clone();
      childBox.applyMatrix4(mesh.matrixWorld);
      box.union(childBox);
      any = true;
    }
    return any ? box : null;
  }

  // ---------------------------------------------------------------------------
  // Section plane visualization (translucent quad)
  // ---------------------------------------------------------------------------

  _updateViz(box, axisIdx, pos) {
    if (!this._sectionViz) {
      const geom = new THREE.PlaneGeometry(1, 1);
      const mat = new THREE.MeshBasicMaterial({
        color: 0x2b95d6,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      this._sectionViz = new THREE.Mesh(geom, mat);
      this._sectionViz.renderOrder = RENDER_ORDER.SECTION_VIZ;
      this._sectionViz.raycast = () => {};
      this.scene.add(this._sectionViz);

      const ringGeom = new THREE.EdgesGeometry(geom);
      this._sectionRing = new THREE.LineSegments(
        ringGeom,
        new THREE.LineBasicMaterial({
          color: 0x2b95d6,
          transparent: true,
          opacity: 0.4,
        }),
      );
      this._sectionRing.raycast = () => {};
      this._sectionViz.add(this._sectionRing);
    }

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const planeSize = Math.max(size.x, size.y, size.z) * 1.5;

    this._sectionViz.scale.set(planeSize, planeSize, 1);
    this._sectionViz.visible = true;

    center.setComponent(axisIdx, pos);
    this._sectionViz.position.copy(center);

    this._sectionViz.rotation.set(0, 0, 0);
    if (axisIdx === 0)
      this._sectionViz.rotation.y = Math.PI / 2; // X
    else if (axisIdx === 1) this._sectionViz.rotation.x = Math.PI / 2; // Y
    // Z = default orientation (no rotation needed)
  }

  _removeViz() {
    if (this._sectionViz) {
      this._sectionViz.visible = false;
    }
    this._clearCaps();
  }

  // ---------------------------------------------------------------------------
  // Stencil-based section caps
  // ---------------------------------------------------------------------------

  _rebuildCaps(meshes, clips) {
    this._clearCaps();
    if (!clips.length || meshes.length === 0) return;

    this._capGroup = new THREE.Group();
    this._capGroup.name = 'section-caps';
    this.scene.add(this._capGroup);

    const box = this._computeBBox(meshes);
    const capSize = box ? Math.max(...box.getSize(new THREE.Vector3()).toArray()) * 1.5 : 0.2;

    // Group meshes by body (parent group) to give each body its own stencil ref
    const bodyGroups = new Map(); // parent uuid → { meshes, color }
    for (const mesh of meshes) {
      const key = mesh.parent ? mesh.parent.uuid : 'root';
      if (!bodyGroups.has(key)) {
        bodyGroups.set(key, { meshes: [], color: null });
      }
      const group = bodyGroups.get(key);
      group.meshes.push(mesh);

      // Determine cap color: use callback if provided, else derive from mesh material
      if (!group.color && this._capColorFn) {
        group.color = this._capColorFn(mesh);
      }
      if (!group.color && mesh.material && mesh.material.color) {
        group.color = tintColor(mesh.material.color, 0.6);
      }
    }

    let layerIndex = 0;
    const allMeshes = [];

    for (const [, bodyGroup] of bodyGroups) {
      const ref = layerIndex + 1;
      const orderBase = SECTION_STENCIL_BASE + layerIndex * SECTION_STENCIL_STRIDE;
      layerIndex++;

      for (const mesh of bodyGroup.meshes) {
        this._addStencilHelper(mesh, THREE.BackSide, ref, clips, orderBase + RENDER_ORDER.STENCIL_BACK);
        this._addStencilHelper(mesh, THREE.FrontSide, 0, clips, orderBase + RENDER_ORDER.STENCIL_FRONT);
      }

      const capColor = bodyGroup.color || new THREE.Color(0x8a9ba8);

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
        stencilZPass: THREE.ZeroStencilOp,
      });
      const capPlane = new THREE.Mesh(capGeom, capMat);
      capPlane.raycast = () => {};
      capPlane.renderOrder = orderBase + RENDER_ORDER.STENCIL_CAP;

      if (this._sectionViz) {
        capPlane.position.copy(this._sectionViz.position);
        capPlane.rotation.copy(this._sectionViz.rotation);
      }

      this._capGroup.add(capPlane);
      allMeshes.push(...bodyGroup.meshes);
    }

    // Contour lines
    const contourSegments = [];
    for (const mesh of allMeshes) {
      this._computeContour(mesh, this.plane, contourSegments);
    }
    if (contourSegments.length > 0) {
      const lineGeom = new LineSegmentsGeometry();
      lineGeom.setPositions(contourSegments);
      const lineMat = new LineMaterial({
        color: BP.DARK_GRAY3,
        linewidth: 3,
        resolution: new THREE.Vector2(this.renderer.domElement.width, this.renderer.domElement.height),
        clippingPlanes: clips,
      });
      const contourLines = new LineSegments2(lineGeom, lineMat);
      contourLines.renderOrder = RENDER_ORDER.SECTION_CONTOUR;
      contourLines.raycast = () => {};
      this._capGroup.add(contourLines);
      this._contourLineMat = lineMat;
    }
  }

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

    // Place the stencil helper at the source mesh's world position.
    // Since _capGroup is a direct child of the scene, local = world.
    sourceMesh.updateWorldMatrix(true, false);
    mesh.applyMatrix4(sourceMesh.matrixWorld);

    mesh.renderOrder = renderOrder;
    mesh.raycast = () => {};
    this._capGroup.add(mesh);
  }

  _clearCaps() {
    if (this._capGroup) {
      this._capGroup.traverse((child: any) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      this.scene.remove(this._capGroup);
      this._capGroup = null;
    }
    this._contourLineMat = null;
  }

  // ---------------------------------------------------------------------------
  // Contour line computation
  // ---------------------------------------------------------------------------

  _computeContour(mesh, plane, out) {
    const geom = mesh.geometry;
    const posAttr = geom.getAttribute('position');
    if (!posAttr) return;

    const index = geom.index;
    mesh.updateWorldMatrix(true, false);
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
      if (da * db < 0) crossings.push(this._edgeIntersect(a, b, da, db));
      if (db * dc < 0) crossings.push(this._edgeIntersect(b, c, db, dc));
      if (dc * da < 0) crossings.push(this._edgeIntersect(c, a, dc, da));

      if (crossings.length === 2) {
        out.push(crossings[0].x, crossings[0].y, crossings[0].z, crossings[1].x, crossings[1].y, crossings[1].z);
      }
    }
  }

  _edgeIntersect(p1, p2, d1, d2) {
    const t = d1 / (d1 - d2);
    return new THREE.Vector3().lerpVectors(p1, p2, t);
  }

  /** Clean up all GPU resources. */
  dispose() {
    this._removeViz();
    if (this._sectionViz) {
      this._sectionViz.geometry.dispose();
      (this._sectionViz.material as THREE.Material).dispose();
      if (this._sectionRing) {
        this._sectionRing.geometry.dispose();
        (this._sectionRing.material as THREE.Material).dispose();
      }
      this.scene.remove(this._sectionViz);
      this._sectionViz = null;
    }
  }
}
