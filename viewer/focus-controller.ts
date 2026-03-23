/**
 * Focus Controller — camera animation + mesh ghosting for explore mode.
 *
 * Smoothly animates the camera to frame a selected body, and dims
 * all other meshes to draw attention to the focused one.
 */

import * as THREE from 'three';

const _box = new THREE.Box3();
const _center = new THREE.Vector3();
const _size = new THREE.Vector3();
const _camDir = new THREE.Vector3();

export class FocusController {
  ctx: any;
  ghosted: boolean;
  _savedMaterials: Map<any, any>;
  _animating: boolean;
  _animStart: number;
  _animDuration: number;
  _startPos: THREE.Vector3;
  _endPos: THREE.Vector3;
  _startTarget: THREE.Vector3;
  _endTarget: THREE.Vector3;

  constructor(ctx: any) {
    this.ctx = ctx;
    this.ghosted = false;
    this._savedMaterials = new Map(); // mesh → { opacity, transparent, emissive }
    this._animating = false;
    this._animStart = 0;
    this._animDuration = 0;
    this._startPos = new THREE.Vector3();
    this._endPos = new THREE.Vector3();
    this._startTarget = new THREE.Vector3();
    this._endTarget = new THREE.Vector3();
  }

  /**
   * Compute world-space bounding box for a body's meshes.
   * @param {number} bodyId
   * @returns {THREE.Box3}
   */
  getBodyBoundingBox(bodyId) {
    _box.makeEmpty();
    const group = this.ctx.bodies[bodyId];
    if (!group) return _box;
    group.traverse((child) => {
      if (child.isMesh) {
        child.updateWorldMatrix(true, false);
        const geom = child.geometry;
        if (!geom.boundingBox) geom.computeBoundingBox();
        const meshBox = geom.boundingBox.clone().applyMatrix4(child.matrixWorld);
        _box.union(meshBox);
      }
    });
    return _box;
  }

  /**
   * Compute world-space bounding box for all bodies (the entire bot).
   * @returns {THREE.Box3}
   */
  getAllBoundingBox() {
    _box.makeEmpty();
    for (const [, group] of Object.entries(this.ctx.bodies) as [string, any][]) {
      group.traverse((child: any) => {
        if (child.isMesh) {
          child.updateWorldMatrix(true, false);
          const geom = child.geometry;
          if (!geom.boundingBox) geom.computeBoundingBox();
          const meshBox = geom.boundingBox.clone().applyMatrix4(child.matrixWorld);
          _box.union(meshBox);
        }
      });
    }
    return _box;
  }

  /**
   * Animate camera to frame the entire bot.
   * @param {number} duration - seconds
   */
  focusOnAll(duration = 0.6) {
    const box = this.getAllBoundingBox();
    if (box.isEmpty()) return;

    box.getCenter(_center);
    box.getSize(_size);
    const maxDim = Math.max(_size.x, _size.y, _size.z, 0.05);
    const dist = maxDim * 2.0;

    // Standard isometric viewing direction (upper-right-front)
    const camDir = _camDir.set(0.4, 0.45, 0.55).normalize();

    this._startPos.copy(this.ctx.camera.position);
    this._endPos.copy(_center).addScaledVector(camDir, dist);
    this._startTarget.copy(this.ctx.controls.target);
    this._endTarget.copy(_center);
    this._animStart = performance.now() / 1000;
    this._animDuration = duration;
    this._animating = true;
  }

  /**
   * Animate camera to frame a body.
   * @param {number} bodyId
   * @param {number} duration - seconds
   */
  focusOnBody(bodyId, duration = 0.6) {
    const box = this.getBodyBoundingBox(bodyId);
    if (box.isEmpty()) return;

    box.getCenter(_center);
    box.getSize(_size);
    const maxDim = Math.max(_size.x, _size.y, _size.z, 0.05);
    const dist = maxDim * 3.5;

    // Position camera at an offset from the body center
    const cam = this.ctx.camera;
    const camDir = _camDir.subVectors(cam.position, this.ctx.controls.target).normalize();

    this._startPos.copy(cam.position);
    this._endPos.copy(_center).addScaledVector(camDir, dist);
    this._startTarget.copy(this.ctx.controls.target);
    this._endTarget.copy(_center);
    this._animStart = performance.now() / 1000;
    this._animDuration = duration;
    this._animating = true;
  }

  /**
   * Ghost (dim) all meshes except those belonging to the specified body IDs.
   * @param {number[]} keepBodyIds - body IDs to keep fully visible
   */
  ghost(keepBodyIds) {
    const keepSet = new Set(keepBodyIds);
    this.unghost(); // restore first

    this.ctx.mujocoRoot.traverse((obj) => {
      if (!obj.isMesh || !obj.material) return;
      if (keepSet.has(obj.bodyID)) return;

      this._savedMaterials.set(obj, {
        opacity: obj.material.opacity,
        transparent: obj.material.transparent,
        emissive: obj.material.emissive ? obj.material.emissive.getHex() : 0,
      });
      obj.material.opacity = 0.06;
      obj.material.transparent = true;
    });
    this.ghosted = true;
  }

  /**
   * Restore all ghosted meshes to their original material state.
   */
  unghost() {
    for (const [mesh, saved] of this._savedMaterials) {
      if (mesh.material) {
        mesh.material.opacity = saved.opacity;
        mesh.material.transparent = saved.transparent;
        if (mesh.material.emissive) mesh.material.emissive.setHex(saved.emissive);
      }
    }
    this._savedMaterials.clear();
    this.ghosted = false;
  }

  /**
   * Drive camera lerp animation. Call from requestAnimationFrame.
   */
  update() {
    if (!this._animating) return;

    const now = performance.now() / 1000;
    let t = (now - this._animStart) / this._animDuration;
    if (t >= 1) {
      t = 1;
      this._animating = false;
    }

    // Smooth ease-out
    const ease = 1 - (1 - t) ** 3;

    this.ctx.camera.position.lerpVectors(this._startPos, this._endPos, ease);
    this.ctx.controls.target.lerpVectors(this._startTarget, this._endTarget, ease);
    this.ctx.controls.update();
  }
}
