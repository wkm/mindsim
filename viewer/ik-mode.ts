/**
 * IK Mode — select anchor body, drag target body, joints flex to follow.
 *
 * Uses MuJoCo's physics engine: applies forces toward the drag target
 * while holding the anchor body fixed, stepping the sim to solve IK.
 */

import * as THREE from 'three';
import { clearGroup, createMarker, radToDegStr } from './utils.ts';

export class IKMode {
  ctx: any;
  active: boolean;
  anchorBodyId: number | null;
  targetBodyId: number | null;
  dragging: boolean;
  grabDistance: number;
  raycaster: THREE.Raycaster;
  mouse: THREE.Vector2;
  overlayGroup: THREE.Group;
  _dragTarget: THREE.Vector3;
  _bodyPos: THREE.Vector3;
  _mjTarget: THREE.Vector3;
  _arrowDir: THREE.Vector3;
  _pendingDragEvt: any;
  arrow: THREE.ArrowHelper | null;
  anchorMarker: any;
  targetMarker: any;
  savedQpos: Float64Array | null;
  _jointEls: any[] | null;
  _onPointerDown: (evt: PointerEvent) => void;
  _onPointerMove: (evt: PointerEvent) => void;
  _onPointerUp: (evt: PointerEvent) => void;

  constructor(ctx: any) {
    this.ctx = ctx;
    this.active = false;
    this.anchorBodyId = null;
    this.targetBodyId = null;
    this.dragging = false;
    this.grabDistance = 0;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.overlayGroup = new THREE.Group();
    this.overlayGroup.name = 'IKOverlays';

    // Pre-allocated temporaries for drag hot path
    this._dragTarget = new THREE.Vector3();
    this._bodyPos = new THREE.Vector3();
    this._mjTarget = new THREE.Vector3();
    this._arrowDir = new THREE.Vector3();

    // Throttle: only solve IK once per animation frame
    this._pendingDragEvt = null;

    this.arrow = null;
    this.anchorMarker = null;
    this.targetMarker = null;

    this.savedQpos = null;

    // Joint readout DOM elements (created once, updated by text)
    this._jointEls = null;

    this._onPointerDown = this.onPointerDown.bind(this);
    this._onPointerMove = this.onPointerMove.bind(this);
    this._onPointerUp = this.onPointerUp.bind(this);
  }

  activate() {
    this.active = true;
    this.anchorBodyId = null;
    this.targetBodyId = null;
    this.ctx.scene.add(this.overlayGroup);
    this.buildUI();
    this.buildOverlays();
    this.saveState();

    // Ensure BotScene is in a clean state when entering IK mode
    this.ctx.botScene.showAll();
    this.ctx.syncScene();

    const el = this.ctx.renderer.domElement;
    el.addEventListener('pointerdown', this._onPointerDown);
    el.addEventListener('pointermove', this._onPointerMove);
    el.addEventListener('pointerup', this._onPointerUp);
  }

  deactivate() {
    this.active = false;
    this.restoreState();
    this.clearHighlights();
    this.ctx.scene.remove(this.overlayGroup);
    clearGroup(this.overlayGroup);
    this.arrow = null;
    this.anchorMarker = null;
    this.targetMarker = null;
    this._jointEls = null;

    // Restore BotScene to clean state when leaving IK mode
    this.ctx.botScene.showAll();
    this.ctx.syncScene();

    const el = this.ctx.renderer.domElement;
    el.removeEventListener('pointerdown', this._onPointerDown);
    el.removeEventListener('pointermove', this._onPointerMove);
    el.removeEventListener('pointerup', this._onPointerUp);
  }

  saveState() {
    this.savedQpos = new Float64Array(this.ctx.data.qpos);
  }

  restoreState() {
    if (this.savedQpos) {
      const { data, model, mujoco } = this.ctx;
      data.qpos.set(this.savedQpos);
      mujoco.mj_forward(model, data);
      this.ctx.syncTransforms();
    }
  }

  buildUI() {
    const panel = document.getElementById('side-panel');
    let html = '<h2>Inverse Kinematics</h2>';
    html += `<div class="ik-info">
      <p style="margin-bottom:12px;">Drag bodies to pose the robot using inverse kinematics.</p>
      <p><strong>How to use:</strong></p>
      <ol style="padding-left:18px;line-height:2;">
        <li>Click a body to set as <span class="anchor">anchor</span> (blue)</li>
        <li>Click another body to set as <span class="target">target</span> (red)</li>
        <li>Drag the target body to pose the robot</li>
      </ol>
      <div style="margin-top:16px;">
        <p>Anchor: <span class="anchor" id="ik-anchor-name">none</span></p>
        <p>Target: <span class="target" id="ik-target-name">none</span></p>
      </div>
    </div>`;

    html += '<h3 id="ik-joint-header" style="display:none;">Joint Angles</h3>';
    html += '<div id="ik-joint-readout"></div>';

    html += `<div style="margin-top:16px;display:flex;gap:8px;">
      <button class="btn btn-sm" id="ik-reset-btn">Reset Pose</button>
      <button class="btn btn-sm" id="ik-clear-btn">Clear Selection</button>
    </div>`;

    panel.innerHTML = html;

    document.getElementById('ik-reset-btn').addEventListener('click', () => {
      this.restoreState();
      this.updateJointReadout();
    });
    document.getElementById('ik-clear-btn').addEventListener('click', () => {
      this.anchorBodyId = null;
      this.targetBodyId = null;
      this.updateSelectionDisplay();
      this.clearHighlights();
    });

    // Build joint readout elements once
    this._buildJointReadoutElements();
  }

  _buildJointReadoutElements() {
    const { model, getMujocoName, mujoco } = this.ctx;
    const HINGE = mujoco.mjtJoint.mjJNT_HINGE.value;
    const readout = document.getElementById('ik-joint-readout');
    const header = document.getElementById('ik-joint-header');
    if (!readout) return;

    this._jointEls = [];
    let html = '';
    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] !== HINGE) continue;
      const name = getMujocoName(model.name_jntadr, j);
      const addr = model.jnt_qposadr[j];
      html += `<div style="display:flex;justify-content:space-between;font-size:11px;padding:2px 0;">
        <span style="color:#394B59;">${name}</span>
        <span style="color:#137CBD;font-family:monospace;" id="ik-jval-${j}">0.0°</span>
      </div>`;
      this._jointEls.push({ jointIdx: j, addr });
    }
    readout.innerHTML = html;
    if (header) header.style.display = this._jointEls.length ? 'block' : 'none';
  }

  buildOverlays() {
    this.arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.1, 0xdb3737);
    this.arrow.visible = false;
    this.overlayGroup.add(this.arrow);

    this.anchorMarker = createMarker(0x137cbd);
    this.overlayGroup.add(this.anchorMarker);

    this.targetMarker = createMarker(0xdb3737);
    this.overlayGroup.add(this.targetMarker);
  }

  clearHighlights() {
    // Clear IK-specific direct emissive colors (anchor blue, target red)
    this.ctx.mujocoRoot.traverse((obj: any) => {
      if (obj.isMesh && obj.material?.emissive) {
        obj.material.emissive.setHex(0x000000);
      }
    });
    // Also clear BotScene hover/selected state
    this.ctx.botScene.setHovered(null);
    this.ctx.botScene.setSelected(null);
    this.ctx.syncScene();
  }

  highlightBody(bodyId, color) {
    this.ctx.mujocoRoot.traverse((obj) => {
      if (obj.isMesh && obj.bodyID === bodyId && obj.material?.emissive) {
        obj.material.emissive.setHex(color);
      }
    });
  }

  updateSelectionDisplay() {
    const { getMujocoName, model } = this.ctx;
    const anchorEl = document.getElementById('ik-anchor-name');
    const targetEl = document.getElementById('ik-target-name');

    if (anchorEl) {
      anchorEl.textContent =
        this.anchorBodyId !== null
          ? getMujocoName(model.name_bodyadr, this.anchorBodyId) || `body ${this.anchorBodyId}`
          : 'none';
    }
    if (targetEl) {
      targetEl.textContent =
        this.targetBodyId !== null
          ? getMujocoName(model.name_bodyadr, this.targetBodyId) || `body ${this.targetBodyId}`
          : 'none';
    }
  }

  updateMarkerPositions() {
    if (this.anchorMarker) {
      if (this.anchorBodyId !== null) {
        this.ctx.getPosition(this.ctx.data.xpos, this.anchorBodyId, this.anchorMarker.position);
        this.anchorMarker.visible = true;
      } else {
        this.anchorMarker.visible = false;
      }
    }
    if (this.targetMarker) {
      if (this.targetBodyId !== null) {
        this.ctx.getPosition(this.ctx.data.xpos, this.targetBodyId, this.targetMarker.position);
        this.targetMarker.visible = true;
      } else {
        this.targetMarker.visible = false;
      }
    }
  }

  updateJointReadout() {
    if (!this._jointEls) return;
    const { data } = this.ctx;
    for (const { jointIdx, addr } of this._jointEls) {
      const el = document.getElementById(`ik-jval-${jointIdx}`);
      if (el) el.textContent = `${radToDegStr(data.qpos[addr])}°`;
    }
  }

  // --- Pointer events ---

  updateRaycaster(evt) {
    const rect = this.ctx.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.ctx.camera);
  }

  findBodyUnderPointer(evt) {
    this.updateRaycaster(evt);
    const intersects = this.raycaster.intersectObjects(this.ctx.scene.children, true);
    for (const hit of intersects) {
      let obj: any = hit.object;
      while (obj && obj.bodyID === undefined) obj = obj.parent;
      if (obj && obj.bodyID > 0) {
        return { bodyId: obj.bodyID, distance: hit.distance, point: hit.point };
      }
    }
    return null;
  }

  onPointerDown(evt) {
    if (!this.active || evt.button !== 0) return;

    const hit = this.findBodyUnderPointer(evt);
    if (!hit) return;

    if (this.anchorBodyId === null) {
      this.anchorBodyId = hit.bodyId;
      this.clearHighlights();
      this.highlightBody(hit.bodyId, 0x224488);
      this.updateSelectionDisplay();
      this.updateMarkerPositions();
    } else if (this.targetBodyId === null && hit.bodyId !== this.anchorBodyId) {
      this.targetBodyId = hit.bodyId;
      this.highlightBody(hit.bodyId, 0x882222);
      this.updateSelectionDisplay();
      this.updateMarkerPositions();
      this.updateJointReadout();
    } else if (hit.bodyId === this.targetBodyId) {
      this.dragging = true;
      this.grabDistance = hit.distance;
      this.ctx.controls.enabled = false;
    }
  }

  onPointerMove(evt) {
    if (!this.active || !this.dragging || this.targetBodyId === null) return;
    // Throttle: store event, solve in update() once per frame
    this._pendingDragEvt = evt;
  }

  onPointerUp() {
    if (this.dragging) {
      this.dragging = false;
      this._pendingDragEvt = null;
      this.ctx.controls.enabled = true;
      if (this.arrow) this.arrow.visible = false;
    }
  }

  /** Solve IK by applying forces and stepping the simulation */
  solveIK(targetWorldPos) {
    const { mujoco, model, data } = this.ctx;
    const bodyId = this.targetBodyId;

    // Convert Three.js position to MuJoCo coordinates (reuse pre-allocated vector)
    this._mjTarget.copy(targetWorldPos);
    this.ctx.toMujocoPos(this._mjTarget);

    const bx = data.xpos[bodyId * 3 + 0];
    const by = data.xpos[bodyId * 3 + 1];
    const bz = data.xpos[bodyId * 3 + 2];

    const stiffness = model.body_mass[bodyId] * 500;
    const fx = (this._mjTarget.x - bx) * stiffness;
    const fy = (this._mjTarget.y - by) * stiffness;
    const fz = (this._mjTarget.z - bz) * stiffness;

    for (let i = 0; i < data.qfrc_applied.length; i++) data.qfrc_applied[i] = 0;

    mujoco.mj_applyFT(model, data, [fx, fy, fz], [0, 0, 0], [bx, by, bz], bodyId, data.qfrc_applied);

    const savedTimestep = model.opt.timestep;
    model.opt.timestep = 0.002;
    for (let i = 0; i < 20; i++) mujoco.mj_step(model, data);
    model.opt.timestep = savedTimestep;

    // Zero velocities to prevent drift
    for (let i = 0; i < data.qvel.length; i++) data.qvel[i] = 0;
    mujoco.mj_forward(model, data);
  }

  update() {
    // Process throttled drag event (once per frame)
    if (this._pendingDragEvt && this.dragging && this.targetBodyId !== null) {
      const evt = this._pendingDragEvt;
      this._pendingDragEvt = null;

      this.updateRaycaster(evt);
      this._dragTarget.copy(this.raycaster.ray.origin).addScaledVector(this.raycaster.ray.direction, this.grabDistance);

      this.solveIK(this._dragTarget);
      this.ctx.syncTransforms();
      this.updateJointReadout();

      // Update arrow
      if (this.arrow) {
        this.ctx.getPosition(this.ctx.data.xpos, this.targetBodyId, this._bodyPos);
        this.arrow.position.copy(this._bodyPos);
        this._arrowDir.subVectors(this._dragTarget, this._bodyPos);
        const len = this._arrowDir.length();
        if (len > 0.001) {
          this.arrow.setDirection(this._arrowDir.normalize());
          this.arrow.setLength(len, len * 0.2, len * 0.1);
          this.arrow.visible = true;
        }
      }
    }

    // Update markers
    this.updateMarkerPositions();
  }
}
