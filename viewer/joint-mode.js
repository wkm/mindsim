/**
 * Joint Validation Mode — sliders for each joint, axis arrows, angle arcs.
 */

import * as THREE from 'three';
import { clearGroup, radToDegStr, createArcGeometry, orientToAxis } from './utils.js';

export class JointMode {
  constructor(ctx) {
    this.ctx = ctx;
    this.sliders = [];
    this.axisArrows = [];
    this.angleArcs = [];
    this.overlayGroup = new THREE.Group();
    this.overlayGroup.name = 'JointOverlays';
    this.active = false;
  }

  activate() {
    this.active = true;
    this.ctx.scene.add(this.overlayGroup);
    this.buildUI();
    this.buildOverlays();
    this.updateFromModel();
  }

  deactivate() {
    this.active = false;
    this.ctx.scene.remove(this.overlayGroup);
    clearGroup(this.overlayGroup);
    this.axisArrows = [];
    this.angleArcs = [];
  }

  buildUI() {
    const panel = document.getElementById('side-panel');
    const { model, getMujocoName, mujoco } = this.ctx;
    const HINGE = mujoco.mjtJoint.mjJNT_HINGE.value;

    let html = '<h2>Joint Validation</h2>';
    html += '<p style="font-size:12px;color:#888;margin-bottom:16px;">Drag sliders to rotate joints. Axis arrows and angle arcs show joint geometry.</p>';

    const jointColors = ['#ff6666', '#66cc66', '#6688ff', '#ffaa44', '#cc66cc', '#66cccc'];
    this.sliders = [];

    for (let j = 0; j < model.njnt; j++) {
      if (model.jnt_type[j] !== HINGE) continue;

      const name = getMujocoName(model.name_jntadr, j);
      if (!name) continue;

      const qposAddr = model.jnt_qposadr[j];
      let rangeMin = -3.14, rangeMax = 3.14;
      if (model.jnt_limited[j]) {
        rangeMin = model.jnt_range[j * 2 + 0];
        rangeMax = model.jnt_range[j * 2 + 1];
      }

      const color = jointColors[this.sliders.length % jointColors.length];
      const sliderId = `joint-slider-${j}`;

      html += `<div class="slider-group">
        <div class="slider-label">
          <span class="name"><span class="joint-color" style="background:${color}"></span>${name}</span>
          <span class="value" id="${sliderId}-val">0.0°</span>
        </div>
        <input type="range" id="${sliderId}" min="${rangeMin}" max="${rangeMax}" step="0.01" value="0" />
      </div>`;

      this.sliders.push({ jointIdx: j, qposAddr, rangeMin, rangeMax, sliderId, color, name });
    }

    html += '<div style="margin-top:16px;"><button class="btn" id="joint-reset-btn">Reset All</button></div>';
    panel.innerHTML = html;

    for (const s of this.sliders) {
      const el = document.getElementById(s.sliderId);
      el.addEventListener('input', () => this.onSliderChange(s, parseFloat(el.value)));
    }
    document.getElementById('joint-reset-btn').addEventListener('click', () => this.resetAll());
  }

  onSliderChange(sliderInfo, value) {
    const { model, data, mujoco } = this.ctx;
    data.qpos[sliderInfo.qposAddr] = value;
    mujoco.mj_forward(model, data);
    this.ctx.syncTransforms();
    this.updateOverlays();
    this.updateSliderDisplay(sliderInfo, value);
  }

  updateSliderDisplay(sliderInfo, value) {
    const valEl = document.getElementById(`${sliderInfo.sliderId}-val`);
    if (valEl) valEl.textContent = `${radToDegStr(value)}°`;
  }

  resetAll() {
    const { model, data, mujoco } = this.ctx;
    for (const s of this.sliders) {
      data.qpos[s.qposAddr] = 0;
      const el = document.getElementById(s.sliderId);
      if (el) el.value = 0;
      this.updateSliderDisplay(s, 0);
    }
    mujoco.mj_forward(model, data);
    this.ctx.syncTransforms();
    this.updateOverlays();
  }

  buildOverlays() {
    clearGroup(this.overlayGroup);
    this.axisArrows = [];
    this.angleArcs = [];

    for (const s of this.sliders) {
      const j = s.jointIdx;
      const bodyId = this.ctx.model.jnt_bodyid[j];

      // Axis arrow
      const axisDir = new THREE.Vector3();
      this.ctx.getPosition(this.ctx.model.jnt_axis, j, axisDir);
      axisDir.normalize();

      const arrow = new THREE.ArrowHelper(axisDir, new THREE.Vector3(), 0.06, s.color, 0.015, 0.008);
      arrow.userData = { jointIdx: j, bodyId };
      this.overlayGroup.add(arrow);
      this.axisArrows.push(arrow);

      // Angle arc group — children rebuilt on update, but the group persists
      const arcGroup = new THREE.Group();
      arcGroup.userData = { jointIdx: j, bodyId, axisDir: axisDir.clone(), rangeMin: s.rangeMin, rangeMax: s.rangeMax };
      this.overlayGroup.add(arcGroup);
      this.angleArcs.push(arcGroup);
    }

    this.updateOverlays();
  }

  updateOverlays() {
    const { data } = this.ctx;

    for (const arrow of this.axisArrows) {
      this.ctx.getPosition(data.xpos, arrow.userData.bodyId, arrow.position);
    }

    for (let i = 0; i < this.angleArcs.length; i++) {
      const arcGroup = this.angleArcs[i];
      const { bodyId, axisDir, rangeMin, rangeMax } = arcGroup.userData;
      const sliderInfo = this.sliders[i];
      const currentAngle = this.ctx.data.qpos[sliderInfo.qposAddr];

      this.ctx.getPosition(data.xpos, bodyId, arcGroup.position);

      // Clear old arcs
      clearGroup(arcGroup);

      const arcRadius = 0.04;
      const totalAngle = rangeMax - rangeMin;
      if (totalAngle <= 0.01) continue;

      // ROM disk (filled sector in joint color)
      const sectorShape = new THREE.Shape();
      sectorShape.moveTo(0, 0);
      for (let s = 0; s <= 32; s++) {
        const t = rangeMin + (rangeMax - rangeMin) * (s / 32);
        sectorShape.lineTo(Math.cos(t) * arcRadius, Math.sin(t) * arcRadius);
      }
      sectorShape.lineTo(0, 0);
      const sectorGeo = new THREE.ShapeGeometry(sectorShape);
      const sectorColor = new THREE.Color(sliderInfo.color);
      const sector = new THREE.Mesh(sectorGeo, new THREE.MeshBasicMaterial({
        color: sectorColor, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false,
      }));
      orientToAxis(sector, axisDir);
      arcGroup.add(sector);

      // ROM edge outline
      const rangeArc = new THREE.Line(
        createArcGeometry(arcRadius, rangeMin, rangeMax, 32),
        new THREE.LineBasicMaterial({ color: sectorColor, transparent: true, opacity: 0.4 })
      );
      orientToAxis(rangeArc, axisDir);
      arcGroup.add(rangeArc);

      // Current angle arc (colored)
      if (Math.abs(currentAngle) > 0.01) {
        const curMin = Math.min(0, currentAngle);
        const curMax = Math.max(0, currentAngle);
        const limitProximity = Math.max(
          1 - Math.abs(currentAngle - rangeMin) / (totalAngle * 0.2),
          1 - Math.abs(rangeMax - currentAngle) / (totalAngle * 0.2)
        );
        let color;
        if (limitProximity > 0.5) color = 0xff4444;
        else if (limitProximity > 0) color = 0xffaa44;
        else color = 0x44cc44;

        const curArc = new THREE.Line(
          createArcGeometry(arcRadius, curMin, curMax, 16),
          new THREE.LineBasicMaterial({ color, linewidth: 2 })
        );
        orientToAxis(curArc, axisDir);
        arcGroup.add(curArc);
      }
    }
  }

  updateFromModel() {
    for (const s of this.sliders) {
      const val = this.ctx.data.qpos[s.qposAddr];
      const el = document.getElementById(s.sliderId);
      if (el) el.value = val;
      this.updateSliderDisplay(s, val);
    }
    this.updateOverlays();
  }

  update() {}
}
