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
    this.angleArcs = [];  // { group, curArcGroup, axisDir, rangeMin, rangeMax, color }
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
    html += '<p style="font-size:12px;color:#5C7080;margin-bottom:16px;">Drag sliders to rotate joints. Axis arrows and angle arcs show joint geometry.</p>';

    // Blueprint.js palette: red, green, blue, gold, violet, turquoise
    const jointColors = ['#DB3737', '#0F9960', '#137CBD', '#D99E0B', '#7157D9', '#00B3A4'];
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

    html += '<div style="margin-top:16px;"><button class="bp5-button bp5-small" id="joint-reset-btn">Reset All</button></div>';
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

    const arcRadius = 0.04;

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

      // Group for this joint's arcs
      const arcGroup = new THREE.Group();
      arcGroup.userData = { bodyId };
      this.overlayGroup.add(arcGroup);

      const totalAngle = s.rangeMax - s.rangeMin;
      const sectorColor = new THREE.Color(s.color);

      // Static ROM sector (built once, never rebuilt)
      if (totalAngle > 0.01) {
        const sectorShape = new THREE.Shape();
        sectorShape.moveTo(0, 0);
        for (let seg = 0; seg <= 32; seg++) {
          const t = s.rangeMin + totalAngle * (seg / 32);
          sectorShape.lineTo(Math.cos(t) * arcRadius, Math.sin(t) * arcRadius);
        }
        sectorShape.lineTo(0, 0);
        const sector = new THREE.Mesh(
          new THREE.ShapeGeometry(sectorShape),
          new THREE.MeshBasicMaterial({
            color: sectorColor, transparent: true, opacity: 0.18,
            side: THREE.DoubleSide, depthWrite: false,
          })
        );
        orientToAxis(sector, axisDir);
        arcGroup.add(sector);

        // Static ROM edge outline
        const rangeArc = new THREE.Line(
          createArcGeometry(arcRadius, s.rangeMin, s.rangeMax, 32),
          new THREE.LineBasicMaterial({ color: sectorColor, transparent: true, opacity: 0.4 })
        );
        orientToAxis(rangeArc, axisDir);
        arcGroup.add(rangeArc);
      }

      // Sub-group for the dynamic current-angle arc (rebuilt on update)
      const curArcGroup = new THREE.Group();
      arcGroup.add(curArcGroup);

      this.angleArcs.push({
        group: arcGroup, curArcGroup, axisDir: axisDir.clone(),
        rangeMin: s.rangeMin, rangeMax: s.rangeMax, arcRadius,
      });
    }

    this.updateOverlays();
  }

  updateOverlays() {
    const { data } = this.ctx;

    for (const arrow of this.axisArrows) {
      this.ctx.getPosition(data.xpos, arrow.userData.bodyId, arrow.position);
    }

    for (let i = 0; i < this.angleArcs.length; i++) {
      const arc = this.angleArcs[i];
      const sliderInfo = this.sliders[i];
      const currentAngle = this.ctx.data.qpos[sliderInfo.qposAddr];

      this.ctx.getPosition(data.xpos, arc.group.userData.bodyId, arc.group.position);

      // Only rebuild the dynamic current-angle arc
      clearGroup(arc.curArcGroup);

      const totalAngle = arc.rangeMax - arc.rangeMin;
      if (totalAngle <= 0.01) continue;

      if (Math.abs(currentAngle) > 0.01) {
        const curMin = Math.min(0, currentAngle);
        const curMax = Math.max(0, currentAngle);
        const limitProximity = Math.max(
          1 - Math.abs(currentAngle - arc.rangeMin) / (totalAngle * 0.2),
          1 - Math.abs(arc.rangeMax - currentAngle) / (totalAngle * 0.2)
        );
        let color;
        if (limitProximity > 0.5) color = 0xDB3737;      // BP_RED3
        else if (limitProximity > 0) color = 0xD99E0B;  // BP_GOLD3
        else color = 0x0F9960;                          // BP_GREEN3

        const curArc = new THREE.Line(
          createArcGeometry(arc.arcRadius, curMin, curMax, 16),
          new THREE.LineBasicMaterial({ color, linewidth: 2 })
        );
        orientToAxis(curArc, arc.axisDir);
        arc.curArcGroup.add(curArc);
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
