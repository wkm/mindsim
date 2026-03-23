/**
 * Assembly Mode — step-through build visualization with slider.
 *
 * Reads viewer_manifest.json for structured assembly steps.
 * Falls back to auto-generating steps from the MuJoCo model if no manifest.
 */

import { FocusController } from './focus-controller.ts';
import { GEOM_GROUP_DETAIL, GEOM_GROUP_STRUCTURAL, GEOM_GROUP_WIRE } from './utils.ts';

export class AssemblyMode {
  ctx: any;
  steps: any[];
  currentStep: number;
  subProgress: number;
  active: boolean;
  allGeomMeshes: any[];
  rootBodyMeshes: any[];
  focus: FocusController;
  bodyNameToId: Record<string, number>;
  bodyGeoms: Record<number, any[]>;

  constructor(ctx: any) {
    this.ctx = ctx;
    this.steps = [];
    this.currentStep = 0;
    this.subProgress = 1.0;
    this.active = false;
    this.allGeomMeshes = [];
    this.rootBodyMeshes = []; // cached meshes on the base body (body 1)
    this.focus = new FocusController(ctx);
  }

  activate() {
    this.active = true;
    this.cacheGeomMeshes();
    this.buildBodyNameMap();
    this.loadManifest().then(() => {
      this.resolveManifestSteps();
      this.buildUI();
      this.applyStep(this.currentStep, 1.0);
    });
  }

  deactivate() {
    this.active = false;
    this.showAll();
  }

  cacheGeomMeshes() {
    this.allGeomMeshes = [];
    this.rootBodyMeshes = [];
    this.ctx.mujocoRoot.traverse((obj) => {
      if (obj.isMesh && obj.geomName !== undefined) {
        this.allGeomMeshes.push(obj);
        if (obj.bodyID === 1) this.rootBodyMeshes.push(obj);
      }
    });
  }

  /** Map body name → body ID and body ID → list of geom {name, group} */
  buildBodyNameMap() {
    const { model, getMujocoName } = this.ctx;
    this.bodyNameToId = {};
    this.bodyGeoms = {};
    for (let b = 0; b < model.nbody; b++) {
      const name = getMujocoName(model.name_bodyadr, b);
      if (name) this.bodyNameToId[name] = b;
      this.bodyGeoms[b] = [];
    }
    for (let g = 0; g < model.ngeom; g++) {
      const b = model.geom_bodyid[g];
      const gname = getMujocoName(model.name_geomadr, g);
      if (gname) this.bodyGeoms[b].push({ name: gname, group: model.geom_group[g] });
    }
  }

  /** Resolve manifest body names to bodyId and geom name lists */
  resolveManifestSteps() {
    for (const step of this.steps) {
      // Manifest uses "body" (string name), code needs "bodyId" (number)
      if (step.body && step.bodyId === undefined) {
        step.bodyId = this.bodyNameToId[step.body];
      }
      // If no geom name lists, populate from the body's geoms
      if (!step.structural && step.bodyId !== undefined) {
        const geoms = this.bodyGeoms[step.bodyId] || [];
        step.structural = geoms.map((g) => g.name);
        step.details = [];
        step.wires = [];
      }
    }
  }

  async loadManifest() {
    try {
      const resp = await fetch(`../bots/${this.ctx.botName}/viewer_manifest.json`);
      if (resp.ok) {
        const manifest = await resp.json();
        this.steps = manifest.assembly_steps;
        return;
      }
    } catch {
      /* fall through to auto-generate */
    }
    this.steps = this.autoGenerateSteps();
  }

  /** Generate assembly steps by walking the body tree (fallback when no manifest) */
  autoGenerateSteps() {
    const { model, getMujocoName, mujoco } = this.ctx;
    const HINGE = mujoco.mjtJoint.mjJNT_HINGE.value;
    const steps = [];

    for (let b = 1; b < model.nbody; b++) {
      const bodyName = getMujocoName(model.name_bodyadr, b);
      const geoms = this.bodyGeoms[b] || [];

      const structural = geoms.filter((g) => g.group === GEOM_GROUP_STRUCTURAL).map((g) => g.name);
      const details = geoms.filter((g) => g.group === GEOM_GROUP_DETAIL).map((g) => g.name);
      const wires = geoms.filter((g) => g.group === GEOM_GROUP_WIRE).map((g) => g.name);

      let jointName = '';
      for (let j = 0; j < model.njnt; j++) {
        if (model.jnt_bodyid[j] === b && model.jnt_type[j] === HINGE) {
          jointName = getMujocoName(model.name_jntadr, j);
          break;
        }
      }

      steps.push({
        title: bodyName,
        description: jointName ? `Attach via joint: ${jointName}` : 'Base structure',
        bodyId: b,
        structural,
        details,
        wires,
      });
    }

    return steps;
  }

  buildUI() {
    const panel = document.getElementById('side-panel');
    const total = this.steps.length;

    let html = '<h2>Assembly</h2>';
    html += '<p style="font-size:12px;color:#5C7080;margin-bottom:12px;">Step through the build sequence.</p>';

    html += `<div class="slider-group">
      <div class="slider-label">
        <span class="name">Build Step</span>
        <span class="value" id="asm-step-val">${this.currentStep + 1} / ${total}</span>
      </div>
      <input type="range" id="asm-step-slider" min="0" max="${total - 1}" step="1" value="${this.currentStep}" />
    </div>`;

    html += `<div class="slider-group">
      <div class="slider-label">
        <span class="name">Detail</span>
        <span class="value" id="asm-sub-val">100%</span>
      </div>
      <input type="range" id="asm-sub-slider" min="0" max="1" step="0.02" value="1" />
    </div>`;

    html += '<div id="asm-step-info" class="step-info"></div>';

    html += `<div style="display:flex;gap:8px;margin-top:12px;">
      <button class="btn btn-sm" id="asm-prev-btn">\u2190 Prev</button>
      <button class="btn btn-sm" id="asm-next-btn">Next \u2192</button>
      <button class="btn btn-sm" id="asm-show-all-btn">Show All</button>
    </div>`;

    panel.innerHTML = html;

    document.getElementById('asm-step-slider')!.addEventListener('input', (e) => {
      this.goToStep(parseInt((e.target as HTMLInputElement).value, 10));
    });

    document.getElementById('asm-sub-slider')!.addEventListener('input', (e) => {
      this.subProgress = parseFloat((e.target as HTMLInputElement).value);
      this.applyStep(this.currentStep, this.subProgress);
    });

    document.getElementById('asm-prev-btn').addEventListener('click', () => {
      if (this.currentStep > 0) this.goToStep(this.currentStep - 1);
    });

    document.getElementById('asm-next-btn').addEventListener('click', () => {
      if (this.currentStep < this.steps.length - 1) this.goToStep(this.currentStep + 1);
    });

    document.getElementById('asm-show-all-btn').addEventListener('click', () => this.showAll());
  }

  goToStep(idx) {
    this.currentStep = idx;
    this.subProgress = 1.0;
    (document.getElementById('asm-step-slider') as HTMLInputElement).value = String(idx);
    (document.getElementById('asm-sub-slider') as HTMLInputElement).value = '1';
    this.applyStep(idx, 1.0);

    // Focus camera on the step's body
    const step = this.steps[idx];
    if (step && step.bodyId !== undefined) {
      this.focus.focusOnBody(step.bodyId, 0.5);
    }
  }

  applyStep(stepIdx, progress) {
    // Update UI labels
    const stepValEl = document.getElementById('asm-step-val');
    if (stepValEl) stepValEl.textContent = `${stepIdx + 1} / ${this.steps.length}`;
    const subValEl = document.getElementById('asm-sub-val');
    if (subValEl) subValEl.textContent = `${Math.round(progress * 100)}%`;

    const step = this.steps[stepIdx];
    const infoEl = document.getElementById('asm-step-info');
    if (infoEl && step) {
      infoEl.innerHTML = `<div class="step-title">Step ${stepIdx + 1}: ${step.title}</div>
        <div class="step-desc">${step.description || ''}</div>`;
    }

    // Collect geom names visible up to this step
    const visibleGeomNames = new Set();
    const currentStepGeomNames = new Set();
    for (let i = 0; i <= stepIdx; i++) {
      const s = this.steps[i];
      if (!s) continue;
      const names = s.structural ? [...s.structural, ...s.details, ...s.wires] : s.allGeomNames || [];
      for (const n of names) {
        if (i < stepIdx) visibleGeomNames.add(n);
        else currentStepGeomNames.add(n);
      }
    }

    const visibleBodyIds = new Set([0]);
    for (let i = 0; i <= stepIdx; i++) {
      if (this.steps[i]?.bodyId !== undefined) visibleBodyIds.add(this.steps[i].bodyId);
    }

    // Apply visibility to all cached meshes
    for (const mesh of this.allGeomMeshes) {
      const geomName = mesh.geomName;
      if (!geomName && visibleBodyIds.has(mesh.bodyID)) {
        mesh.visible = true;
        mesh.material.opacity = mesh.material._origOpacity ?? mesh.material.opacity;
        continue;
      }

      if (visibleGeomNames.has(geomName)) {
        mesh.visible = true;
        if (!mesh.material._origOpacity) mesh.material._origOpacity = mesh.material.opacity;
        mesh.material.opacity = mesh.material._origOpacity;
        mesh.material.transparent = mesh.material.opacity < 1.0;
      } else if (currentStepGeomNames.has(geomName)) {
        mesh.visible = true;
        if (!mesh.material._origOpacity) mesh.material._origOpacity = mesh.material.opacity;
        mesh.material.opacity = progress * mesh.material._origOpacity;
        mesh.material.transparent = true;
      } else {
        mesh.visible = false;
      }
    }

    // Root body meshes not in any step — show if first step covers body 1
    for (const mesh of this.rootBodyMeshes) {
      if (!visibleGeomNames.has(mesh.geomName) && !currentStepGeomNames.has(mesh.geomName)) {
        if (stepIdx >= 0 && this.steps[0]?.bodyId === 1) {
          mesh.visible = true;
        }
      }
    }
  }

  showAll() {
    for (const mesh of this.allGeomMeshes) {
      mesh.visible = true;
      if (mesh.material._origOpacity) {
        mesh.material.opacity = mesh.material._origOpacity;
        mesh.material.transparent = mesh.material.opacity < 1.0;
      }
    }
    this.ctx.botScene.showAll();
    this.ctx.syncScene();
  }

  update() {
    this.focus.update();
  }
}
