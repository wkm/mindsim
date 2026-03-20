/**
 * Explore Mode — CAD-app-style component browser.
 *
 * Shows a component tree on the left, context-sensitive properties on
 * the right, and click-to-focus with camera animation and visual isolation.
 */

import * as THREE from 'three';
import { ComponentTree } from './component-tree.js';
import { FocusController } from './focus-controller.js';
import { SemanticViz } from './semantic-viz.js';

export class ExploreMode {
  constructor(ctx, manifest) {
    this.ctx = ctx;
    this.manifest = manifest;
    this.active = false;
    this.tree = null;
    this.focus = new FocusController(ctx);
    this.viz = new SemanticViz(ctx);
    this.focusedNodeId = null;
    this.focusedData = null;

    // Index manifest data for quick lookup
    this.bodyNameToId = {};
    this._buildBodyNameMap();
  }

  _buildBodyNameMap() {
    const { model, getMujocoName } = this.ctx;
    for (let b = 0; b < model.nbody; b++) {
      const name = getMujocoName(model.name_bodyadr, b);
      if (name) this.bodyNameToId[name] = b;
    }
  }

  activate() {
    this.active = true;
    const treePanel = document.getElementById('tree-panel');
    if (treePanel) treePanel.style.display = 'block';

    // Build component tree
    this.tree = new ComponentTree(
      document.getElementById('tree-content'),
      this.manifest,
      (nodeId, data) => this.onNodeClick(nodeId, data)
    );
    this.tree.build();

    // Show initial properties
    this._showWelcomeProperties();
    this.viz.show();
  }

  deactivate() {
    this.active = false;
    this.unfocus();
    this.viz.hide();
    const treePanel = document.getElementById('tree-panel');
    if (treePanel) treePanel.style.display = 'none';
  }

  onNodeClick(nodeId, nodeData) {
    this.focusedNodeId = nodeId;
    this.focusedData = nodeData;

    const [type] = nodeId.split(':');

    // Camera + isolate
    this._applyIsolation(nodeId, nodeData);
    this._focusCamera(nodeId, nodeData);

    // Properties panel
    if (type === 'body') this._buildBodyProperties(nodeData);
    else if (type === 'joint') this._buildJointProperties(nodeData);
    else if (type === 'mount') this._buildMountProperties(nodeData);
  }

  /** Resolve a nodeId to its primary MuJoCo body ID. */
  resolveBodyId(nodeId, nodeData) {
    const [type, ...rest] = nodeId.split(':');
    if (type === 'body') return this.bodyNameToId[rest[0]];
    if (type === 'joint') return this.bodyNameToId[nodeData?.child_body];
    if (type === 'mount') return this.bodyNameToId[rest[0]];
    return undefined;
  }

  /** Animate camera to frame the relevant body for a node. */
  _focusCamera(nodeId, nodeData) {
    const bodyId = this.resolveBodyId(nodeId, nodeData);
    if (bodyId !== undefined) this.focus.focusOnBody(bodyId);
  }

  /** Re-frame camera on the currently focused node. */
  refocusCurrent(duration = 0.4) {
    if (!this.focusedNodeId) return;
    const bodyId = this.resolveBodyId(this.focusedNodeId, this.focusedData);
    if (bodyId !== undefined) this.focus.focusOnBody(bodyId, duration);
  }

  /** Ghost non-focused meshes and show semantic overlays. */
  _applyIsolation(nodeId, nodeData) {
    const [type, ...rest] = nodeId.split(':');

    if (type === 'body') {
      const bodyId = this.bodyNameToId[rest[0]];
      if (bodyId !== undefined) {
        this.focus.ghost([bodyId]);
        const bbox = this.focus.getBodyBoundingBox(bodyId);
        this.viz.showBodyOverlay(nodeData, bbox);
      }
    } else if (type === 'joint') {
      const childBodyId = this.bodyNameToId[nodeData.child_body];
      if (childBodyId !== undefined) {
        const parentBodyId = this.bodyNameToId[nodeData.parent_body];
        const keepIds = [childBodyId];
        if (parentBodyId !== undefined) keepIds.push(parentBodyId);
        this.focus.ghost(keepIds);
        const bbox = this.focus.getBodyBoundingBox(childBodyId);
        this.viz.showJointOverlay(nodeData, bbox);
      }
    } else if (type === 'mount') {
      const bodyId = this.bodyNameToId[rest[0]];
      if (bodyId !== undefined) {
        this.focus.ghost([bodyId]);
        const bbox = this.focus.getBodyBoundingBox(bodyId);
        const center = bbox.getCenter(new THREE.Vector3());
        if (nodeData.component_type === 'camera') this.viz.showCameraOverlay(nodeData, center);
        else if (nodeData.component_type === 'wheel') this.viz.showWheelOverlay(nodeData, center);
        else this.viz.clear();
      }
    }
  }

  unfocus() {
    this.focus.unghost();
    this.viz.clear();
    this.focusedNodeId = null;
    this.focusedData = null;
    if (this.tree) this.tree.clearFocus();
  }

  _showWelcomeProperties() {
    const panel = document.getElementById('side-panel');
    let html = '<h2>Explore</h2>';
    html += '<p style="font-size:12px;color:#5C7080;margin-bottom:16px;">Click components in the tree to inspect them.</p>';

    // Bot summary
    const m = this.manifest;
    html += '<h3>Bot Summary</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Name', m.bot_name);
    html += this._propRow('Bodies', m.bodies.length);
    html += this._propRow('Joints', m.joints.length);

    const totalMass = m.bodies.reduce((sum, b) => sum + (b.mass || 0), 0);
    html += this._propRow('Total mass', `${(totalMass * 1000).toFixed(0)} g`);
    html += '</div>';

    panel.innerHTML = html;
  }

  _buildBodyProperties(body) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${body.name}</h2>`;
    html += '<span class="prop-badge body-badge">Body</span>';

    html += '<h3>Geometry</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Shape', body.shape);
    if (body.dimensions) {
      const d = body.dimensions;
      html += this._propRow('Width', `${(d[0] * 1000).toFixed(1)} mm`);
      html += this._propRow('Depth', `${(d[1] * 1000).toFixed(1)} mm`);
      html += this._propRow('Height', `${(d[2] * 1000).toFixed(1)} mm`);
    }
    html += this._propRow('Mass', `${(body.mass * 1000).toFixed(1)} g`);
    html += '</div>';

    // Child joints
    const childJoints = this.manifest.joints.filter(j => j.parent_body === body.name);
    if (childJoints.length > 0) {
      html += '<h3>Joints</h3>';
      for (const j of childJoints) {
        html += `<div class="prop-chip joint-chip" data-node-id="joint:${j.name}">${j.name}</div>`;
      }
    }

    // Mounts
    if (body.mounts && body.mounts.length > 0) {
      html += '<h3>Mounted Components</h3>';
      for (const m of body.mounts) {
        const typeClass = (m.component_type || 'component') + '-chip';
        html += `<div class="prop-chip ${typeClass}">${m.label} <span style="color:#5C7080;">(${m.component_name})</span></div>`;
      }
    }

    // CAD steps link
    const botName = this.manifest.bot_name;
    html += `<a href="?cadsteps=${encodeURIComponent(botName)}:${encodeURIComponent(body.name)}" class="bp5-button bp5-small" style="display:inline-block;margin-top:8px;text-decoration:none;">CAD Steps</a>`;

    panel.innerHTML = html;
    this._bindPropertyChipClicks(panel);
  }

  _buildJointProperties(joint) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${joint.name}</h2>`;
    html += '<span class="prop-badge joint-badge">Joint</span>';

    html += '<h3>Kinematics</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Axis', `[${joint.axis.map(v => v.toFixed(1)).join(', ')}]`);
    if (joint.range_deg) {
      html += this._propRow('Range', `${joint.range_deg[0]}° to ${joint.range_deg[1]}°`);
    }
    html += this._propRow('Continuous', joint.continuous ? 'Yes' : 'No');
    html += '</div>';

    html += '<h3>Servo</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Model', joint.servo);
    const specs = joint.servo_specs;
    if (specs) {
      html += this._propRow('Torque', `${specs.stall_torque_nm.toFixed(2)} N·m`);
      const rpm = (specs.no_load_speed_rad_s * 60 / (2 * Math.PI)).toFixed(0);
      html += this._propRow('Speed', `${rpm} RPM`);
      html += this._propRow('Voltage', `${specs.voltage} V`);
      html += this._propRow('Gear ratio', `1:${specs.gear_ratio}`);
      html += this._propRow('Servo mass', `${(specs.mass * 1000).toFixed(0)} g`);
    }
    html += '</div>';

    html += '<h3>Connected Bodies</h3>';
    html += `<div class="prop-chip body-chip" data-node-id="body:${joint.parent_body}">${joint.parent_body} <span style="color:#5C7080;">(parent)</span></div>`;
    html += `<div class="prop-chip body-chip" data-node-id="body:${joint.child_body}">${joint.child_body} <span style="color:#5C7080;">(child)</span></div>`;

    panel.innerHTML = html;
    this._bindPropertyChipClicks(panel);
  }

  _buildMountProperties(mount) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${mount.label}</h2>`;
    const typeLabel = mount.component_type || 'component';
    html += `<span class="prop-badge ${typeLabel}-badge">${typeLabel}</span>`;

    html += '<h3>Component</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Name', mount.component_name);
    if (mount.dimensions) {
      const d = mount.dimensions;
      html += this._propRow('Size', `${(d[0] * 1000).toFixed(1)} × ${(d[1] * 1000).toFixed(1)} × ${(d[2] * 1000).toFixed(1)} mm`);
    }
    html += this._propRow('Mass', `${(mount.mass * 1000).toFixed(1)} g`);
    html += '</div>';
    html += `<a href="?component=${encodeURIComponent(mount.component_name)}" class="bp5-button bp5-small" style="display:inline-block;margin-top:8px;text-decoration:none;">View in Component Browser</a>`;

    // Type-specific specs
    if (mount.component_type === 'camera') {
      html += '<h3>Camera Specs</h3>';
      html += '<div class="prop-grid">';
      html += this._propRow('FOV', `${mount.fov_deg}°`);
      if (mount.resolution) {
        html += this._propRow('Resolution', `${mount.resolution[0]} × ${mount.resolution[1]}`);
      }
      html += '</div>';
    } else if (mount.component_type === 'battery') {
      html += '<h3>Battery Specs</h3>';
      html += '<div class="prop-grid">';
      html += this._propRow('Chemistry', mount.chemistry);
      html += this._propRow('Voltage', `${mount.voltage} V`);
      html += this._propRow('Cells', `${mount.cells_s}S`);
      html += '</div>';
    }

    panel.innerHTML = html;
  }

  _propRow(label, value) {
    return `<div class="prop-row"><span class="prop-label">${label}</span><span class="prop-value">${value}</span></div>`;
  }

  _bindPropertyChipClicks(panel) {
    panel.querySelectorAll('.prop-chip[data-node-id]').forEach(chip => {
      chip.style.cursor = 'pointer';
      chip.addEventListener('click', () => {
        const nodeId = chip.dataset.nodeId;
        const [type, ...rest] = nodeId.split(':');
        let data;
        if (type === 'body') {
          data = this.manifest.bodies.find(b => b.name === rest[0]);
        } else if (type === 'joint') {
          data = this.manifest.joints.find(j => j.name === rest[0]);
        }
        if (data) {
          if (this.tree) this.tree.setFocused(nodeId);
          this.onNodeClick(nodeId, data);
        }
      });
    });
  }

  update() {
    this.focus.update();
  }
}
