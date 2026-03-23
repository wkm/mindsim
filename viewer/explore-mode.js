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
import { GEOM_GROUP_STRUCTURAL } from './utils.js';

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
    this._isolated = false;

    // Index manifest data for quick lookup
    this.bodyNameToId = {};
    this.bodyIdToName = {};
    this.bodiesByName = {};
    for (const b of manifest.bodies) this.bodiesByName[b.name] = b;
    this._buildBodyNameMap();

    // Raycasting state
    this._raycaster = new THREE.Raycaster();
    this._mouse = new THREE.Vector2();
    this._hoveredBodyId = null;
    this._hoverEmissives = new Map(); // mesh -> original emissive hex

    // Bind event handlers (stored for cleanup on deactivate)
    this._onPointerMove = this._handlePointerMove.bind(this);
    this._onPointerDown = this._handlePointerDown.bind(this);
    this._onPointerUp = this._handlePointerUp.bind(this);
    this._pointerDownPos = new THREE.Vector2();
  }

  _buildBodyNameMap() {
    const { model, getMujocoName } = this.ctx;
    for (let b = 0; b < model.nbody; b++) {
      const name = getMujocoName(model.name_bodyadr, b);
      if (name) {
        this.bodyNameToId[name] = b;
        this.bodyIdToName[b] = name;
      }
    }
  }

  activate() {
    this.active = true;
    const treePanel = document.getElementById('tree-panel');
    if (treePanel) treePanel.style.display = 'block';

    // Build component tree with ShapeScript navigation and visibility support
    this.tree = new ComponentTree(
      document.getElementById('tree-content'),
      this.manifest,
      (nodeId, data) => this.onNodeClick(nodeId, data),
      {
        onShapeScript: (url) => { window.location.href = url; },
        onToggleVisibility: (bodyName, visible) => this._setBodyVisible(bodyName, visible),
        onIsolate: (bodyName) => this._isolateBody(bodyName),
        onShowAll: () => this._showAllBodies(),
      }
    );
    this.tree.build();

    // Show initial properties
    this._showWelcomeProperties();
    this.viz.show();

    // Attach 3D viewport interaction handlers
    const canvas = this.ctx.renderer.domElement;
    canvas.addEventListener('pointermove', this._onPointerMove);
    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointerup', this._onPointerUp);
  }

  deactivate() {
    this.active = false;
    this._clearHover();
    this.unfocus();
    this.viz.hide();
    const treePanel = document.getElementById('tree-panel');
    if (treePanel) treePanel.style.display = 'none';

    // Remove 3D viewport interaction handlers
    const canvas = this.ctx.renderer.domElement;
    canvas.removeEventListener('pointermove', this._onPointerMove);
    canvas.removeEventListener('pointerdown', this._onPointerDown);
    canvas.removeEventListener('pointerup', this._onPointerUp);
  }

  onNodeClick(nodeId, nodeData) {
    // If isolated and selecting a different node, restore visibility first
    if (this._isolated) {
      const { bodies, model } = this.ctx;
      for (let b = 1; b < model.nbody; b++) {
        if (bodies[b]) bodies[b].visible = true;
      }
      this._isolated = false;
    }

    this.focusedNodeId = nodeId;
    this.focusedData = nodeData;

    const [type] = nodeId.split(':');

    // Camera + ghost dimming
    this._applyIsolation(nodeId, nodeData);
    this._focusCamera(nodeId, nodeData);

    // Properties panel
    if (type === 'body') this._buildBodyProperties(nodeData);
    else if (type === 'joint') this._buildJointProperties(nodeData);
    else if (type === 'mount') this._buildMountProperties(nodeData);
    else if (type === 'part') this._buildPartProperties(nodeData);
    else if (type === 'fastener-group') this._buildFastenerGroupProperties(nodeData);
    else if (type === 'wire-group') this._buildWireGroupProperties(nodeData);
    else if (type === 'assembly') this._buildAssemblyProperties(nodeData);
  }

  /** Resolve a nodeId to its primary MuJoCo body ID. */
  resolveBodyId(nodeId, nodeData) {
    const [type, ...rest] = nodeId.split(':');
    if (type === 'body') return this.bodyNameToId[rest[0]];
    if (type === 'joint') return this.bodyNameToId[nodeData?.child_body];
    if (type === 'mount') return this.bodyNameToId[rest[0]];
    if (type === 'part' || type === 'fastener-group' || type === 'wire-group') {
      return this.bodyNameToId[nodeData?.parent_body];
    }
    if (type === 'assembly') {
      // Focus on the first body of the assembly
      const bodies = nodeData?.bodies || [];
      if (bodies.length > 0) return this.bodyNameToId[bodies[0]];
    }
    return undefined;
  }

  /** Resolve all MuJoCo body IDs for an assembly (for multi-body focus). */
  _resolveAssemblyBodyIds(nodeData) {
    const bodies = nodeData?.bodies || [];
    return bodies.map(n => this.bodyNameToId[n]).filter(id => id !== undefined);
  }

  /** Animate camera to frame the relevant body for a node. */
  _focusCamera(nodeId, nodeData) {
    const [type] = nodeId.split(':');

    if (type === 'assembly') {
      // Focus on all bodies in the assembly
      const bodyIds = this._resolveAssemblyBodyIds(nodeData);
      if (bodyIds.length > 0) this.focus.focusOnBody(bodyIds[0]);
    } else {
      const bodyId = this.resolveBodyId(nodeId, nodeData);
      if (bodyId !== undefined) this.focus.focusOnBody(bodyId);
    }
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
    } else if (type === 'part' || type === 'fastener-group' || type === 'wire-group') {
      // Focus on the parent body
      const parentBody = nodeData.parent_body;
      const bodyId = this.bodyNameToId[parentBody];
      if (bodyId !== undefined) {
        this.focus.ghost([bodyId]);
        this.viz.clear();
      }
    } else if (type === 'assembly') {
      // Keep all assembly bodies visible
      const bodyIds = this._resolveAssemblyBodyIds(nodeData);
      if (bodyIds.length > 0) {
        this.focus.ghost(bodyIds);
        this.viz.clear();
      }
    }
  }

  // ── Body visibility controls (driven by tree node actions) ──

  /** Set a single body's Three.js group visibility. */
  _setBodyVisible(bodyName, visible) {
    const bodyId = this.bodyNameToId[bodyName];
    if (bodyId === undefined) return;
    const group = this.ctx.bodies[bodyId];
    if (group) group.visible = visible;
  }

  /** Hide all bodies except the named one. */
  _isolateBody(bodyName) {
    for (const [name, id] of Object.entries(this.bodyNameToId)) {
      const group = this.ctx.bodies[id];
      if (!group) continue;
      if (name === bodyName) {
        group.visible = true;
      } else if (id === 0) {
        // Body 0 (world) is the parent of all others in the scene graph.
        // Hide its own meshes but keep the group visible so children render.
        group.visible = true;
        group.traverse(ch => {
          if (ch.isMesh) ch.visible = false;
        });
      } else {
        group.visible = false;
      }
    }
  }

  /** Restore all bodies to visible. */
  _showAllBodies() {
    for (const id of Object.values(this.bodyNameToId)) {
      const group = this.ctx.bodies[id];
      if (!group) continue;
      group.visible = true;
      // Restore any meshes hidden during isolation (e.g., body 0)
      group.traverse(ch => { if (ch.isMesh) ch.visible = true; });
    }
  }

  /** Isolate the currently focused node — hide everything else. */
  isolateCurrent() {
    if (!this.focusedNodeId) return;
    const [type, ...rest] = this.focusedNodeId.split(':');
    // Remove ghosting so isolated bodies render at full opacity
    this.focus.unghost();

    if (type === 'body') {
      this._isolateBody(rest[0]);
    } else if (type === 'joint' && this.focusedData) {
      // Show parent + child bodies
      const keep = new Set();
      if (this.focusedData.child_body) keep.add(this.focusedData.child_body);
      if (this.focusedData.parent_body) keep.add(this.focusedData.parent_body);
      for (const [name, id] of Object.entries(this.bodyNameToId)) {
        const group = this.ctx.bodies[id];
        if (!group) continue;
        if (keep.has(name)) {
          group.visible = true;
        } else if (id === 0) {
          // Keep body 0 visible (it's the scene graph parent) but hide its meshes
          group.visible = true;
          group.traverse(ch => { if (ch.isMesh) ch.visible = false; });
        } else {
          group.visible = false;
        }
      }
    } else if (type === 'mount' && this.focusedData) {
      this._isolateBody(this.focusedData.parent_body || rest[0]);
    } else {
      return; // can't isolate assemblies etc.
    }
    this._isolated = true;
    this._updateIsolateButton();
  }

  /** Restore all bodies and re-apply ghost dimming. */
  showAll() {
    this._showAllBodies();
    this._isolated = false;
    this._updateIsolateButton();
    if (this.focusedNodeId) {
      this._applyIsolation(this.focusedNodeId, this.focusedData);
    }
  }

  _updateIsolateButton() {
    const btn = document.getElementById('isolate-btn');
    if (btn) btn.textContent = this._isolated ? 'Show All' : 'Isolate';
  }

  unfocus() {
    this.focus.unghost();
    this.viz.clear();
    if (this._isolated) this.showAll();
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

    const parts = m.parts || [];
    const servos = parts.filter(p => p.category === 'servo');
    const fasteners = parts.filter(p => p.category === 'fastener');
    const wires = parts.filter(p => p.category === 'wire');
    html += this._propRow('Servos', servos.length);
    html += this._propRow('Fasteners', fasteners.length);
    html += this._propRow('Wire segments', wires.length);

    const totalMass = m.bodies.reduce((sum, b) => sum + (b.mass || 0), 0);
    html += this._propRow('Total mass', `${(totalMass * 1000).toFixed(0)} g`);
    html += '</div>';

    panel.innerHTML = html;
  }

  _buildBodyProperties(body) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${body.name}</h2>`;
    html += '<span class="prop-badge body-badge">Body</span>';
    if (body.kind) {
      html += ` <span style="font-size:11px;color:#5C7080;">${body.kind}</span>`;
    }

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

    // Actions row
    html += '<div style="display:flex;gap:4px;margin-top:8px;">';
    html += `<button id="isolate-btn" class="btn btn-sm">${this._isolated ? 'Show All' : 'Isolate'}</button>`;
    if (body.kind === 'fabricated') {
      const botName = this.manifest.bot_name;
      html += `<a href="?cadsteps=${encodeURIComponent(botName)}:${encodeURIComponent(body.name)}&from=${encodeURIComponent(botName)}" class="btn btn-sm" style="text-decoration:none;">View ShapeScript</a>`;
    }
    html += '</div>';

    panel.innerHTML = html;
    this._bindPropertyChipClicks(panel);
    document.getElementById('isolate-btn')?.addEventListener('click', () => {
      if (this._isolated) this.showAll(); else this.isolateCurrent();
    });
  }

  _buildJointProperties(joint) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${joint.name}</h2>`;
    html += '<span class="prop-badge joint-badge">Joint</span>';

    html += '<h3>Kinematics</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Axis', `[${joint.axis.map(v => v.toFixed(1)).join(', ')}]`);
    if (joint.range_deg) {
      html += this._propRow('Range', `${joint.range_deg[0]}\u00b0 to ${joint.range_deg[1]}\u00b0`);
    }
    html += this._propRow('Continuous', joint.continuous ? 'Yes' : 'No');
    html += '</div>';

    html += '<h3>Servo</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Model', joint.servo);
    const specs = joint.servo_specs;
    if (specs) {
      html += this._propRow('Torque', `${specs.stall_torque_nm.toFixed(2)} N\u00b7m`);
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

    html += `<div style="margin-top:8px;"><button id="isolate-btn" class="btn btn-sm">${this._isolated ? 'Show All' : 'Isolate'}</button></div>`;

    panel.innerHTML = html;
    this._bindPropertyChipClicks(panel);
    document.getElementById('isolate-btn')?.addEventListener('click', () => {
      if (this._isolated) this.showAll(); else this.isolateCurrent();
    });
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
      html += this._propRow('Size', `${(d[0] * 1000).toFixed(1)} \u00d7 ${(d[1] * 1000).toFixed(1)} \u00d7 ${(d[2] * 1000).toFixed(1)} mm`);
    }
    html += this._propRow('Mass', `${(mount.mass * 1000).toFixed(1)} g`);
    html += '</div>';
    html += `<a href="?component=${encodeURIComponent(mount.component_name)}" class="btn btn-sm" style="display:inline-block;margin-top:8px;text-decoration:none;">View in Component Browser</a>`;

    // Type-specific specs
    if (mount.component_type === 'camera') {
      html += '<h3>Camera Specs</h3>';
      html += '<div class="prop-grid">';
      html += this._propRow('FOV', `${mount.fov_deg}\u00b0`);
      if (mount.resolution) {
        html += this._propRow('Resolution', `${mount.resolution[0]} \u00d7 ${mount.resolution[1]}`);
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

    html += `<div style="margin-top:8px;"><button id="isolate-btn" class="btn btn-sm">${this._isolated ? 'Show All' : 'Isolate'}</button></div>`;

    panel.innerHTML = html;
    document.getElementById('isolate-btn')?.addEventListener('click', () => {
      if (this._isolated) this.showAll(); else this.isolateCurrent();
    });
  }

  _buildPartProperties(part) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${part.name}</h2>`;
    const catLabel = part.category || 'part';
    html += `<span class="prop-badge ${catLabel}-badge">${catLabel}</span>`;
    if (part.kind) {
      html += ` <span style="font-size:11px;color:#5C7080;">${part.kind}</span>`;
    }

    html += '<h3>Details</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Category', part.category);
    if (part.parent_body) html += this._propRow('Parent body', part.parent_body);
    if (part.joint) html += this._propRow('Joint', part.joint);
    if (part.mass) html += this._propRow('Mass', `${(part.mass * 1000).toFixed(1)} g`);
    if (part.bus_type) html += this._propRow('Bus type', part.bus_type);
    html += '</div>';

    // Servo specs (carried from joint data)
    if (part.category === 'servo' && part.servo_specs) {
      const specs = part.servo_specs;
      html += '<h3>Servo Specs</h3>';
      html += '<div class="prop-grid">';
      html += this._propRow('Torque', `${specs.stall_torque_nm.toFixed(2)} N\u00b7m`);
      const rpm = (specs.no_load_speed_rad_s * 60 / (2 * Math.PI)).toFixed(0);
      html += this._propRow('Speed', `${rpm} RPM`);
      html += this._propRow('Voltage', `${specs.voltage} V`);
      html += this._propRow('Gear ratio', `1:${specs.gear_ratio}`);
      html += this._propRow('Mass', `${(specs.mass * 1000).toFixed(0)} g`);
      html += '</div>';
    }

    // ShapeScript link
    if (part.shapescript_component) {
      html += `<a href="?cadsteps=component:${encodeURIComponent(part.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}" class="btn btn-sm" style="display:inline-block;margin-top:8px;text-decoration:none;">View ShapeScript</a>`;
    }

    panel.innerHTML = html;
  }

  _buildFastenerGroupProperties(group) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${group.label}</h2>`;
    html += '<span class="prop-badge fastener-badge">fastener</span>';

    html += '<h3>Details</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Type', group.name);
    html += this._propRow('Count', group.count);
    html += '</div>';

    panel.innerHTML = html;
  }

  _buildWireGroupProperties(group) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>Wires</h2>`;
    html += '<span class="prop-badge wire-badge">wire</span>';

    html += '<h3>Details</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Body', group.parent_body);
    html += this._propRow('Segments', group.wires.length);
    html += '</div>';

    // List individual wires
    if (group.wires.length > 0) {
      html += '<h3>Segments</h3>';
      for (const w of group.wires) {
        html += `<div style="font-size:12px;color:#A7B6C2;padding:2px 0;">${w.name}</div>`;
      }
    }

    panel.innerHTML = html;
  }

  _buildAssemblyProperties(asm) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${asm.name}</h2>`;
    html += '<span class="prop-badge" style="background:#30404D;">assembly</span>';

    html += '<h3>Contents</h3>';
    html += '<div class="prop-grid">';
    const bodies = asm.bodies || [];
    const subAsms = asm.sub_assemblies || [];
    html += this._propRow('Bodies', bodies.length);
    html += this._propRow('Sub-assemblies', subAsms.length);
    html += '</div>';

    if (bodies.length > 0) {
      html += '<h3>Bodies</h3>';
      for (const bName of bodies) {
        html += `<div class="prop-chip body-chip" data-node-id="body:${bName}">${bName}</div>`;
      }
    }

    panel.innerHTML = html;
    this._bindPropertyChipClicks(panel);
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

  // ---------------------------------------------------------------------------
  // 3D viewport raycasting — hover highlight + click-to-select
  // ---------------------------------------------------------------------------

  /** Update normalized device coords from a pointer event. */
  _updateMouse(event) {
    const rect = this.ctx.renderer.domElement.getBoundingClientRect();
    this._mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  /** Raycast and return the first structural mesh hit, or null. */
  _raycastBody() {
    this._raycaster.setFromCamera(this._mouse, this.ctx.camera);

    // Collect all structural meshes from all bodies
    const targets = [];
    for (const [, group] of Object.entries(this.ctx.bodies)) {
      group.traverse(child => {
        if (child.isMesh && child.geomGroup === GEOM_GROUP_STRUCTURAL) {
          targets.push(child);
        }
      });
    }

    const hits = this._raycaster.intersectObjects(targets, false);
    return hits.length > 0 ? hits[0].object : null;
  }

  /** Apply emissive highlight to all structural meshes of a body. */
  _highlightBody(bodyId) {
    const group = this.ctx.bodies[bodyId];
    if (!group) return;
    group.traverse(child => {
      if (child.isMesh && child.geomGroup === GEOM_GROUP_STRUCTURAL && child.material?.emissive) {
        this._hoverEmissives.set(child, child.material.emissive.getHex());
        child.material.emissive.setHex(0x666666);
      }
    });
  }

  /** Remove emissive highlight from all previously highlighted meshes. */
  _clearHover() {
    for (const [mesh, origHex] of this._hoverEmissives) {
      if (mesh.material?.emissive) mesh.material.emissive.setHex(origHex);
    }
    this._hoverEmissives.clear();
    this._hoveredBodyId = null;
    this.ctx.renderer.domElement.style.cursor = '';
  }

  _handlePointerMove(event) {
    this._updateMouse(event);
    const hit = this._raycastBody();
    const hitBodyId = hit ? hit.bodyID : null;

    if (hitBodyId === this._hoveredBodyId) return;

    this._clearHover();
    if (hitBodyId !== null && hitBodyId !== undefined) {
      this._hoveredBodyId = hitBodyId;
      this._highlightBody(hitBodyId);
      this.ctx.renderer.domElement.style.cursor = 'pointer';
    }
  }

  _handlePointerDown(event) {
    this._pointerDownPos.set(event.clientX, event.clientY);
  }

  _handlePointerUp(event) {
    // Only treat as click if the pointer didn't move much (not a drag/orbit)
    const dx = event.clientX - this._pointerDownPos.x;
    const dy = event.clientY - this._pointerDownPos.y;
    if (dx * dx + dy * dy > 25) return; // 5px threshold

    this._updateMouse(event);
    const hit = this._raycastBody();

    if (hit) {
      const bodyId = hit.bodyID;
      const bodyName = this.bodyIdToName[bodyId];
      const bodyData = bodyName ? this.bodiesByName[bodyName] : null;
      if (bodyData) {
        const nodeId = `body:${bodyName}`;
        if (this.tree) this.tree.setFocused(nodeId);
        this.onNodeClick(nodeId, bodyData);
      }
    } else {
      // Click on empty space — deselect
      this.unfocus();
      this._showWelcomeProperties();
    }
  }

  update() {
    this.focus.update();
  }
}
