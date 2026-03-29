/**
 * Explore Mode — CAD-app-style component browser.
 *
 * Shows a component tree on the left, context-sensitive properties on
 * the right, and click-to-focus with camera animation and visual isolation.
 */

import * as THREE from 'three';
import { ComponentTree } from './component-tree.ts';
import { FocusController } from './focus-controller.ts';
import { SemanticViz } from './semantic-viz.ts';
import type { ViewerContext } from './types.ts';
import { GEOM_GROUP_STRUCTURAL } from './utils.ts';
import { clearViewState, updateViewState } from './view-state.ts';

/**
 * Find all descendant body IDs for a given body name by walking the
 * kinematic tree through joints.  Pure function — no DOM or Three.js deps.
 */
export function getBodyAndDescendants(
  bodyName: string,
  joints: { parent_body: string; child_body: string }[],
  bodyNameToId: Record<string, number>,
): number[] {
  const ids: number[] = [];
  const id = bodyNameToId[bodyName];
  if (id !== undefined) ids.push(id);
  for (const j of joints) {
    if (j.parent_body === bodyName) {
      ids.push(...getBodyAndDescendants(j.child_body, joints, bodyNameToId));
    }
  }
  return ids;
}

export class ExploreMode {
  ctx: ViewerContext;
  manifest: any; // viewer_manifest.json — deeply nested JSON, kept as any
  active: boolean;
  tree: ComponentTree | null;
  focus: FocusController;
  viz: SemanticViz;
  focusedNodeId: string | null;
  focusedData: any; // node data varies by type (body/joint/mount/part/etc.)
  /** Cached raycast targets — invalidated when visibility changes. */
  _raycastTargets: THREE.Mesh[] | null;
  bodyNameToId: Record<string, number>;
  bodyIdToName: Record<number, string>;
  bodiesByName: Record<string, any>;
  _raycaster: THREE.Raycaster;
  _mouse: THREE.Vector2;
  _hoveredBodyId: number | null;
  _onPointerMove: (e: PointerEvent) => void;
  _onPointerDown: (e: PointerEvent) => void;
  _onPointerUp: (e: PointerEvent) => void;
  _pointerDownPos: THREE.Vector2;

  constructor(ctx: ViewerContext, manifest: any) {
    this.ctx = ctx;
    this.manifest = manifest;
    this.active = false;
    this.tree = null;
    this.focus = new FocusController(ctx);
    this.viz = new SemanticViz(ctx);
    this.focusedNodeId = null;
    this.focusedData = null;
    this._raycastTargets = null;

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

    // Build component tree with ShapeScript navigation
    this.tree = new ComponentTree(
      document.getElementById('tree-content'),
      this.manifest,
      (nodeId, data) => this.onNodeClick(nodeId, data),
      {
        onShapeScript: (url) => {
          window.location.href = url;
        },
      },
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
    // Clear hover state
    this.ctx.botScene.setHovered(null);
    this.ctx.syncScene();
    this._hoveredBodyId = null;
    this.ctx.renderer.domElement.style.cursor = '';

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

  onNodeClick(nodeId: string, nodeData: any) {
    // If isolated and selecting a different node, restore visibility first
    if (this.ctx.botScene.isIsolated()) {
      this.ctx.botScene.showAll();
      this.ctx.syncScene();
    }

    this.focusedNodeId = nodeId;
    this.focusedData = nodeData;

    const [type, ...rest] = nodeId.split(':');

    // Update URL with the selected body name
    if (type === 'body') {
      updateViewState({ select: rest[0] });
    } else {
      updateViewState({ select: nodeId });
    }

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
  resolveBodyId(nodeId: string, nodeData: any): number | undefined {
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
  _resolveAssemblyBodyIds(nodeData: any): number[] {
    const bodies = nodeData?.bodies || [];
    return bodies.map((n) => this.bodyNameToId[n]).filter((id) => id !== undefined);
  }

  /** Animate camera to frame the relevant body for a node. */
  _focusCamera(nodeId: string, nodeData: any) {
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
  _applyIsolation(nodeId: string, nodeData: any) {
    const [type, ...rest] = nodeId.split(':');

    if (type === 'body') {
      const bodyId = this.bodyNameToId[rest[0]];
      if (bodyId !== undefined) {
        this.ctx.botScene.ghostAllExcept([bodyId]);
        this.ctx.syncScene();
        const bbox = this.focus.getBodyBoundingBox(bodyId);
        this.viz.showBodyOverlay(nodeData, bbox);
      }
    } else if (type === 'joint') {
      const childBodyId = this.bodyNameToId[nodeData.child_body];
      if (childBodyId !== undefined) {
        const parentBodyId = this.bodyNameToId[nodeData.parent_body];
        const keepIds = [childBodyId];
        if (parentBodyId !== undefined) keepIds.push(parentBodyId);
        this.ctx.botScene.ghostAllExcept(keepIds);
        this.ctx.syncScene();
        const bbox = this.focus.getBodyBoundingBox(childBodyId);
        this.viz.showJointOverlay(nodeData, bbox);
      }
    } else if (type === 'mount') {
      const bodyId = this.bodyNameToId[rest[0]];
      if (bodyId !== undefined) {
        this.ctx.botScene.ghostAllExcept([bodyId]);
        this.ctx.syncScene();
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
        this.ctx.botScene.ghostAllExcept([bodyId]);
        this.ctx.syncScene();
        this.viz.clear();
      }
    } else if (type === 'assembly') {
      // Keep all assembly bodies visible
      const bodyIds = this._resolveAssemblyBodyIds(nodeData);
      if (bodyIds.length > 0) {
        this.ctx.botScene.ghostAllExcept(bodyIds);
        this.ctx.syncScene();
        this.viz.clear();
      }
    }
  }

  // ── Body visibility controls (driven by tree node actions) ──

  /** Set a single body's visibility via BotScene. */
  _setBodyVisible(bodyName: string, visible: boolean) {
    const bodyId = this.bodyNameToId[bodyName];
    if (bodyId === undefined) return;
    this.ctx.botScene.setBodyVisible(bodyId, visible);
    this.ctx.syncScene();
    this._invalidateRaycastCache();
    this._syncTreeVisualState();
  }

  /** Collect body IDs for a body and all its kinematic descendants. */
  _getBodyAndDescendants(bodyName: string): number[] {
    return getBodyAndDescendants(bodyName, this.manifest.joints, this.bodyNameToId);
  }

  /** Isolate a body and all its kinematic descendants via BotScene. */
  _isolateBody(bodyName: string) {
    const bodyIds = this._getBodyAndDescendants(bodyName);
    if (bodyIds.length === 0) return;
    this.ctx.botScene.isolateMultiple(bodyIds);
    this.ctx.syncScene();
    this._invalidateRaycastCache();
    this._syncTreeVisualState();
  }

  /** Restore all bodies to visible via BotScene. */
  _showAllBodies() {
    this.ctx.botScene.showAll();
    this.ctx.syncScene();
    this._invalidateRaycastCache();
    this._syncTreeVisualState();
  }

  /** Isolate the currently focused node — hide everything else. */
  isolateCurrent() {
    if (!this.focusedNodeId) return;
    const [type, ...rest] = this.focusedNodeId.split(':');

    if (type === 'body') {
      const bodyIds = this._getBodyAndDescendants(rest[0]);
      if (bodyIds.length === 0) return;
      this.ctx.botScene.isolateMultiple(bodyIds);
    } else if (type === 'joint' && this.focusedData) {
      // Show parent body + child body and its descendants
      const keepIds: number[] = [];
      if (this.focusedData.parent_body) {
        const id = this.bodyNameToId[this.focusedData.parent_body];
        if (id !== undefined) keepIds.push(id);
      }
      if (this.focusedData.child_body) {
        keepIds.push(...this._getBodyAndDescendants(this.focusedData.child_body));
      }
      if (keepIds.length === 0) return;
      this.ctx.botScene.isolateMultiple(keepIds);
    } else if (type === 'mount' && this.focusedData) {
      const bodyName = this.focusedData.parent_body || rest[0];
      const bodyIds = this._getBodyAndDescendants(bodyName);
      if (bodyIds.length === 0) return;
      this.ctx.botScene.isolateMultiple(bodyIds);
    } else if (type === 'assembly' && this.focusedData) {
      // Show all bodies in the assembly
      const bodyIds = this._resolveAssemblyBodyIds(this.focusedData);
      if (bodyIds.length === 0) return;
      this.ctx.botScene.isolateMultiple(bodyIds);
    } else {
      return;
    }
    this.ctx.syncScene();
    this._updateIsolateButton();
    this._syncTreeVisualState();
  }

  /** Restore all bodies and re-apply ghost dimming. */
  showAll() {
    this.ctx.botScene.showAll();
    // _applyIsolation will call syncScene(), so skip the sync here if re-applying
    if (this.focusedNodeId) {
      this._applyIsolation(this.focusedNodeId, this.focusedData);
    } else {
      this.ctx.syncScene();
    }
    this._invalidateRaycastCache();
    this._updateIsolateButton();
    this._syncTreeVisualState();
  }

  _updateIsolateButton() {
    const btn = document.getElementById('isolate-btn');
    if (btn) btn.textContent = this.ctx.botScene.isIsolated() ? 'Show All' : 'Isolate';
  }

  /** Placeholder — Sim-mode tree doesn't track per-mesh design visibility. */
  _syncTreeVisualState() {
    // no-op: Sim mode uses BotScene body-level visibility, not SceneTree.
  }

  unfocus() {
    this.ctx.botScene.unghost();
    this.ctx.syncScene();
    this.viz.clear();
    if (this.ctx.botScene.isIsolated()) this.showAll();
    this.focusedNodeId = null;
    this.focusedData = null;
    if (this.tree) this.tree.clearFocus();
    this._syncTreeVisualState();
    clearViewState('select');
  }

  _showWelcomeProperties() {
    const panel = document.getElementById('side-panel');
    let html = '<h2>Explore</h2>';
    html +=
      '<p style="font-size:12px;color:#5C7080;margin-bottom:16px;">Click components in the tree to inspect them.</p>';

    // Bot summary
    const m = this.manifest;
    html += '<h3>Bot Summary</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Name', m.bot_name);
    html += this._propRow('Bodies', m.bodies.length);
    html += this._propRow('Joints', m.joints.length);

    const parts = m.parts || [];
    const servos = parts.filter((p) => p.category === 'servo');
    const fasteners = parts.filter((p) => p.category === 'fastener');
    const wires = parts.filter((p) => p.category === 'wire');
    html += this._propRow('Servos', servos.length);
    html += this._propRow('Fasteners', fasteners.length);
    html += this._propRow('Wire segments', wires.length);

    const totalMass = m.bodies.reduce((sum, b) => sum + (b.mass || 0), 0);
    html += this._propRow('Total mass', `${(totalMass * 1000).toFixed(0)} g`);
    html += '</div>';

    panel.innerHTML = html;
  }

  _buildBodyProperties(body: any) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${body.name}</h2>`;
    html += '<span class="prop-badge body-badge">Body</span>';
    if (body.role) {
      html += ` <span style="font-size:11px;color:#5C7080;">${body.role}</span>`;
    } else if (body.kind) {
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
    const childJoints = this.manifest.joints.filter((j) => j.parent_body === body.name);
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
        const typeClass = `${m.component_type || 'component'}-chip`;
        html += `<div class="prop-chip ${typeClass}">${m.label} <span style="color:#5C7080;">(${m.component_name})</span></div>`;
      }
    }

    // Actions row
    html += '<div style="display:flex;gap:4px;margin-top:8px;">';
    html += `<button id="isolate-btn" class="btn btn-sm">${this.ctx.botScene.isIsolated() ? 'Show All' : 'Isolate'}</button>`;
    if (body.role === 'structure' || body.kind === 'fabricated') {
      const botName = this.manifest.bot_name;
      html += `<a href="?cadsteps=${encodeURIComponent(botName)}:${encodeURIComponent(body.name)}&from=${encodeURIComponent(botName)}" class="btn btn-sm" style="text-decoration:none;">View ShapeScript</a>`;
    }
    html += '</div>';

    panel.innerHTML = html;
    this._bindPropertyChipClicks(panel);
    document.getElementById('isolate-btn')?.addEventListener('click', () => {
      if (this.ctx.botScene.isIsolated()) this.showAll();
      else this.isolateCurrent();
    });
  }

  _buildJointProperties(joint: any) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${joint.name}</h2>`;
    html += '<span class="prop-badge joint-badge">Joint</span>';

    html += '<h3>Kinematics</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Axis', `[${joint.axis.map((v) => v.toFixed(1)).join(', ')}]`);
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
      const rpm = ((specs.no_load_speed_rad_s * 60) / (2 * Math.PI)).toFixed(0);
      html += this._propRow('Speed', `${rpm} RPM`);
      html += this._propRow('Voltage', `${specs.voltage} V`);
      html += this._propRow('Gear ratio', `1:${specs.gear_ratio}`);
      html += this._propRow('Servo mass', `${(specs.mass * 1000).toFixed(0)} g`);
    }
    html += '</div>';

    html += '<h3>Connected Bodies</h3>';
    html += `<div class="prop-chip body-chip" data-node-id="body:${joint.parent_body}">${joint.parent_body} <span style="color:#5C7080;">(parent)</span></div>`;
    html += `<div class="prop-chip body-chip" data-node-id="body:${joint.child_body}">${joint.child_body} <span style="color:#5C7080;">(child)</span></div>`;

    html += `<div style="margin-top:8px;"><button id="isolate-btn" class="btn btn-sm">${this.ctx.botScene.isIsolated() ? 'Show All' : 'Isolate'}</button></div>`;

    panel.innerHTML = html;
    this._bindPropertyChipClicks(panel);
    document.getElementById('isolate-btn')?.addEventListener('click', () => {
      if (this.ctx.botScene.isIsolated()) this.showAll();
      else this.isolateCurrent();
    });
  }

  _buildMountProperties(mount: any) {
    const panel = document.getElementById('side-panel');
    let html = `<h2>${mount.label}</h2>`;
    const typeLabel = mount.component_type || 'component';
    html += `<span class="prop-badge ${typeLabel}-badge">${typeLabel}</span>`;

    html += '<h3>Component</h3>';
    html += '<div class="prop-grid">';
    html += this._propRow('Name', mount.component_name);
    if (mount.dimensions) {
      const d = mount.dimensions;
      html += this._propRow(
        'Size',
        `${(d[0] * 1000).toFixed(1)} \u00d7 ${(d[1] * 1000).toFixed(1)} \u00d7 ${(d[2] * 1000).toFixed(1)} mm`,
      );
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

    html += `<div style="margin-top:8px;"><button id="isolate-btn" class="btn btn-sm">${this.ctx.botScene.isIsolated() ? 'Show All' : 'Isolate'}</button></div>`;

    panel.innerHTML = html;
    document.getElementById('isolate-btn')?.addEventListener('click', () => {
      if (this.ctx.botScene.isIsolated()) this.showAll();
      else this.isolateCurrent();
    });
  }

  _buildPartProperties(part: any) {
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
      const rpm = ((specs.no_load_speed_rad_s * 60) / (2 * Math.PI)).toFixed(0);
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

  _buildFastenerGroupProperties(group: any) {
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

  _buildWireGroupProperties(group: any) {
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

  _buildAssemblyProperties(asm: any) {
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

  _propRow(label: string, value: string | number): string {
    return `<div class="prop-row"><span class="prop-label">${label}</span><span class="prop-value">${value}</span></div>`;
  }

  _bindPropertyChipClicks(panel: HTMLElement) {
    panel.querySelectorAll<HTMLElement>('.prop-chip[data-node-id]').forEach((chip) => {
      chip.style.cursor = 'pointer';
      chip.addEventListener('click', () => {
        const nodeId = chip.dataset.nodeId;
        const [type, ...rest] = nodeId.split(':');
        let data: any;
        if (type === 'body') {
          data = this.manifest.bodies.find((b) => b.name === rest[0]);
        } else if (type === 'joint') {
          data = this.manifest.joints.find((j) => j.name === rest[0]);
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
  _updateMouse(event: PointerEvent) {
    const rect = this.ctx.renderer.domElement.getBoundingClientRect();
    this._mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  /** Build/return cached list of structural meshes for raycasting. */
  _getRaycastTargets(): any[] {
    if (!this._raycastTargets) {
      this._raycastTargets = [];
      for (const [, group] of Object.entries(this.ctx.bodies) as [string, any][]) {
        group.traverse((child: any) => {
          if (child.isMesh && child.geomGroup === GEOM_GROUP_STRUCTURAL) {
            this._raycastTargets!.push(child);
          }
        });
      }
    }
    return this._raycastTargets;
  }

  /** Invalidate cached raycast targets (call when visibility changes). */
  _invalidateRaycastCache() {
    this._raycastTargets = null;
  }

  /** Raycast and return the first structural mesh hit, or null. */
  _raycastBody(): any | null {
    this._raycaster.setFromCamera(this._mouse, this.ctx.camera);
    const hits = this._raycaster.intersectObjects(this._getRaycastTargets(), false);
    return hits.length > 0 ? hits[0].object : null;
  }

  _handlePointerMove(event: PointerEvent) {
    this._updateMouse(event);
    const hit = this._raycastBody();
    const hitBodyId: number | null = hit ? hit.bodyID : null;

    if (hitBodyId === this._hoveredBodyId) return;

    // Batch: set new hover target in one mutation + one sync
    this.ctx.botScene.setHovered(hitBodyId);
    this.ctx.syncScene();
    this._hoveredBodyId = hitBodyId;
    this.ctx.renderer.domElement.style.cursor = hitBodyId != null ? 'pointer' : '';
  }

  _handlePointerDown(event: PointerEvent) {
    this._pointerDownPos.set(event.clientX, event.clientY);
  }

  _handlePointerUp(event: PointerEvent) {
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
        updateViewState({ select: bodyName });
      }
    } else {
      // Click on empty space — deselect
      this.unfocus();
      this._showWelcomeProperties();
      clearViewState('select');
    }
  }

  update() {
    this.focus.update();
  }
}
