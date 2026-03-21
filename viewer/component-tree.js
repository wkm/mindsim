/**
 * Component Tree — DOM tree builder from viewer manifest data.
 *
 * Builds a unified parts hierarchy:
 *   Assembly > Body > Parts (servos, horns, fasteners, mounts, wires) > Joints > child bodies
 *
 * Every physical object in the bot appears in the tree with color-coded icons
 * and ShapeScript navigation links.
 */

const ICONS = {
  assembly:  { symbol: '\u25b8', color: '#5C7080' },    // right triangle — gray
  body:      { symbol: '\u25a0', color: '#2B95D6' },    // filled square — blue
  servo:     { symbol: '\u25a0', color: '#182026' },    // filled square — dark gray
  horn:      { symbol: '\u25a0', color: '#E8E8E8' },    // filled square — light gray
  mount:     { symbol: '\u25a0', color: '#0F9960' },    // filled square — green
  battery:   { symbol: '\u25a0', color: '#0F9960' },    // filled square — green (mount)
  camera:    { symbol: '\u25a0', color: '#0F9960' },    // filled square — green (mount)
  compute:   { symbol: '\u25a0', color: '#0F9960' },    // filled square — green (mount)
  component: { symbol: '\u25a0', color: '#0F9960' },    // filled square — green (mount)
  wheel:     { symbol: '\u25a0', color: '#0F9960' },    // filled square — green (mount)
  fastener:  { symbol: '\u25a0', color: '#D4A843' },    // filled square — gold
  wire:      { symbol: '\u25a0', color: '#9179F2' },    // filled square — purple
  joint:     { symbol: '\u25cf', color: '#0A6640' },    // filled circle — green
};

export class ComponentTree {
  /**
   * @param {HTMLElement} container
   * @param {Object} manifest - parsed viewer_manifest.json
   * @param {function} onSelect - callback(nodeId, nodeData)
   * @param {object} [options]
   * @param {function} [options.onShapeScript] - callback(url) for ShapeScript navigation
   * @param {function} [options.onDoubleClick] - callback(nodeId, nodeData) for double-click
   */
  constructor(container, manifest, onSelect, options = {}) {
    this.container = container;
    this.manifest = manifest;
    this.onSelect = onSelect;
    this.onShapeScript = options.onShapeScript || null;
    this.onDoubleClick = options.onDoubleClick || null;
    this.focusedNodeId = null;

    // Index bodies and joints by name for lookup
    this.bodiesByName = {};
    for (const b of manifest.bodies) this.bodiesByName[b.name] = b;
    this.jointsByName = {};
    for (const j of manifest.joints) this.jointsByName[j.name] = j;

    // Build parent->children map for joints
    this.childJointsOf = {}; // bodyName -> [joint]
    for (const j of manifest.joints) {
      if (!this.childJointsOf[j.parent_body]) this.childJointsOf[j.parent_body] = [];
      this.childJointsOf[j.parent_body].push(j);
    }

    // Index parts by parent_body
    this.partsByBody = {};   // bodyName -> [part]
    this.partsByJoint = {};  // jointName -> [part]
    for (const p of (manifest.parts || [])) {
      const bodyName = p.parent_body;
      if (!this.partsByBody[bodyName]) this.partsByBody[bodyName] = [];
      this.partsByBody[bodyName].push(p);

      if (p.joint) {
        if (!this.partsByJoint[p.joint]) this.partsByJoint[p.joint] = [];
        this.partsByJoint[p.joint].push(p);
      }
    }
  }

  build() {
    this.container.innerHTML = '';

    const assemblies = this.manifest.assemblies;
    if (assemblies && assemblies.length > 0) {
      // Assembly-based hierarchy
      for (const asm of assemblies) {
        this.container.appendChild(this._buildAssemblyNode(asm));
      }
    } else {
      // Fallback: start from root body
      const root = this.manifest.bodies.find(b => b.parent === null);
      if (!root) return;
      this.container.appendChild(this._buildBodyNode(root));
    }
  }

  _buildAssemblyNode(asm) {
    const nodeId = `assembly:${asm.path || asm.name}`;
    const bodyNames = asm.bodies || [];
    const subAsms = asm.sub_assemblies || [];
    const hasChildren = bodyNames.length > 0 || subAsms.length > 0;

    const data = { ...asm, _type: 'assembly' };
    const node = this._createNode(nodeId, `${asm.name}`, 'assembly', data, hasChildren);

    if (hasChildren) {
      const childrenEl = node.querySelector('.tree-node-children');

      // Sub-assemblies first
      for (const sub of subAsms) {
        childrenEl.appendChild(this._buildAssemblyNode(sub));
      }

      // Bodies in this assembly — start from bodies that have no parent or
      // whose parent is in a different assembly. Find root bodies for this assembly.
      const asmBodySet = new Set(bodyNames);
      const rootBodies = bodyNames
        .map(n => this.bodiesByName[n])
        .filter(b => b && (b.parent === null || !asmBodySet.has(b.parent)));

      for (const body of rootBodies) {
        childrenEl.appendChild(this._buildBodyNode(body, asmBodySet));
      }
    }

    return node;
  }

  _buildBodyNode(body, asmScope = null) {
    const nodeId = `body:${body.name}`;
    const parts = this.partsByBody[body.name] || [];
    const childJoints = this.childJointsOf[body.name] || [];

    // Categorize parts for this body (excluding joint-specific parts)
    const mountedComps = parts.filter(p =>
      p.category !== 'servo' && p.category !== 'horn' &&
      p.category !== 'fastener' && p.category !== 'wire' && !p.joint
    );
    const wires = parts.filter(p => p.category === 'wire');

    // Group servos by joint
    const servosByJoint = {};
    for (const p of parts) {
      if (p.joint && (p.category === 'servo' || p.category === 'horn' || p.category === 'fastener')) {
        if (!servosByJoint[p.joint]) servosByJoint[p.joint] = [];
        servosByJoint[p.joint].push(p);
      }
    }

    // Mount-level fasteners (not associated with a joint)
    const mountFasteners = parts.filter(p => p.category === 'fastener' && !p.joint);

    const hasChildren = mountedComps.length > 0 || Object.keys(servosByJoint).length > 0 ||
      wires.length > 0 || childJoints.length > 0 || mountFasteners.length > 0;

    const kindLabel = body.kind === 'fabricated' ? 'fabricated' : (body.kind || '');
    const labelText = body.name;
    const data = { ...body, _type: 'body' };
    const shapescriptUrl = body.kind === 'fabricated'
      ? `?cadsteps=${encodeURIComponent(this.manifest.bot_name)}:${encodeURIComponent(body.name)}&from=${encodeURIComponent(this.manifest.bot_name)}`
      : null;

    const node = this._createNode(nodeId, labelText, 'body', data, hasChildren, shapescriptUrl);

    if (hasChildren) {
      const childrenEl = node.querySelector('.tree-node-children');

      // 1. Mounted components
      for (const comp of mountedComps) {
        const compNodeId = `part:${comp.id}`;
        const iconType = comp.category || 'mount';
        const compData = { ...comp, _type: 'part' };
        const compSsUrl = comp.shapescript_component
          ? `?cadsteps=component:${encodeURIComponent(comp.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
          : null;

        // Check for fasteners associated with this mount
        const compFasteners = mountFasteners.filter(f => f.mount_label === comp.mount_label);
        const hasCompChildren = compFasteners.length > 0;

        const compNode = this._createNode(compNodeId, comp.name, iconType, compData, hasCompChildren, compSsUrl);
        if (hasCompChildren) {
          const compChildrenEl = compNode.querySelector('.tree-node-children');
          // Group fasteners
          const grouped = this._groupFasteners(compFasteners);
          for (const group of grouped) {
            const fNodeId = `fastener-group:${comp.id}:${group.key}`;
            const fData = { ...group, _type: 'fastener-group' };
            const fNode = this._createNode(fNodeId, group.label, 'fastener', fData, false);
            compChildrenEl.appendChild(fNode);
          }
        }
        childrenEl.appendChild(compNode);
      }

      // 2. Servo groups per joint
      for (const joint of childJoints) {
        const jointParts = servosByJoint[joint.name] || [];
        const servos = jointParts.filter(p => p.category === 'servo');
        const horns = jointParts.filter(p => p.category === 'horn');
        const fasteners = jointParts.filter(p => p.category === 'fastener');

        for (const servo of servos) {
          const servoNodeId = `part:${servo.id}`;
          const servoLabel = `${servo.name} @ ${joint.name}`;
          const servoData = { ...servo, _type: 'part', servo_specs: joint.servo_specs };
          const servoSsUrl = servo.shapescript_component
            ? `?cadsteps=component:${encodeURIComponent(servo.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
            : null;
          const servoHasChildren = horns.length > 0 || fasteners.length > 0;
          const servoNode = this._createNode(servoNodeId, servoLabel, 'servo', servoData, servoHasChildren, servoSsUrl);

          if (servoHasChildren) {
            const servoChildrenEl = servoNode.querySelector('.tree-node-children');

            // Horns
            for (const horn of horns) {
              const hornNodeId = `part:${horn.id}`;
              const hornData = { ...horn, _type: 'part' };
              const hornSsUrl = horn.shapescript_component
                ? `?cadsteps=component:${encodeURIComponent(horn.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
                : null;
              const hornNode = this._createNode(hornNodeId, horn.name, 'horn', hornData, false, hornSsUrl);
              servoChildrenEl.appendChild(hornNode);
            }

            // Fasteners (grouped)
            const grouped = this._groupFasteners(fasteners);
            for (const group of grouped) {
              const fNodeId = `fastener-group:${joint.name}:${group.key}`;
              const fData = { ...group, _type: 'fastener-group' };
              const fNode = this._createNode(fNodeId, group.label, 'fastener', fData, false);
              servoChildrenEl.appendChild(fNode);
            }
          }

          childrenEl.appendChild(servoNode);
        }

        // Joint node -> child body
        const jointNodeId = `joint:${joint.name}`;
        const jointData = { ...joint, _type: 'joint' };
        const childBody = this.bodiesByName[joint.child_body];
        const hasJointChild = !!childBody;
        const jointNode = this._createNode(jointNodeId, `${joint.name}`, 'joint', jointData, hasJointChild);

        // Add arrow indicator
        const header = jointNode.querySelector('.tree-node-header');
        const arrow = document.createElement('span');
        arrow.style.cssText = 'color:#5C7080;font-size:11px;margin-left:4px;';
        arrow.textContent = '\u2192';
        header.appendChild(arrow);

        if (hasJointChild) {
          const jointChildrenEl = jointNode.querySelector('.tree-node-children');
          jointChildrenEl.appendChild(this._buildBodyNode(childBody, asmScope));
        }

        childrenEl.appendChild(jointNode);
      }

      // 3. Wires (collapsible group)
      if (wires.length > 0) {
        const wireGroupId = `wire-group:${body.name}`;
        const wireGroupData = { wires, _type: 'wire-group', parent_body: body.name };
        const wireGroupNode = this._createNode(
          wireGroupId, `Wires (${wires.length})`, 'wire', wireGroupData, true
        );
        const wireChildrenEl = wireGroupNode.querySelector('.tree-node-children');
        // Start collapsed
        wireChildrenEl.style.display = 'none';
        const chevron = wireGroupNode.querySelector('.tree-chevron');
        if (chevron) chevron.textContent = '\u25b6';

        for (const wire of wires) {
          const wireNodeId = `part:${wire.id}`;
          const wireData = { ...wire, _type: 'part' };
          const wireNode = this._createNode(wireNodeId, wire.name, 'wire', wireData, false);
          wireChildrenEl.appendChild(wireNode);
        }
        childrenEl.appendChild(wireGroupNode);
      }
    }

    return node;
  }

  /**
   * Group identical fasteners: "4x M2 SHC" instead of 4 separate nodes.
   */
  _groupFasteners(fasteners) {
    const groups = {};
    for (const f of fasteners) {
      const key = f.name;
      if (!groups[key]) groups[key] = { key, name: f.name, count: 0, items: [] };
      groups[key].count++;
      groups[key].items.push(f);
    }
    return Object.values(groups).map(g => ({
      ...g,
      label: g.count > 1 ? `${g.count}\u00d7 ${g.name}` : g.name,
    }));
  }

  _createNode(nodeId, label, iconType, data, hasChildren, shapescriptUrl = null) {
    const node = document.createElement('div');
    node.className = 'tree-node';
    node.dataset.nodeId = nodeId;

    const header = document.createElement('div');
    header.className = 'tree-node-header';

    // Chevron (toggle)
    const chevron = document.createElement('span');
    chevron.className = 'tree-chevron';
    if (hasChildren) {
      chevron.textContent = '\u25bc'; // down-pointing triangle (expanded by default)
      chevron.addEventListener('click', (e) => {
        e.stopPropagation();
        const children = node.querySelector('.tree-node-children');
        const expanded = children.style.display !== 'none';
        children.style.display = expanded ? 'none' : 'block';
        chevron.textContent = expanded ? '\u25b6' : '\u25bc';
      });
    } else {
      chevron.textContent = ' ';
      chevron.style.visibility = 'hidden';
    }
    header.appendChild(chevron);

    // Icon
    const icon = document.createElement('span');
    icon.className = 'tree-node-icon';
    const iconDef = ICONS[iconType] || ICONS.component;
    icon.textContent = iconDef.symbol;
    icon.style.color = iconDef.color;
    header.appendChild(icon);

    // Label
    const labelEl = document.createElement('span');
    labelEl.className = 'tree-node-label';
    labelEl.textContent = label;
    header.appendChild(labelEl);

    // ShapeScript code icon
    if (shapescriptUrl) {
      const codeBtn = document.createElement('span');
      codeBtn.className = 'tree-code-icon';
      codeBtn.textContent = '</>';
      codeBtn.title = 'View in ShapeScript';
      codeBtn.style.cssText =
        'margin-left:6px;font-size:10px;color:#5C7080;cursor:pointer;' +
        'font-family:monospace;opacity:0;transition:opacity 0.15s;';
      codeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.onShapeScript) this.onShapeScript(shapescriptUrl);
        else window.location.href = shapescriptUrl;
      });
      header.appendChild(codeBtn);

      // Show on hover
      header.addEventListener('mouseenter', () => { codeBtn.style.opacity = '1'; });
      header.addEventListener('mouseleave', () => { codeBtn.style.opacity = '0'; });
    }

    // Click selects
    header.addEventListener('click', () => {
      this.setFocused(nodeId);
      this.onSelect(nodeId, data);
    });

    // Double-click for ShapeScript navigation
    if (shapescriptUrl) {
      header.addEventListener('dblclick', (e) => {
        e.preventDefault();
        if (this.onDoubleClick) this.onDoubleClick(nodeId, data);
        else if (this.onShapeScript) this.onShapeScript(shapescriptUrl);
        else window.location.href = shapescriptUrl;
      });
    }

    node.appendChild(header);

    // Children container
    if (hasChildren) {
      const children = document.createElement('div');
      children.className = 'tree-node-children';
      // Start expanded
      children.style.display = 'block';
      node.appendChild(children);
    }

    return node;
  }

  setFocused(nodeId) {
    // Remove old focus
    const prev = this.container.querySelector('.tree-node.focused');
    if (prev) prev.classList.remove('focused');

    this.focusedNodeId = nodeId;
    const el = this.container.querySelector(`[data-node-id="${nodeId}"]`);
    if (el) el.classList.add('focused');
  }

  clearFocus() {
    this.focusedNodeId = null;
    const prev = this.container.querySelector('.tree-node.focused');
    if (prev) prev.classList.remove('focused');
  }
}
