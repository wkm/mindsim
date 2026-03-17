/**
 * Component Tree — DOM tree builder from viewer manifest data.
 *
 * Builds a hierarchy: root body → child joints → child bodies (recursive),
 * with mounts as leaf children of their body.
 */

const ICONS = {
  body: { symbol: '\u25a0', color: '#8888cc' },    // filled square
  joint: { symbol: '\u25cf', color: '#66cc66' },   // filled circle
  camera: { symbol: '\u25c6', color: '#cc66cc' },  // diamond
  battery: { symbol: '\u25a0', color: '#ffaa44' },  // square (orange)
  wheel: { symbol: '\u25cf', color: '#66cccc' },   // circle (teal)
  servo: { symbol: '\u2699', color: '#66cc66' },   // gear
  component: { symbol: '\u25cb', color: '#999' },  // open circle
};

export class ComponentTree {
  /**
   * @param {HTMLElement} container
   * @param {Object} manifest - parsed viewer_manifest.json
   * @param {function} onSelect - callback(nodeId, nodeData)
   */
  constructor(container, manifest, onSelect) {
    this.container = container;
    this.manifest = manifest;
    this.onSelect = onSelect;
    this.focusedNodeId = null;

    // Index bodies and joints by name for lookup
    this.bodiesByName = {};
    for (const b of manifest.bodies) this.bodiesByName[b.name] = b;
    this.jointsByName = {};
    for (const j of manifest.joints) this.jointsByName[j.name] = j;

    // Build parent→children map
    this.childJointsOf = {}; // bodyName → [joint]
    for (const j of manifest.joints) {
      if (!this.childJointsOf[j.parent_body]) this.childJointsOf[j.parent_body] = [];
      this.childJointsOf[j.parent_body].push(j);
    }
  }

  build() {
    this.container.innerHTML = '';
    // Find root body (parent === null)
    const root = this.manifest.bodies.find(b => b.parent === null);
    if (!root) return;
    this.container.appendChild(this._buildBodyNode(root));
  }

  _buildBodyNode(body) {
    const nodeId = `body:${body.name}`;
    const hasChildren = (this.childJointsOf[body.name]?.length > 0) || (body.mounts?.length > 0);

    const node = this._createNode(nodeId, body.name, 'body', body, hasChildren);

    if (hasChildren) {
      const childrenEl = node.querySelector('.tree-node-children');

      // Mount children
      if (body.mounts) {
        for (const mount of body.mounts) {
          const mountId = `mount:${body.name}:${mount.label}`;
          const iconType = mount.component_type || 'component';
          const mountNode = this._createNode(mountId, mount.label, iconType, mount, false);
          childrenEl.appendChild(mountNode);
        }
      }

      // Joint children → body children
      const joints = this.childJointsOf[body.name] || [];
      for (const joint of joints) {
        const jointId = `joint:${joint.name}`;
        const jointNode = this._createNode(jointId, joint.name, 'joint', joint, true);
        const jointChildren = jointNode.querySelector('.tree-node-children');

        // The child body of this joint
        const childBody = this.bodiesByName[joint.child_body];
        if (childBody) {
          jointChildren.appendChild(this._buildBodyNode(childBody));
        }

        childrenEl.appendChild(jointNode);
      }
    }

    return node;
  }

  _createNode(nodeId, label, iconType, data, hasChildren) {
    const node = document.createElement('div');
    node.className = 'tree-node';
    node.dataset.nodeId = nodeId;

    const header = document.createElement('div');
    header.className = 'tree-node-header';

    // Chevron (toggle)
    const chevron = document.createElement('span');
    chevron.className = 'tree-chevron';
    if (hasChildren) {
      chevron.textContent = '\u25b6'; // right-pointing triangle
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

    // Click selects
    header.addEventListener('click', () => {
      this.setFocused(nodeId);
      this.onSelect(nodeId, data);
    });

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
