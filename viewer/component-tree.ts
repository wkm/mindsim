/**
 * Component Tree — DOM tree builder from viewer manifest data.
 *
 * Builds a unified parts hierarchy:
 *   Assembly > Body > Parts (servos, horns, fasteners, mounts, wires) > Joints > child bodies
 *
 * Every physical object in the bot appears in the tree with color-coded icons
 * and ShapeScript navigation links.
 *
 * Features:
 *   - Search input filters nodes by name (case-insensitive, debounced)
 *   - Category filter chips toggle visibility of node types
 *   - Collapsible sections with animated disclosure triangles (SVG chevrons)
 *   - Filled/empty circle visibility toggles (click to toggle, right-click to solo)
 *   - Indentation guides, hover highlights, selected state styling
 */

import type { SceneTree } from './design-scene.ts';

// ── SVG Icons ──

const CHEVRON_RIGHT = `<svg viewBox="0 0 16 16" width="10" height="10"><path d="M6 3l5 5-5 5" stroke="currentColor" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

const ICONS = {
  assembly: { symbol: '\u25b8', color: '#5C7080' }, // right triangle — gray
  body: { symbol: '\u25a0', color: '#2B95D6' }, // filled square — blue
  servo: { symbol: '\u25a0', color: '#182026' }, // filled square — dark gray
  horn: { symbol: '\u25a0', color: '#E8E8E8' }, // filled square — light gray
  mount: { symbol: '\u25a0', color: '#0F9960' }, // filled square — green
  battery: { symbol: '\u25a0', color: '#0F9960' }, // filled square — green (mount)
  camera: { symbol: '\u25a0', color: '#0F9960' }, // filled square — green (mount)
  compute: { symbol: '\u25a0', color: '#0F9960' }, // filled square — green (mount)
  component: { symbol: '\u25a0', color: '#0F9960' }, // filled square — green (mount)
  wheel: { symbol: '\u25a0', color: '#0F9960' }, // filled square — green (mount)
  fastener: { symbol: '\u25a0', color: '#D4A843' }, // filled square — gold
  wire: { symbol: '\u25a0', color: '#9179F2' }, // filled square — purple
  joint: { symbol: '\u25cf', color: '#0A6640' }, // filled circle — green
};

/** Map category filter keys to display labels and icon colors. */
const CATEGORY_CHIPS = [
  { key: 'body', label: 'Bodies', color: '#2B95D6' },
  { key: 'servo', label: 'Servos', color: '#182026' },
  { key: 'mount', label: 'Components', color: '#0F9960' },
  { key: 'fastener', label: 'Fasteners', color: '#D4A843' },
  { key: 'wire', label: 'Wires', color: '#9179F2' },
];

/** Map icon types to filter categories for data-category. */
function iconTypeToCategory(iconType) {
  if (iconType === 'body') return 'body';
  if (iconType === 'servo') return 'servo';
  if (iconType === 'horn') return 'servo'; // horns group with servos
  if (iconType === 'fastener') return 'fastener';
  if (iconType === 'wire') return 'wire';
  if (iconType === 'joint') return 'joint';
  if (iconType === 'assembly') return 'assembly';
  // mount, battery, camera, compute, component, wheel → mount
  return 'mount';
}

/** Options for ComponentTree constructor. */
interface ComponentTreeOptions {
  onShapeScript?: (url: string) => void;
  onDoubleClick?: (nodeId: string, data: any) => void;
  onToggleNodeHidden?: (nodeId: string) => void;
  onSolo?: (nodeId: string) => void;
  onUnsolo?: () => void;
}

export class ComponentTree {
  container: HTMLElement;
  manifest: any;
  onSelect: (nodeId: string, data: any) => void;
  onShapeScript: ((url: string) => void) | null;
  onDoubleClick: ((nodeId: string, data: any) => void) | null;
  onToggleNodeHidden: ((nodeId: string) => void) | null;
  onSolo: ((nodeId: string) => void) | null;
  onUnsolo: (() => void) | null;
  focusedNodeId: string | null;
  _filters: Record<string, boolean>;
  _searchQuery: string;
  _searchTimeout: ReturnType<typeof setTimeout> | null;
  _searchInput: HTMLInputElement | null;
  _treeRoot: HTMLDivElement | null;
  bodiesByName: Record<string, any>;
  jointsByName: Record<string, any>;
  childJointsOf: Record<string, any[]>;
  partsByBody: Record<string, any[]>;
  partsByJoint: Record<string, any[]>;

  constructor(
    container: HTMLElement,
    manifest: any,
    onSelect: (nodeId: string, data: any) => void,
    options: ComponentTreeOptions = {},
  ) {
    this.container = container;
    this.manifest = manifest;
    this.onSelect = onSelect;
    this.onShapeScript = options.onShapeScript || null;
    this.onDoubleClick = options.onDoubleClick || null;
    this.onToggleNodeHidden = options.onToggleNodeHidden || null;
    this.onSolo = options.onSolo || null;
    this.onUnsolo = options.onUnsolo || null;
    this.focusedNodeId = null;

    // Escape key un-solos
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.onUnsolo) this.onUnsolo();
    });

    // Category filter state
    this._filters = {
      body: true,
      servo: true,
      mount: true,
      horn: true,
      fastener: false, // off by default
      wire: false, // off by default
    };

    this._searchQuery = '';
    this._searchTimeout = null;

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
    this.partsByBody = {}; // bodyName -> [part]
    this.partsByJoint = {}; // jointName -> [part]
    for (const p of manifest.parts || []) {
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

    // Inject CSS for the polished tree (once)
    this._injectStyles();

    // Build toolbar: search + filter chips
    this._buildToolbar();

    // Tree content wrapper
    this._treeRoot = document.createElement('div');
    this._treeRoot.className = 'tree-root';
    this.container.appendChild(this._treeRoot);

    const assemblies = this.manifest.assemblies;
    if (assemblies && assemblies.length > 0) {
      for (const asm of assemblies) {
        this._treeRoot.appendChild(this._buildAssemblyNode(asm));
      }
    } else {
      // Auto-generate assemblies from kinematic structure
      const rootBody = this.manifest.bodies.find((b) => b.parent === null);
      if (!rootBody) return;
      const autoAsm = this._buildAutoAssemblyFromKinematics(rootBody);
      this._treeRoot.appendChild(autoAsm);
    }

    // Apply initial filter state (hide fasteners + wires by default)
    this._applyFilters();
  }

  // ── Toolbar: search input + category filter chips ──

  _buildToolbar() {
    const toolbar = document.createElement('div');
    toolbar.className = 'tree-toolbar';

    // Top row: search + Show All button
    const topRow = document.createElement('div');
    topRow.style.cssText = 'display:flex; gap:4px; align-items:center; margin-bottom:6px;';

    // Search input wrapper
    const searchWrap = document.createElement('div');
    searchWrap.className = 'tree-search-wrap';
    searchWrap.style.flex = '1';

    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Search parts\u2026';
    searchInput.className = 'tree-search';
    this._searchInput = searchInput;

    const clearBtn = document.createElement('span');
    clearBtn.className = 'tree-search-clear';
    clearBtn.textContent = '\u00d7';
    clearBtn.style.display = 'none';
    clearBtn.addEventListener('click', () => {
      searchInput.value = '';
      clearBtn.style.display = 'none';
      this._searchQuery = '';
      this._applySearch();
    });

    searchInput.addEventListener('input', () => {
      clearBtn.style.display = searchInput.value ? 'block' : 'none';
      clearTimeout(this._searchTimeout);
      this._searchTimeout = setTimeout(() => {
        this._searchQuery = searchInput.value.toLowerCase();
        this._applySearch();
      }, 150);
    });

    searchWrap.appendChild(searchInput);
    searchWrap.appendChild(clearBtn);
    topRow.appendChild(searchWrap);

    toolbar.appendChild(topRow);

    // Category filter chips
    const chipBar = document.createElement('div');
    chipBar.className = 'tree-filter-bar';

    for (const chip of CATEGORY_CHIPS) {
      const btn = document.createElement('button');
      btn.className = 'tree-filter-chip';
      btn.dataset.filterKey = chip.key;

      const dot = document.createElement('span');
      dot.className = 'tree-filter-dot';
      dot.style.background = chip.color;
      btn.appendChild(dot);

      const lbl = document.createTextNode(chip.label);
      btn.appendChild(lbl);

      // Reflect initial state
      const active = this._filters[chip.key] !== false;
      btn.classList.toggle('active', active);

      btn.addEventListener('click', () => {
        const key = chip.key;
        this._filters[key] = !this._filters[key];
        btn.classList.toggle('active', this._filters[key]);
        this._applyFilters();
      });

      chipBar.appendChild(btn);
    }

    toolbar.appendChild(chipBar);
    this.container.appendChild(toolbar);
  }

  // ── Visibility controls ──

  /**
   * Resolve the parent body name for any node.
   * Body nodes return their own name; parts/joints return their parent body.
   */
  _resolveBodyName(nodeId, data) {
    const [type, ...rest] = nodeId.split(':');
    if (type === 'body') return rest[0];
    if (type === 'joint') return data?.parent_body;
    if (type === 'part' || type === 'fastener-group' || type === 'wire-group') return data?.parent_body;
    if (type === 'assembly') {
      // Return first body of assembly
      const bodies = data?.bodies || [];
      return bodies.length > 0 ? bodies[0] : null;
    }
    return null;
  }

  /**
   * Update tree node styling from a SceneTree visibility model.
   *
   * Three visual states per node:
   *   - Visible (own hidden=false, all ancestors visible):
   *       dot = filled blue, row full opacity
   *   - Explicitly hidden (own hidden=true):
   *       dot = empty circle, row dimmed
   *   - Implicitly hidden (own hidden=false but an ancestor is hidden):
   *       dot = filled blue (shows it will reappear when ancestor is unhidden),
   *       row dimmed
   */
  updateFromDesignScene(tree: SceneTree): void {
    const allNodes = this._treeRoot?.querySelectorAll<HTMLElement>('.tree-node[data-node-id]');
    if (!allNodes) return;
    for (const domNode of allNodes) {
      const nodeId = domNode.dataset.nodeId!;
      const node = tree.getNode(nodeId);
      if (!node) continue;

      const ownHidden = node.hidden;
      const effectivelyVisible = tree.resolveVisibility(nodeId);

      const dot = domNode.querySelector('.vis-dot') as HTMLElement;
      if (dot) {
        // Dot shows OWN state: filled blue if not hidden, empty circle if hidden
        dot.style.background = ownHidden ? 'transparent' : '#58a6ff';
        dot.style.border = ownHidden ? '2px solid #484f58' : 'none';
        dot.style.boxSizing = 'border-box';
      }

      // Row opacity shows EFFECTIVE visibility (considers ancestors + solo)
      domNode.style.opacity = effectivelyVisible ? '1' : '0.45';
    }
  }

  // ── Filter + search application ──

  _applyFilters() {
    if (!this._treeRoot) return;
    const nodes = this._treeRoot.querySelectorAll<HTMLElement>('.tree-node[data-category]');
    for (const node of nodes) {
      const cat = node.dataset.category!;
      // Categories not in the filter map are always shown (assembly, joint)
      if (cat in this._filters) {
        node.style.display = this._filters[cat] ? '' : 'none';
      }
    }
    // Hide chevrons on nodes where all children are now hidden
    const parents = this._treeRoot.querySelectorAll<HTMLElement>('.tree-node-children');
    for (const childrenEl of parents) {
      const visibleKids = [...childrenEl.children].filter((c) => (c as HTMLElement).style.display !== 'none');
      const header = childrenEl.parentElement?.querySelector(':scope > .tree-node-header');
      const chevron = header?.querySelector('.tree-chevron') as HTMLElement | null;
      if (chevron) {
        chevron.style.display = visibleKids.length > 0 ? '' : 'none';
      }
    }
  }

  _applySearch() {
    if (!this._treeRoot) return;
    const q = this._searchQuery;
    const allNodes = this._treeRoot.querySelectorAll<HTMLElement>('.tree-node');

    if (!q) {
      // Clear search — restore all, re-apply category filters
      for (const n of allNodes) {
        n.classList.remove('search-hidden', 'search-match');
      }
      this._applyFilters();
      return;
    }

    // First pass: mark matches
    for (const n of allNodes) {
      const label = n.querySelector(':scope > .tree-node-header .tree-node-label');
      const text = label ? label.textContent.toLowerCase() : '';
      if (text.includes(q)) {
        n.classList.add('search-match');
        n.classList.remove('search-hidden');
      } else {
        n.classList.remove('search-match');
        n.classList.add('search-hidden');
      }
    }

    // Second pass: show ancestors of matches
    const matches = this._treeRoot.querySelectorAll('.tree-node.search-match');
    for (const m of matches) {
      let parent = m.parentElement;
      while (parent && parent !== this._treeRoot) {
        if (parent.classList.contains('tree-node')) {
          parent.classList.remove('search-hidden');
          // Expand parent so the match is visible
          const children = parent.querySelector(':scope > .tree-node-children') as HTMLElement | null;
          if (children) children.style.display = 'block';
          const chevron = parent.querySelector(':scope > .tree-node-header .tree-chevron');
          if (chevron) {
            chevron.classList.add('expanded');
          }
        }
        parent = parent.parentElement;
      }
    }

    // Apply display
    for (const n of allNodes) {
      if (n.classList.contains('search-hidden')) {
        n.style.display = 'none';
      } else {
        n.style.display = '';
      }
    }
  }

  // ── Node builders ──

  _buildAssemblyNode(asm) {
    const nodeId = `assembly:${asm.path || asm.name}`;
    const bodyNames = asm.bodies || [];
    const subAsms = asm.sub_assemblies || [];
    const hasChildren = bodyNames.length > 0 || subAsms.length > 0;

    const data = { ...asm, _type: 'assembly' };
    const node = this._createNode(nodeId, `${asm.name}`, 'assembly', data, hasChildren, null, { startExpanded: true });

    if (hasChildren) {
      const childrenEl = node.querySelector('.tree-node-children');

      for (const sub of subAsms) {
        childrenEl.appendChild(this._buildAssemblyNode(sub));
      }

      const asmBodySet = new Set(bodyNames);
      const rootBodies = bodyNames
        .map((n) => this.bodiesByName[n])
        .filter((b) => b && (b.parent === null || !asmBodySet.has(b.parent)));

      for (const body of rootBodies) {
        childrenEl.appendChild(this._buildBodyNode(body, asmBodySet));
      }
    }

    return node;
  }

  /** Humanize a joint name: "left_wheel" → "Left Wheel" */
  _humanizeJointName(name: string): string {
    return name
      .split('_')
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');
  }

  /**
   * Auto-generate an assembly tree from kinematic structure when no
   * manifest.assemblies are defined. Bodies never contain sub-bodies;
   * each joint creates a sub-assembly.
   */
  _buildAutoAssemblyFromKinematics(rootBody) {
    const botName = this.manifest.bot_name;
    const asmLabel = `${this._humanizeJointName(botName)} Assembly`;
    const nodeId = `assembly:${botName}`;
    const data = { name: asmLabel, _type: 'assembly' };

    const childJoints = this.childJointsOf[rootBody.name] || [];
    const hasChildren = true; // root body always has at least the body node

    const node = this._createNode(nodeId, asmLabel, 'assembly', data, hasChildren, null, {
      startExpanded: true,
    });
    const childrenEl = node.querySelector('.tree-node-children');

    // Root body (flat, no recursive children)
    childrenEl.appendChild(this._buildFlatBodyNode(rootBody));

    // Mounted components on root body
    this._appendMountedComponents(rootBody, childrenEl);

    // Sub-assemblies from joints
    for (const joint of childJoints) {
      childrenEl.appendChild(this._buildJointAssemblyNode(joint));
    }

    // Wire group for root body
    this._appendWireGroup(rootBody, childrenEl);

    return node;
  }

  /** Build a flat body node with no joint/servo children (those go in sub-assemblies). */
  _buildFlatBodyNode(body) {
    const nodeId = `body:${body.name}`;
    const data = { ...body, _type: 'body' };
    const shapescriptUrl =
      body.kind === 'fabricated'
        ? `?cadsteps=${encodeURIComponent(this.manifest.bot_name)}:${encodeURIComponent(body.name)}&from=${encodeURIComponent(this.manifest.bot_name)}`
        : null;

    return this._createNode(nodeId, body.name, 'body', data, false, shapescriptUrl, {
      bodyName: body.name,
    });
  }

  /** Append mounted component nodes (non-servo, non-horn, non-fastener, non-wire, non-joint) to a parent element. */
  _appendMountedComponents(body, parentEl) {
    const parts = this.partsByBody[body.name] || [];

    const mountedComps = parts.filter(
      (p) =>
        p.category !== 'servo' &&
        p.category !== 'horn' &&
        p.category !== 'fastener' &&
        p.category !== 'wire' &&
        !p.joint,
    );
    const mountFasteners = parts.filter((p) => p.category === 'fastener' && !p.joint);

    for (const comp of mountedComps) {
      const compNodeId = `part:${comp.id}`;
      const iconType = comp.category || 'mount';
      const compData = { ...comp, _type: 'part' };
      const compSsUrl = comp.shapescript_component
        ? `?cadsteps=component:${encodeURIComponent(comp.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
        : null;

      const compFasteners = mountFasteners.filter((f) => f.mount_label === comp.mount_label);
      const hasCompChildren = compFasteners.length > 0;

      const compNode = this._createNode(compNodeId, comp.name, iconType, compData, hasCompChildren, compSsUrl, {
        startExpanded: false,
        bodyName: body.name,
      });
      if (hasCompChildren) {
        const compChildrenEl = compNode.querySelector('.tree-node-children');
        const grouped = this._groupFasteners(compFasteners);
        for (const group of grouped) {
          const fNodeId = `fastener-group:${comp.id}:${group.key}`;
          const fData = { ...group, _type: 'fastener-group' };
          const fNode = this._createNode(fNodeId, group.label, 'fastener', fData, false, null, {
            bodyName: body.name,
          });
          compChildrenEl.appendChild(fNode);
        }
      }
      parentEl.appendChild(compNode);
    }
  }

  /** Append wire group node for a body to the parent element. */
  _appendWireGroup(body, parentEl) {
    const parts = this.partsByBody[body.name] || [];
    const wires = parts.filter((p) => p.category === 'wire');
    if (wires.length > 0) {
      const wireGroupId = `wire-group:${body.name}`;
      const wireGroupData = { wires, _type: 'wire-group', parent_body: body.name };
      const wireGroupNode = this._createNode(wireGroupId, `Wires`, 'wire', wireGroupData, true, null, {
        startExpanded: false,
        countBadge: `${wires.length} segments`,
        bodyName: body.name,
      });

      const wireChildrenEl = wireGroupNode.querySelector('.tree-node-children');
      for (const wire of wires) {
        const wireNodeId = `part:${wire.id}`;
        const wireData = { ...wire, _type: 'part' };
        const wireNode = this._createNode(wireNodeId, wire.name, 'wire', wireData, false, null, {
          bodyName: body.name,
        });
        wireChildrenEl.appendChild(wireNode);
      }
      parentEl.appendChild(wireGroupNode);
    }
  }

  /** Build a sub-assembly from a joint: servo + joint node + child body + child components + recurse. */
  _buildJointAssemblyNode(joint) {
    const asmId = `assembly:joint:${joint.name}`;
    const asmLabel = this._humanizeJointName(joint.name);
    const data = { ...joint, _type: 'assembly' };

    const node = this._createNode(asmId, asmLabel, 'assembly', data, true, null, {
      startExpanded: true,
    });
    const childrenEl = node.querySelector('.tree-node-children');

    // Servo components with horn/fastener sub-parts
    const jointParts = this.partsByJoint[joint.name] || [];
    const servos = jointParts.filter((p) => p.category === 'servo');
    const horns = jointParts.filter((p) => p.category === 'horn');
    const fasteners = jointParts.filter((p) => p.category === 'fastener');

    for (const servo of servos) {
      const servoNodeId = `part:${servo.id}`;
      const servoLabel = `${servo.name} @ ${joint.name}`;
      const servoData = { ...servo, _type: 'part', servo_specs: joint.servo_specs };
      const servoSsUrl = servo.shapescript_component
        ? `?cadsteps=component:${encodeURIComponent(servo.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
        : null;
      const servoHasChildren = horns.length > 0 || fasteners.length > 0;
      const servoNode = this._createNode(servoNodeId, servoLabel, 'servo', servoData, servoHasChildren, servoSsUrl, {
        startExpanded: true,
        bodyName: joint.parent_body,
      });

      if (servoHasChildren) {
        const servoChildrenEl = servoNode.querySelector('.tree-node-children');

        for (const horn of horns) {
          const hornNodeId = `part:${horn.id}`;
          const hornData = { ...horn, _type: 'part' };
          const hornSsUrl = horn.shapescript_component
            ? `?cadsteps=component:${encodeURIComponent(horn.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
            : null;
          const hornNode = this._createNode(hornNodeId, horn.name, 'horn', hornData, false, hornSsUrl, {
            bodyName: joint.parent_body,
          });
          servoChildrenEl.appendChild(hornNode);
        }

        const grouped = this._groupFasteners(fasteners);
        for (const group of grouped) {
          const fNodeId = `fastener-group:${joint.name}:${group.key}`;
          const fData = { ...group, _type: 'fastener-group' };
          const fNode = this._createNode(fNodeId, group.label, 'fastener', fData, false, null, {
            bodyName: joint.parent_body,
          });
          servoChildrenEl.appendChild(fNode);
        }
      }

      childrenEl.appendChild(servoNode);
    }

    // Joint node
    const jointNodeId = `joint:${joint.name}`;
    const jointData = { ...joint, _type: 'joint' };
    const jointNode = this._createNode(jointNodeId, joint.name, 'joint', jointData, false, null, {
      bodyName: joint.parent_body,
    });
    childrenEl.appendChild(jointNode);

    // Child body + its components
    const childBody = this.bodiesByName[joint.child_body];
    if (childBody) {
      childrenEl.appendChild(this._buildFlatBodyNode(childBody));
      this._appendMountedComponents(childBody, childrenEl);

      // Recurse: joints from the child body create nested sub-assemblies
      const childJoints = this.childJointsOf[childBody.name] || [];
      for (const childJoint of childJoints) {
        childrenEl.appendChild(this._buildJointAssemblyNode(childJoint));
      }

      this._appendWireGroup(childBody, childrenEl);
    }

    return node;
  }

  _buildBodyNode(body, asmScope = null) {
    const nodeId = `body:${body.name}`;
    const parts = this.partsByBody[body.name] || [];
    const childJoints = this.childJointsOf[body.name] || [];

    const mountedComps = parts.filter(
      (p) =>
        p.category !== 'servo' &&
        p.category !== 'horn' &&
        p.category !== 'fastener' &&
        p.category !== 'wire' &&
        !p.joint,
    );
    const wires = parts.filter((p) => p.category === 'wire');

    const servosByJoint = {};
    for (const p of parts) {
      if (p.joint && (p.category === 'servo' || p.category === 'horn' || p.category === 'fastener')) {
        if (!servosByJoint[p.joint]) servosByJoint[p.joint] = [];
        servosByJoint[p.joint].push(p);
      }
    }

    const mountFasteners = parts.filter((p) => p.category === 'fastener' && !p.joint);

    const hasChildren =
      mountedComps.length > 0 ||
      Object.keys(servosByJoint).length > 0 ||
      wires.length > 0 ||
      childJoints.length > 0 ||
      mountFasteners.length > 0;

    const labelText = body.name;
    const data = { ...body, _type: 'body' };
    const shapescriptUrl =
      body.kind === 'fabricated'
        ? `?cadsteps=${encodeURIComponent(this.manifest.bot_name)}:${encodeURIComponent(body.name)}&from=${encodeURIComponent(this.manifest.bot_name)}`
        : null;

    const node = this._createNode(nodeId, labelText, 'body', data, hasChildren, shapescriptUrl, {
      startExpanded: true,
      bodyName: body.name,
    });

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

        const compFasteners = mountFasteners.filter((f) => f.mount_label === comp.mount_label);
        const hasCompChildren = compFasteners.length > 0;

        const compNode = this._createNode(compNodeId, comp.name, iconType, compData, hasCompChildren, compSsUrl, {
          startExpanded: false,
          bodyName: body.name,
        });
        if (hasCompChildren) {
          const compChildrenEl = compNode.querySelector('.tree-node-children');
          const grouped = this._groupFasteners(compFasteners);
          for (const group of grouped) {
            const fNodeId = `fastener-group:${comp.id}:${group.key}`;
            const fData = { ...group, _type: 'fastener-group' };
            const fNode = this._createNode(fNodeId, group.label, 'fastener', fData, false, null, {
              bodyName: body.name,
            });
            compChildrenEl.appendChild(fNode);
          }
        }
        childrenEl.appendChild(compNode);
      }

      // 2. Servo groups per joint
      for (const joint of childJoints) {
        const jointParts = servosByJoint[joint.name] || [];
        const servos = jointParts.filter((p) => p.category === 'servo');
        const horns = jointParts.filter((p) => p.category === 'horn');
        const fasteners = jointParts.filter((p) => p.category === 'fastener');

        for (const servo of servos) {
          const servoNodeId = `part:${servo.id}`;
          const servoLabel = `${servo.name} @ ${joint.name}`;
          const servoData = { ...servo, _type: 'part', servo_specs: joint.servo_specs };
          const servoSsUrl = servo.shapescript_component
            ? `?cadsteps=component:${encodeURIComponent(servo.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
            : null;
          const servoHasChildren = horns.length > 0 || fasteners.length > 0;
          const servoNode = this._createNode(
            servoNodeId,
            servoLabel,
            'servo',
            servoData,
            servoHasChildren,
            servoSsUrl,
            { startExpanded: true, bodyName: body.name },
          );

          if (servoHasChildren) {
            const servoChildrenEl = servoNode.querySelector('.tree-node-children');

            for (const horn of horns) {
              const hornNodeId = `part:${horn.id}`;
              const hornData = { ...horn, _type: 'part' };
              const hornSsUrl = horn.shapescript_component
                ? `?cadsteps=component:${encodeURIComponent(horn.shapescript_component)}&from=${encodeURIComponent(this.manifest.bot_name)}`
                : null;
              const hornNode = this._createNode(hornNodeId, horn.name, 'horn', hornData, false, hornSsUrl, {
                bodyName: body.name,
              });
              servoChildrenEl.appendChild(hornNode);
            }

            const grouped = this._groupFasteners(fasteners);
            for (const group of grouped) {
              const fNodeId = `fastener-group:${joint.name}:${group.key}`;
              const fData = { ...group, _type: 'fastener-group' };
              const fNode = this._createNode(fNodeId, group.label, 'fastener', fData, false, null, {
                bodyName: body.name,
              });
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
        const jointNode = this._createNode(jointNodeId, `${joint.name}`, 'joint', jointData, hasJointChild, null, {
          startExpanded: true,
          bodyName: body.name,
        });

        if (hasJointChild) {
          const jointChildrenEl = jointNode.querySelector('.tree-node-children');
          jointChildrenEl.appendChild(this._buildBodyNode(childBody, asmScope));
        }

        childrenEl.appendChild(jointNode);
      }

      // 3. Wires (collapsible group, collapsed by default)
      if (wires.length > 0) {
        const wireGroupId = `wire-group:${body.name}`;
        const wireGroupData = { wires, _type: 'wire-group', parent_body: body.name };
        const wireGroupNode = this._createNode(wireGroupId, `Wires`, 'wire', wireGroupData, true, null, {
          startExpanded: false,
          countBadge: wires.length,
          bodyName: body.name,
        });

        const wireChildrenEl = wireGroupNode.querySelector('.tree-node-children');
        for (const wire of wires) {
          const wireNodeId = `part:${wire.id}`;
          const wireData = { ...wire, _type: 'part' };
          const wireNode = this._createNode(wireNodeId, wire.name, 'wire', wireData, false, null, {
            bodyName: body.name,
          });
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
  _groupFasteners(fasteners: any[]): any[] {
    const groups: Record<string, any> = {};
    for (const f of fasteners) {
      const key = f.name;
      if (!groups[key]) groups[key] = { key, name: f.name, count: 0, items: [] };
      groups[key].count++;
      groups[key].items.push(f);
    }
    return Object.values(groups).map((g: any) => ({
      ...g,
      label: g.count > 1 ? `${g.count}\u00d7 ${g.name}` : g.name,
    }));
  }

  _createNode(
    nodeId: string,
    label: string,
    iconType: string,
    data: any,
    hasChildren: boolean,
    shapescriptUrl: string | null = null,
    opts: any = {},
  ) {
    const { startExpanded = true, countBadge = null, bodyName = null } = opts;
    const category = iconTypeToCategory(iconType);

    const node = document.createElement('div');
    node.className = 'tree-node';
    node.dataset.nodeId = nodeId;
    node.dataset.category = category;
    if (bodyName) node.dataset.bodyName = bodyName;

    const header = document.createElement('div');
    header.className = 'tree-node-header';

    // Chevron (SVG disclosure triangle) or spacer for alignment
    if (hasChildren) {
      const chevron = document.createElement('span');
      chevron.className = 'tree-chevron';
      chevron.innerHTML = CHEVRON_RIGHT;
      chevron.classList.toggle('expanded', startExpanded);
      const toggleFn = (e: Event) => {
        e.stopPropagation();
        const children = node.querySelector(':scope > .tree-node-children') as HTMLElement | null;
        if (!children) return;
        const expanded = children.style.display !== 'none';
        children.style.display = expanded ? 'none' : 'block';
        chevron.classList.toggle('expanded', !expanded);
      };
      chevron.addEventListener('click', toggleFn);
      header.appendChild(chevron);
    } else {
      // Spacer matching chevron width so siblings align regardless of children
      const spacer = document.createElement('span');
      spacer.className = 'tree-chevron-spacer';
      header.appendChild(spacer);
    }

    // Category color dot
    const dot = document.createElement('span');
    dot.className = 'tree-cat-dot';
    const iconDef = ICONS[iconType] || ICONS.component;
    dot.style.background = iconDef.color;
    header.appendChild(dot);

    // Label
    const labelEl = document.createElement('span');
    labelEl.className = 'tree-node-label';
    labelEl.textContent = label;
    header.appendChild(labelEl);

    // Count badge
    if (countBadge !== null) {
      const badge = document.createElement('span');
      badge.className = 'tree-count-badge';
      badge.textContent = countBadge;
      header.appendChild(badge);
    }

    // ShapeScript code icon (visible on hover, before vis-dot)
    if (shapescriptUrl) {
      const codeBtn = document.createElement('span');
      codeBtn.className = 'tree-code-icon';
      codeBtn.textContent = '</>';
      codeBtn.title = 'View in ShapeScript';
      codeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.onShapeScript) this.onShapeScript(shapescriptUrl);
        else window.location.href = shapescriptUrl;
      });
      header.appendChild(codeBtn);
    }

    // Visibility dot indicator (always last, right-aligned via margin-left:auto)
    {
      const visDot = document.createElement('span');
      visDot.className = 'vis-dot';
      visDot.title = 'Toggle visibility';
      visDot.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.onToggleNodeHidden) this.onToggleNodeHidden(nodeId);
      });
      header.appendChild(visDot);

      // Right-click to solo
      header.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.onSolo) this.onSolo(nodeId);
      });
    }

    // Click header to select (also toggles if has children)
    header.addEventListener('click', () => {
      this.setFocused(nodeId);
      this.onSelect(nodeId, data);
    });

    // Click header text to toggle children too
    if (hasChildren) {
      labelEl.addEventListener('click', (e) => {
        // Only toggle if the label itself is clicked, not propagated from a child
        e.stopPropagation();
        const children = node.querySelector(':scope > .tree-node-children') as HTMLElement | null;
        if (!children) return;
        const expanded = children.style.display !== 'none';
        children.style.display = expanded ? 'none' : 'block';
        const chevronEl = node.querySelector(':scope > .tree-node-header .tree-chevron');
        if (chevronEl) chevronEl.classList.toggle('expanded', !expanded);
        this.setFocused(nodeId);
        this.onSelect(nodeId, data);
      });
    }

    node.appendChild(header);

    // Children container
    if (hasChildren) {
      const children = document.createElement('div');
      children.className = 'tree-node-children';
      children.style.display = startExpanded ? 'block' : 'none';
      node.appendChild(children);
    }

    return node;
  }

  setFocused(nodeId) {
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

  // ── Inject scoped CSS ──

  _injectStyles() {
    if (document.getElementById('component-tree-styles')) return;
    const style = document.createElement('style');
    style.id = 'component-tree-styles';
    style.textContent = `
      /* ── Tree toolbar ── */
      .tree-toolbar {
        position: sticky; top: 0; z-index: 10;
        background: rgba(255,255,255,0.96);
        padding-bottom: 8px; margin-bottom: 4px;
        border-bottom: 1px solid var(--border);
      }

      .tree-search-wrap {
        position: relative;
      }
      .tree-search {
        width: 100%; height: 28px;
        padding: 0 28px 0 8px;
        border: 1px solid var(--border); border-radius: var(--radius-sm);
        font-family: var(--font); font-size: 12px; color: var(--foreground);
        background: var(--card); outline: none;
        transition: border-color 0.15s;
      }
      .tree-search:focus { border-color: var(--primary); }
      .tree-search::placeholder { color: var(--gray3); }
      .tree-search-clear {
        position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
        width: 18px; height: 18px; line-height: 18px; text-align: center;
        font-size: 14px; color: var(--gray3); cursor: pointer;
        border-radius: 50%; transition: background 0.1s, color 0.1s;
      }
      .tree-search-clear:hover { background: var(--secondary); color: var(--foreground); }

      /* ── Category filter chips ── */
      .tree-filter-bar {
        display: flex; flex-wrap: wrap; gap: 4px;
      }
      .tree-filter-chip {
        display: inline-flex; align-items: center; gap: 4px;
        height: 22px; padding: 0 8px;
        border: 1px solid var(--border); border-radius: 11px;
        font-family: var(--font); font-size: 10px; font-weight: 500;
        color: var(--muted-fg); background: var(--card);
        cursor: pointer; transition: all 0.15s; outline: none;
        user-select: none;
      }
      .tree-filter-chip:hover { border-color: var(--gray3); }
      .tree-filter-chip.active {
        background: rgba(19,124,189,0.06); border-color: var(--gray5);
        color: var(--foreground);
      }
      .tree-filter-chip:not(.active) .tree-filter-dot { opacity: 0.3; }
      .tree-filter-dot {
        width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
      }

      /* ── Tree nodes ── */
      .tree-node { }
      .tree-node.search-hidden { display: none; }
      /* Node opacity controlled inline by updateFromDesignScene */

      .tree-node-header {
        display: flex; align-items: center; gap: 4px;
        padding: 3px 28px 3px 6px; border-radius: 4px; cursor: pointer;
        font-size: 12px; color: var(--dark5);
        transition: background 0.1s;
        position: relative;
      }
      .tree-node-header:hover { background: rgba(19,124,189,0.08); }
      .tree-node.focused > .tree-node-header {
        background: rgba(19,124,189,0.12);
        color: var(--accent);
        border-left: 3px solid var(--primary);
        padding-left: 3px;
      }

      /* Disclosure triangle (SVG chevron) */
      .tree-chevron {
        width: 14px; height: 14px;
        display: inline-flex; align-items: center; justify-content: center;
        flex-shrink: 0; cursor: pointer;
        color: var(--gray3);
        transition: transform 0.15s ease;
      }
      .tree-chevron.expanded {
        transform: rotate(90deg);
      }
      .tree-chevron:hover { color: var(--gray1); }
      .tree-chevron-spacer {
        width: 14px; height: 14px; flex-shrink: 0;
        display: inline-flex;
      }

      /* Category color dot */
      .tree-cat-dot {
        width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
      }

      /* Label */
      .tree-node-label {
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        flex: 1; min-width: 0;
      }

      /* Count badge */
      .tree-count-badge {
        font-size: 10px; color: var(--gray3);
        background: rgba(206,217,224,0.3);
        padding: 0 5px; border-radius: 8px;
        line-height: 16px; flex-shrink: 0;
      }

      /* ── Visibility dot indicator (absolutely positioned at right edge) ── */
      .vis-dot {
        width: 12px; height: 12px; border-radius: 50%;
        position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
        cursor: pointer; background: #58a6ff;
        box-sizing: border-box;
        transition: background 0.15s, border 0.15s;
      }
      .vis-dot:hover { opacity: 0.7; }

      /* Code icon (right-aligned, hover-visible) */
      .tree-code-icon {
        flex-shrink: 0;
        font-size: 10px; font-family: var(--font-mono);
        color: var(--gray3); cursor: pointer;
        opacity: 0; transition: opacity 0.15s;
        padding: 0 2px;
      }
      .tree-node-header:hover .tree-code-icon { opacity: 1; }
      .tree-code-icon:hover { color: var(--primary); }

      /* Children indentation with vertical guide line */
      .tree-node-children {
        padding-left: 16px;
        position: relative;
      }
      .tree-node-children::before {
        content: '';
        position: absolute; left: 7px; top: 0; bottom: 0;
        width: 1px; background: rgba(206,217,224,0.5);
      }

      /* Remove icon from old style (replaced by dot) */
      .tree-node-icon { display: none; }
    `;
    document.head.appendChild(style);
  }
}
