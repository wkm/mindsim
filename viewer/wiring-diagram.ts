/**
 * Wiring Diagram — transforms WireNet[] into an elkjs graph, layouts it,
 * and renders to SVG.
 *
 * Uses elkjs directly for layout and hand-rendered SVG (same pattern as
 * assembly-dag.ts). Avoids the sprotty DI/inversify stack for simplicity.
 *
 * Entry point: new WiringDiagram(container, callbacks)
 */

import ElkConstructor from 'elkjs/lib/elk.bundled';
import type { ElkExtendedEdge, ElkNode, ElkPort } from 'elkjs/lib/elk-api';

// ---------------------------------------------------------------------------
// WireNet types (matches API response from GET /api/bots/{bot}/wirenets)
// ---------------------------------------------------------------------------

export interface NetPort {
  component_id: string;
  port_label: string;
}

export interface WireNet {
  label: string;
  bus_type: string;
  topology: 'daisy_chain' | 'point_to_point' | 'star';
  ports: NetPort[];
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

export interface WiringDiagramCallbacks {
  onNodeClick?: (nodeId: string) => void;
  onEdgeClick?: (edgeId: string, net: WireNet) => void;
}

// ---------------------------------------------------------------------------
// Bus type colors — matches _BUS_TYPE_COLORS in viewer.py
// ---------------------------------------------------------------------------

const BUS_TYPE_COLORS: Record<string, string> = {
  uart_half_duplex: 'rgb(51, 153, 219)',
  csi: 'rgb(102, 186, 107)',
  power: 'rgb(230, 77, 64)',
  usb: 'rgb(230, 192, 87)',
};

const BUS_COLOR_DEFAULT = 'rgb(135, 135, 135)';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const NODE_W = 140;
const PORT_H = 10;
const PORT_W = 10;
const NODE_HEADER_H = 28;
const PORT_ROW_H = 22;
const NODE_PAD_BOTTOM = 8;
const SVG_NS = 'http://www.w3.org/2000/svg';

// ---------------------------------------------------------------------------
// Internal graph model (pre-layout)
// ---------------------------------------------------------------------------

interface GraphNode {
  id: string;
  label: string;
  ports: GraphPort[];
  width: number;
  height: number;
}

interface GraphPort {
  id: string;
  label: string;
  busType: string;
  orphaned: boolean;
  side: 'WEST' | 'EAST'; // WEST = input (left), EAST = output (right)
}

interface GraphEdge {
  id: string;
  sourcePortId: string;
  targetPortId: string;
  busType: string;
  netLabel: string;
}

// ---------------------------------------------------------------------------
// Pure transform: WireNet[] -> graph model
// ---------------------------------------------------------------------------

function buildGraphModel(nets: WireNet[]): { nodes: GraphNode[]; edges: GraphEdge[] } {
  // Collect all unique components and their ports
  const componentPorts: Map<string, Map<string, string>> = new Map(); // component_id -> (port_label -> bus_type)

  for (const net of nets) {
    for (const port of net.ports) {
      if (!componentPorts.has(port.component_id)) {
        componentPorts.set(port.component_id, new Map());
      }
      componentPorts.get(port.component_id)!.set(port.port_label, net.bus_type);
    }
  }

  // Build edges based on topology (before nodes, so we can determine port sides)
  const edges: GraphEdge[] = [];

  for (const net of nets) {
    const portIds = net.ports.map((p) => `${p.component_id}:${p.port_label}`);

    if (net.topology === 'daisy_chain') {
      // Skip edges between ports on the same component — those represent
      // internal pass-through, not a physical wire between components.
      for (let i = 0; i < portIds.length - 1; i++) {
        const srcComp = net.ports[i].component_id;
        const tgtComp = net.ports[i + 1].component_id;
        if (srcComp === tgtComp) continue;
        edges.push({
          id: `${net.label}:${i}`,
          sourcePortId: portIds[i],
          targetPortId: portIds[i + 1],
          busType: net.bus_type,
          netLabel: net.label,
        });
      }
    } else if (net.topology === 'point_to_point') {
      if (portIds.length >= 2) {
        edges.push({
          id: `${net.label}:0`,
          sourcePortId: portIds[0],
          targetPortId: portIds[1],
          busType: net.bus_type,
          netLabel: net.label,
        });
      }
    } else if (net.topology === 'star') {
      for (let i = 1; i < portIds.length; i++) {
        edges.push({
          id: `${net.label}:${i - 1}`,
          sourcePortId: portIds[0],
          targetPortId: portIds[i],
          busType: net.bus_type,
          netLabel: net.label,
        });
      }
    }
  }

  // Determine port sides: edge sources → EAST (output, right), edge targets → WEST (input, left)
  const sourcePorts = new Set<string>();
  const targetPorts = new Set<string>();
  for (const edge of edges) {
    sourcePorts.add(edge.sourcePortId);
    targetPorts.add(edge.targetPortId);
  }

  // Build nodes
  const nodes: GraphNode[] = [];
  for (const [compId, ports] of componentPorts) {
    const westPorts: GraphPort[] = [];
    const eastPorts: GraphPort[] = [];
    for (const [portLabel, busType] of ports) {
      const portId = `${compId}:${portLabel}`;
      const isSource = sourcePorts.has(portId);
      const isTarget = targetPorts.has(portId);
      // If only a source → output (right). If only a target → input (left).
      // If both or neither, use naming convention.
      let side: 'WEST' | 'EAST';
      if (isTarget && !isSource) {
        side = 'WEST';
      } else if (isSource && !isTarget) {
        side = 'EAST';
      } else {
        // Ambiguous — use naming: labels ending in _in/input → left, else right
        side = /_in$|_input$|^power_in$|^usb_power$|^csi_in$/.test(portLabel) ? 'WEST' : 'EAST';
      }
      const gp: GraphPort = { id: portId, label: portLabel, busType, orphaned: false, side };
      if (side === 'WEST') {
        westPorts.push(gp);
      } else {
        eastPorts.push(gp);
      }
    }
    // Interleave: west ports first, then east ports (for consistent vertical ordering)
    const portList = [...westPorts, ...eastPorts];
    const maxSidePorts = Math.max(westPorts.length, eastPorts.length);
    const height = NODE_HEADER_H + maxSidePorts * PORT_ROW_H + NODE_PAD_BOTTOM;
    nodes.push({
      id: compId,
      label: compId,
      ports: portList,
      width: NODE_W,
      height,
    });
  }

  // Orphaned port detection: ports not referenced by any edge
  for (const node of nodes) {
    for (const port of node.ports) {
      if (!sourcePorts.has(port.id) && !targetPorts.has(port.id)) {
        port.orphaned = true;
      }
    }
  }

  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// Convert graph model to elkjs input
// ---------------------------------------------------------------------------

function toElkGraph(nodes: GraphNode[], edges: GraphEdge[]): ElkNode {
  const elkNodes: ElkNode[] = nodes.map((node) => {
    // Separate ports by side for independent vertical spacing
    const westPorts = node.ports.filter((p) => p.side === 'WEST');
    const eastPorts = node.ports.filter((p) => p.side === 'EAST');

    const elkPorts: ElkPort[] = node.ports.map((port) => {
      const sameSide = port.side === 'WEST' ? westPorts : eastPorts;
      const sideIdx = sameSide.indexOf(port);
      const portX = port.side === 'WEST' ? 0 : node.width - PORT_W;
      const portY = NODE_HEADER_H + sideIdx * PORT_ROW_H + (PORT_ROW_H - PORT_H) / 2;
      return {
        id: port.id,
        width: PORT_W,
        height: PORT_H,
        x: portX,
        y: portY,
        layoutOptions: {
          'port.side': port.side,
        },
      };
    });
    return {
      id: node.id,
      width: node.width,
      height: node.height,
      ports: elkPorts,
      layoutOptions: {
        portConstraints: 'FIXED_POS',
      },
    };
  });

  const elkEdges: ElkExtendedEdge[] = edges.map((edge) => ({
    id: edge.id,
    sources: [edge.sourcePortId],
    targets: [edge.targetPortId],
  }));

  return {
    id: 'root',
    children: elkNodes,
    edges: elkEdges,
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.spacing.nodeNode': '40',
      'elk.layered.spacing.nodeNodeBetweenLayers': '60',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.layered.mergeEdges': 'false',
    },
  };
}

// ---------------------------------------------------------------------------
// SVG rendering of laid-out graph
// ---------------------------------------------------------------------------

interface LayoutResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  elkResult: ElkNode;
  nets: WireNet[];
}

function renderSVG(layout: LayoutResult, callbacks: WiringDiagramCallbacks): SVGSVGElement {
  const { nodes, edges, elkResult, nets } = layout;

  // Build lookup from elkResult
  const elkNodeMap = new Map<string, ElkNode>();
  for (const child of elkResult.children ?? []) {
    elkNodeMap.set(child.id, child);
  }

  const elkEdgeMap = new Map<string, ElkExtendedEdge>();
  for (const edge of (elkResult.edges ?? []) as ElkExtendedEdge[]) {
    elkEdgeMap.set(edge.id, edge);
  }

  // Net label -> WireNet lookup for click callbacks
  const netByLabel = new Map<string, WireNet>();
  for (const net of nets) {
    netByLabel.set(net.label, net);
  }

  const svgW = (elkResult.width ?? 600) + 40;
  const svgH = (elkResult.height ?? 400) + 40;

  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('width', String(svgW));
  svg.setAttribute('height', String(svgH));
  svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);
  svg.style.cssText = 'display: block; margin: 0 auto;';

  // Offset everything by a small margin
  const offsetG = document.createElementNS(SVG_NS, 'g');
  offsetG.setAttribute('transform', 'translate(20, 20)');
  svg.appendChild(offsetG);

  // --- Edges ---
  const edgeGroup = document.createElementNS(SVG_NS, 'g');
  edgeGroup.setAttribute('class', 'wiring-edges');

  for (const edge of edges) {
    const elkEdge = elkEdgeMap.get(edge.id);
    if (!elkEdge) continue;

    const color = BUS_TYPE_COLORS[edge.busType] ?? BUS_COLOR_DEFAULT;
    const sections = elkEdge.sections ?? [];

    for (const section of sections) {
      const points: { x: number; y: number }[] = [];
      points.push(section.startPoint);
      if (section.bendPoints) {
        points.push(...section.bendPoints);
      }
      points.push(section.endPoint);

      const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

      const path = document.createElementNS(SVG_NS, 'path');
      path.setAttribute('d', d);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '2');
      path.setAttribute('data-edge-id', edge.id);
      path.setAttribute('data-bus-type', edge.busType);
      path.style.cursor = 'pointer';

      path.addEventListener('click', (e) => {
        e.stopPropagation();
        const net = netByLabel.get(edge.netLabel);
        if (net && callbacks.onEdgeClick) {
          callbacks.onEdgeClick(edge.id, net);
        }
      });

      // Hover effect
      path.addEventListener('mouseenter', () => {
        path.setAttribute('stroke-width', '4');
        path.setAttribute('stroke-opacity', '1');
      });
      path.addEventListener('mouseleave', () => {
        path.setAttribute('stroke-width', '2');
        path.removeAttribute('stroke-opacity');
      });

      edgeGroup.appendChild(path);
    }
  }
  offsetG.appendChild(edgeGroup);

  // --- Nodes ---
  const nodeGroup = document.createElementNS(SVG_NS, 'g');
  nodeGroup.setAttribute('class', 'wiring-nodes');

  for (const node of nodes) {
    const elkNode = elkNodeMap.get(node.id);
    if (!elkNode) continue;

    const nx = elkNode.x ?? 0;
    const ny = elkNode.y ?? 0;

    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('transform', `translate(${nx}, ${ny})`);
    g.setAttribute('data-node-id', node.id);
    g.style.cursor = 'pointer';

    // Node background
    const rect = document.createElementNS(SVG_NS, 'rect');
    rect.setAttribute('width', String(node.width));
    rect.setAttribute('height', String(node.height));
    rect.setAttribute('rx', '4');
    rect.setAttribute('ry', '4');
    rect.setAttribute('fill', '#1a1a2e');
    rect.setAttribute('stroke', '#4a4a6a');
    rect.setAttribute('stroke-width', '1');

    g.appendChild(rect);

    // Node label
    const label = document.createElementNS(SVG_NS, 'text');
    label.setAttribute('x', '8');
    label.setAttribute('y', '18');
    label.setAttribute('font-size', '12');
    label.setAttribute('font-weight', '600');
    label.setAttribute('font-family', 'var(--font, system-ui)');
    label.setAttribute('fill', '#e0e0e0');
    label.textContent = node.label;
    g.appendChild(label);

    // Header divider
    const divider = document.createElementNS(SVG_NS, 'line');
    divider.setAttribute('x1', '0');
    divider.setAttribute('y1', String(NODE_HEADER_H));
    divider.setAttribute('x2', String(node.width));
    divider.setAttribute('y2', String(NODE_HEADER_H));
    divider.setAttribute('stroke', '#4a4a6a');
    divider.setAttribute('stroke-width', '0.5');
    g.appendChild(divider);

    // Ports — west (input) on left, east (output) on right
    const westPorts = node.ports.filter((p) => p.side === 'WEST');
    const eastPorts = node.ports.filter((p) => p.side === 'EAST');

    for (const [sidePorts, side] of [
      [westPorts, 'WEST'],
      [eastPorts, 'EAST'],
    ] as const) {
      for (let si = 0; si < sidePorts.length; si++) {
        const port = sidePorts[si];
        const portY = NODE_HEADER_H + si * PORT_ROW_H + (PORT_ROW_H - PORT_H) / 2;
        const isWest = side === 'WEST';

        // Port dot
        const portRect = document.createElementNS(SVG_NS, 'rect');
        const dotX = isWest ? 4 : node.width - PORT_W - 4;
        portRect.setAttribute('x', String(dotX));
        portRect.setAttribute('y', String(portY));
        portRect.setAttribute('width', String(PORT_W));
        portRect.setAttribute('height', String(PORT_H));
        portRect.setAttribute('rx', '2');

        const portColor = BUS_TYPE_COLORS[port.busType] ?? '#666';

        if (port.orphaned) {
          portRect.setAttribute('fill', 'none');
          portRect.setAttribute('stroke', 'orange');
          portRect.setAttribute('stroke-width', '2');
        } else {
          portRect.setAttribute('fill', portColor);
        }

        g.appendChild(portRect);

        // Port label — next to the dot, inward from the edge
        const portLabel = document.createElementNS(SVG_NS, 'text');
        const labelX = isWest ? PORT_W + 8 : node.width - PORT_W - 8;
        portLabel.setAttribute('x', String(labelX));
        portLabel.setAttribute('y', String(portY + PORT_H - 1));
        portLabel.setAttribute('font-size', '10');
        portLabel.setAttribute('font-family', 'var(--font-mono, monospace)');
        portLabel.setAttribute('fill', '#999');
        if (!isWest) {
          portLabel.setAttribute('text-anchor', 'end');
        }
        const maxLabelLen = 14;
        portLabel.textContent = port.label.length > maxLabelLen ? `${port.label.slice(0, maxLabelLen)}...` : port.label;
        g.appendChild(portLabel);
      }
    }

    g.addEventListener('click', (e) => {
      e.stopPropagation();
      if (callbacks.onNodeClick) {
        callbacks.onNodeClick(node.id);
      }
    });

    // Hover effect
    g.addEventListener('mouseenter', () => {
      rect.setAttribute('stroke', '#7a7aaa');
      rect.setAttribute('stroke-width', '2');
    });
    g.addEventListener('mouseleave', () => {
      rect.setAttribute('stroke', '#4a4a6a');
      rect.setAttribute('stroke-width', '1');
    });

    nodeGroup.appendChild(g);
  }
  offsetG.appendChild(nodeGroup);

  return svg;
}

// ---------------------------------------------------------------------------
// Bus type legend
// ---------------------------------------------------------------------------

function renderLegend(busTypes: Set<string>): HTMLDivElement {
  const legend = document.createElement('div');
  legend.style.cssText =
    'position: absolute; bottom: 8px; right: 8px; background: rgba(26,26,46,0.9); ' +
    'border: 1px solid #4a4a6a; border-radius: 4px; padding: 6px 10px; font-size: 10px; ' +
    'font-family: var(--font, system-ui); color: #999; display: flex; flex-direction: column; gap: 3px;';

  for (const bt of busTypes) {
    const row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 6px;';

    const swatch = document.createElement('div');
    const color = BUS_TYPE_COLORS[bt] ?? BUS_COLOR_DEFAULT;
    swatch.style.cssText = `width: 16px; height: 3px; background: ${color}; border-radius: 1px;`;
    row.appendChild(swatch);

    const label = document.createElement('span');
    label.textContent = bt.replace(/_/g, ' ');
    row.appendChild(label);

    legend.appendChild(row);
  }

  return legend;
}

// ---------------------------------------------------------------------------
// WiringDiagram class
// ---------------------------------------------------------------------------

export class WiringDiagram {
  private container: HTMLElement;
  private callbacks: WiringDiagramCallbacks;
  private svgEl: SVGSVGElement | null = null;
  private legendEl: HTMLDivElement | null = null;

  constructor(container: HTMLElement, callbacks: WiringDiagramCallbacks = {}) {
    this.container = container;
    this.callbacks = callbacks;
    // Make container position relative for legend absolute positioning
    if (getComputedStyle(container).position === 'static') {
      container.style.position = 'relative';
    }
  }

  async setNets(nets: WireNet[]): Promise<void> {
    // Clear previous
    this.dispose();

    if (nets.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding: 16px; color: var(--muted-fg); font-size: 12px;';
      empty.textContent = 'No wiring nets defined.';
      this.container.appendChild(empty);
      return;
    }

    // Build graph model
    const { nodes, edges } = buildGraphModel(nets);

    if (nodes.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding: 16px; color: var(--muted-fg); font-size: 12px;';
      empty.textContent = 'No components in wiring nets.';
      this.container.appendChild(empty);
      return;
    }

    // Run elkjs layout
    const elk = new ElkConstructor();
    const elkInput = toElkGraph(nodes, edges);
    const elkResult = await elk.layout(elkInput);

    // Render SVG
    const svg = renderSVG({ nodes, edges, elkResult, nets }, this.callbacks);
    this.svgEl = svg;
    this.container.appendChild(svg);

    // Render legend
    const busTypes = new Set<string>();
    for (const net of nets) {
      busTypes.add(net.bus_type);
    }
    if (busTypes.size > 0) {
      this.legendEl = renderLegend(busTypes);
      this.container.appendChild(this.legendEl);
    }
  }

  dispose(): void {
    if (this.svgEl) {
      this.svgEl.remove();
      this.svgEl = null;
    }
    if (this.legendEl) {
      this.legendEl.remove();
      this.legendEl = null;
    }
    // Clear any text-only children (empty state messages)
    this.container.innerHTML = '';
  }
}
