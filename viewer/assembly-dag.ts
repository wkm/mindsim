/**
 * Assembly DAG — prerequisite graph visualization for assembly ops.
 *
 * Renders an SVG directed graph where:
 * - Nodes = assembly ops (colored by action type)
 * - Edges = prerequisite relationships
 * - Layout: topological layers (top-to-bottom)
 *
 * Click a node to scrub to that step.
 */

import type { AssemblyOpData } from './assembly-scrubber.ts';

// ---------------------------------------------------------------------------
// Action type colors
// ---------------------------------------------------------------------------

const ACTION_COLORS: Record<string, string> = {
  insert: '#2B95D6',
  fasten: '#8A9BA8',
  route_wire: '#D9822B',
  connect: '#0F9960',
  articulate: '#9179F2',
};

const ACTION_COLOR_DEFAULT = '#A7B6C2';

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

const NODE_W = 160;
const NODE_H = 36;
const LAYER_GAP_Y = 56;
const NODE_GAP_X = 24;
const PAD_X = 24;
const PAD_Y = 24;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface DagNode {
  step: number;
  action: string;
  description: string;
  prerequisites: number[];
  layer: number;
  x: number;
  y: number;
}

export type DagStepHandler = (step: number) => void;

// ---------------------------------------------------------------------------
// AssemblyDAG class
// ---------------------------------------------------------------------------

export class AssemblyDAG {
  private container: HTMLElement;
  private onStepClick: DagStepHandler;
  private ops: AssemblyOpData[] = [];
  private nodes: DagNode[] = [];
  private svgEl: SVGSVGElement | null = null;
  private highlightedStep = -1;

  constructor(container: HTMLElement, onStepClick: DagStepHandler) {
    this.container = container;
    this.onStepClick = onStepClick;
  }

  setOps(ops: AssemblyOpData[]): void {
    this.ops = ops;
    this.layout();
    this.render();
  }

  highlightStep(step: number): void {
    this.highlightedStep = step;
    this.updateHighlight();
  }

  dispose(): void {
    this.container.innerHTML = '';
    this.svgEl = null;
  }

  // ---------------------------------------------------------------------------
  // Layout: assign layers based on longest path from roots
  // ---------------------------------------------------------------------------

  private layout(): void {
    const stepToIdx: Map<number, number> = new Map();
    for (let i = 0; i < this.ops.length; i++) {
      stepToIdx.set(this.ops[i].step, i);
    }

    // Compute layer (depth) for each op via BFS / dynamic programming
    const layers: number[] = new Array(this.ops.length).fill(0);

    for (let i = 0; i < this.ops.length; i++) {
      const op = this.ops[i];
      const prereqs = op.prerequisites ?? [];
      let maxParentLayer = -1;
      for (const prereq of prereqs) {
        const pi = stepToIdx.get(prereq);
        if (pi !== undefined) {
          maxParentLayer = Math.max(maxParentLayer, layers[pi]);
        }
      }
      layers[i] = maxParentLayer + 1;
    }

    // Group by layer
    const layerGroups: Map<number, number[]> = new Map();
    for (let i = 0; i < this.ops.length; i++) {
      const l = layers[i];
      if (!layerGroups.has(l)) layerGroups.set(l, []);
      layerGroups.get(l)!.push(i);
    }

    // Assign x/y positions
    const maxLayerWidth = Math.max(...[...layerGroups.values()].map((g) => g.length));
    const totalWidth = maxLayerWidth * (NODE_W + NODE_GAP_X) - NODE_GAP_X + PAD_X * 2;

    this.nodes = [];
    for (let i = 0; i < this.ops.length; i++) {
      const op = this.ops[i];
      const layer = layers[i];
      const group = layerGroups.get(layer)!;
      const idxInLayer = group.indexOf(i);
      const layerWidth = group.length * (NODE_W + NODE_GAP_X) - NODE_GAP_X;
      const offsetX = (totalWidth - layerWidth) / 2;

      this.nodes.push({
        step: op.step,
        action: op.action,
        description: op.description,
        prerequisites: op.prerequisites ?? [],
        layer,
        x: offsetX + idxInLayer * (NODE_W + NODE_GAP_X),
        y: PAD_Y + layer * (NODE_H + LAYER_GAP_Y),
      });
    }
  }

  // ---------------------------------------------------------------------------
  // SVG rendering
  // ---------------------------------------------------------------------------

  private render(): void {
    this.container.innerHTML = '';

    if (this.nodes.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding: 16px; color: var(--muted-fg); font-size: 12px;';
      empty.textContent = 'No assembly operations.';
      this.container.appendChild(empty);
      return;
    }

    const maxLayer = Math.max(...this.nodes.map((n) => n.layer));
    const maxLayerWidth = Math.max(...this.nodes.map((n) => n.x)) + NODE_W + PAD_X;
    const svgHeight = PAD_Y * 2 + (maxLayer + 1) * (NODE_H + LAYER_GAP_Y) - LAYER_GAP_Y + NODE_H;
    const svgWidth = Math.max(maxLayerWidth, 200);

    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('width', String(svgWidth));
    svg.setAttribute('height', String(svgHeight));
    svg.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);
    svg.style.cssText = 'display: block; margin: 0 auto;';

    // Build step-to-node index for edge drawing
    const stepToNode: Map<number, DagNode> = new Map();
    for (const node of this.nodes) {
      stepToNode.set(node.step, node);
    }

    // Draw edges first (behind nodes)
    const edgeGroup = document.createElementNS(ns, 'g');
    edgeGroup.setAttribute('class', 'dag-edges');

    for (const node of this.nodes) {
      for (const prereq of node.prerequisites) {
        const parent = stepToNode.get(prereq);
        if (!parent) continue;

        const x1 = parent.x + NODE_W / 2;
        const y1 = parent.y + NODE_H;
        const x2 = node.x + NODE_W / 2;
        const y2 = node.y;

        // Curved path
        const midY = (y1 + y2) / 2;
        const path = document.createElementNS(ns, 'path');
        path.setAttribute('d', `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', '#8A9BA8');
        path.setAttribute('stroke-width', '1.5');
        path.setAttribute('opacity', '0.5');

        // Arrowhead
        const arrowSize = 5;
        const arrow = document.createElementNS(ns, 'polygon');
        arrow.setAttribute(
          'points',
          `${x2},${y2} ${x2 - arrowSize},${y2 - arrowSize * 1.5} ${x2 + arrowSize},${y2 - arrowSize * 1.5}`,
        );
        arrow.setAttribute('fill', '#8A9BA8');
        arrow.setAttribute('opacity', '0.5');

        edgeGroup.appendChild(path);
        edgeGroup.appendChild(arrow);
      }
    }
    svg.appendChild(edgeGroup);

    // Draw nodes
    const nodeGroup = document.createElementNS(ns, 'g');
    nodeGroup.setAttribute('class', 'dag-nodes');

    for (const node of this.nodes) {
      const g = document.createElementNS(ns, 'g');
      g.setAttribute('data-step', String(node.step));
      g.style.cursor = 'pointer';

      const color = ACTION_COLORS[node.action] ?? ACTION_COLOR_DEFAULT;

      // Node rectangle
      const rect = document.createElementNS(ns, 'rect');
      rect.setAttribute('x', String(node.x));
      rect.setAttribute('y', String(node.y));
      rect.setAttribute('width', String(NODE_W));
      rect.setAttribute('height', String(NODE_H));
      rect.setAttribute('rx', '6');
      rect.setAttribute('ry', '6');
      rect.setAttribute('fill', color);
      rect.setAttribute('opacity', '0.9');
      rect.setAttribute('stroke', 'transparent');
      rect.setAttribute('stroke-width', '2');

      // Step number (small, top-left)
      const stepLabel = document.createElementNS(ns, 'text');
      stepLabel.setAttribute('x', String(node.x + 8));
      stepLabel.setAttribute('y', String(node.y + 13));
      stepLabel.setAttribute('font-size', '9');
      stepLabel.setAttribute('font-family', 'var(--font-mono, monospace)');
      stepLabel.setAttribute('fill', 'rgba(255,255,255,0.7)');
      stepLabel.textContent = `#${node.step}`;

      // Description text (truncated)
      const desc = document.createElementNS(ns, 'text');
      desc.setAttribute('x', String(node.x + 8));
      desc.setAttribute('y', String(node.y + 27));
      desc.setAttribute('font-size', '11');
      desc.setAttribute('font-family', 'var(--font, system-ui)');
      desc.setAttribute('fill', '#fff');
      const maxChars = 20;
      const truncated =
        node.description.length > maxChars ? `${node.description.slice(0, maxChars)}...` : node.description;
      desc.textContent = truncated;

      g.appendChild(rect);
      g.appendChild(stepLabel);
      g.appendChild(desc);

      // Click handler
      g.addEventListener('click', () => {
        this.onStepClick(node.step);
      });

      // Hover effect
      g.addEventListener('mouseenter', () => {
        rect.setAttribute('opacity', '1');
      });
      g.addEventListener('mouseleave', () => {
        rect.setAttribute('opacity', node.step === this.highlightedStep ? '1' : '0.9');
      });

      nodeGroup.appendChild(g);
    }
    svg.appendChild(nodeGroup);

    this.svgEl = svg;
    this.container.appendChild(svg);

    // Apply initial highlight
    this.updateHighlight();
  }

  // ---------------------------------------------------------------------------
  // Highlight update
  // ---------------------------------------------------------------------------

  private updateHighlight(): void {
    if (!this.svgEl) return;

    const nodeGroups = this.svgEl.querySelectorAll('g[data-step]');
    for (const g of nodeGroups) {
      const step = Number.parseInt(g.getAttribute('data-step') ?? '-1', 10);
      const rect = g.querySelector('rect');
      if (!rect) continue;

      if (step === this.highlightedStep) {
        rect.setAttribute('stroke', '#fff');
        rect.setAttribute('stroke-width', '3');
        rect.setAttribute('opacity', '1');
      } else if (step <= this.highlightedStep) {
        rect.setAttribute('stroke', 'transparent');
        rect.setAttribute('stroke-width', '2');
        rect.setAttribute('opacity', '0.9');
      } else {
        // Future steps — dim
        rect.setAttribute('stroke', 'transparent');
        rect.setAttribute('stroke-width', '2');
        rect.setAttribute('opacity', '0.4');
      }
    }
  }
}
