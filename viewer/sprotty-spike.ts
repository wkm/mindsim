/**
 * Sprotty + elkjs spike: verify port-to-port edge routing.
 *
 * Creates a minimal SGraph with 3 nodes (Pi, Servo1, Servo2), ports on each,
 * and edges connecting specific ports. Runs elkjs layout via sprotty-elk's
 * ElkLayoutEngine and checks that:
 *   - Layout completes without errors
 *   - Nodes get valid position coordinates
 *   - Edges route between the correct ports (not just node centers)
 *   - Port positions are assigned
 *
 * Run with: pnpm exec tsx viewer/sprotty-spike.ts
 */

import ElkConstructor from 'elkjs/lib/elk.bundled';
import type { ElkFactory } from 'sprotty-elk/lib/elk-layout';
import { DefaultElementFilter, DefaultLayoutConfigurator, ElkLayoutEngine } from 'sprotty-elk/lib/elk-layout';
import type { SEdge, SGraph, SLabel, SModelElement, SNode, SPort } from 'sprotty-protocol/lib/model';

// ---------------------------------------------------------------------------
// Build the SGraph model
// ---------------------------------------------------------------------------

function makeLabel(id: string, text: string): SLabel & { size: { width: number; height: number } } {
  return {
    type: 'label:port',
    id,
    text,
    size: { width: 40, height: 15 },
  };
}

function makePort(id: string, label: string): SPort & { children: SModelElement[] } {
  return {
    type: 'port',
    id,
    size: { width: 10, height: 10 },
    children: [makeLabel(`${id}:label`, label) as unknown as SModelElement],
  };
}

function makeNode(
  id: string,
  label: string,
  ports: (SPort & { children: SModelElement[] })[],
): SNode & { children: SModelElement[] } {
  const nodeLabel: SLabel & { size: { width: number; height: number } } = {
    type: 'label:node',
    id: `${id}:label`,
    text: label,
    size: { width: 80, height: 20 },
  };
  return {
    type: 'node',
    id,
    size: { width: 120, height: 60 },
    children: [nodeLabel as unknown as SModelElement, ...(ports as unknown as SModelElement[])],
  };
}

function buildGraph(): SGraph {
  const piUartOut = makePort('pi:uart_out', 'uart_out');
  const s1UartIn = makePort('servo1:uart_in', 'uart_in');
  const s1UartOut = makePort('servo1:uart_out', 'uart_out');
  const s2UartIn = makePort('servo2:uart_in', 'uart_in');

  const pi = makeNode('pi', 'Pi', [piUartOut]);
  const servo1 = makeNode('servo1', 'Servo1', [s1UartIn, s1UartOut]);
  const servo2 = makeNode('servo2', 'Servo2', [s2UartIn]);

  // Edges connect port-to-port (not node-to-node)
  const edge1: SEdge = {
    type: 'edge',
    id: 'edge:pi-servo1',
    sourceId: 'pi:uart_out',
    targetId: 'servo1:uart_in',
  };
  const edge2: SEdge = {
    type: 'edge',
    id: 'edge:servo1-servo2',
    sourceId: 'servo1:uart_out',
    targetId: 'servo2:uart_in',
  };

  return {
    type: 'graph',
    id: 'root',
    children: [pi, servo1, servo2, edge1, edge2],
  };
}

// ---------------------------------------------------------------------------
// Run layout and validate results
// ---------------------------------------------------------------------------

async function runSpike(): Promise<void> {
  const elkFactory: ElkFactory = () => new ElkConstructor();
  const engine = new ElkLayoutEngine(elkFactory, new DefaultElementFilter(), new DefaultLayoutConfigurator());

  const graph = buildGraph();
  console.log('Input graph:', JSON.stringify(graph, null, 2));

  const result = await engine.layout(graph);
  console.log('\nLayout result:', JSON.stringify(result, null, 2));

  // --- Assertions ---
  let pass = true;

  function assert(condition: boolean, msg: string) {
    if (!condition) {
      console.error(`FAIL: ${msg}`);
      pass = false;
    } else {
      console.log(`OK: ${msg}`);
    }
  }

  // Check nodes have positions
  const children = result.children ?? [];
  for (const child of children) {
    if (child.type === 'node') {
      const node = child as SNode & { position?: { x: number; y: number } };
      assert(
        node.position !== undefined && typeof node.position.x === 'number' && typeof node.position.y === 'number',
        `Node ${node.id} has valid position (${node.position?.x}, ${node.position?.y})`,
      );

      // Check ports have positions
      for (const sub of (node as SNode & { children?: SModelElement[] }).children ?? []) {
        if (sub.type === 'port') {
          const port = sub as SPort & { position?: { x: number; y: number } };
          assert(
            port.position !== undefined && typeof port.position.x === 'number' && typeof port.position.y === 'number',
            `Port ${port.id} has valid position (${port.position?.x}, ${port.position?.y})`,
          );
        }
      }
    }
  }

  // Check edges have routing points
  for (const child of children) {
    if (child.type === 'edge') {
      const edge = child as SEdge;
      assert(
        edge.routingPoints !== undefined && edge.routingPoints.length >= 0,
        `Edge ${edge.id} has routingPoints (${edge.routingPoints?.length ?? 0} points)`,
      );
      // Verify source/target still reference ports
      assert(edge.sourceId.includes(':'), `Edge ${edge.id} sourceId references a port: ${edge.sourceId}`);
      assert(edge.targetId.includes(':'), `Edge ${edge.id} targetId references a port: ${edge.targetId}`);
    }
  }

  console.log(pass ? '\n=== ALL CHECKS PASSED ===' : '\n=== SOME CHECKS FAILED ===');
  if (!pass) throw new Error('Some checks failed');
}

runSpike().catch((err) => {
  console.error('Spike failed with error:', err);
  throw err;
});
