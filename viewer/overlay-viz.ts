/**
 * Overlay Viz -- wire stubs and fastener instance rendering.
 *
 * Creates Three.js meshes for:
 *   - Wire stubs: small cylinders at connector sockets, colored by bus type
 *   - Fastener instances: instanced STL meshes at mount points
 *
 * All meshes are parented to their body's Three.js group so they
 * automatically inherit body transforms, visibility, and isolation.
 */

import * as THREE from 'three';
import type { DesignScene } from './design-scene.ts';
import type { ManifestPart } from './manifest-types.ts';
import { fetchSTL } from './utils.ts';

// ---------------------------------------------------------------------------
// Wire stub bus type colors
// ---------------------------------------------------------------------------

const BUS_TYPE_COLORS: Record<string, number> = {
  uart_half_duplex: 0x4444ff, // blue
  csi: 0x44ff44, // green
  power: 0xff4444, // red
  usb: 0xffffff, // white
  gpio: 0xaa44ff, // purple
  pwm: 0xffaa44, // orange
  balance: 0xffff44, // yellow
};
const BUS_TYPE_FALLBACK = 0x888888; // gray

// ---------------------------------------------------------------------------
// Fastener material (metallic gray)
// ---------------------------------------------------------------------------

function createFastenerMaterial(): THREE.MeshPhysicalMaterial {
  return new THREE.MeshPhysicalMaterial({
    color: 0x888888,
    metalness: 0.8,
    roughness: 0.3,
  });
}

// ---------------------------------------------------------------------------
// OverlayViz
// ---------------------------------------------------------------------------

export class OverlayViz {
  private _wireStubMeshes: THREE.Mesh[] = [];
  private _fastenerMeshes: THREE.Mesh[] = [];
  private _bodyGroups: Record<string, THREE.Group>;
  private _botName: string;
  showWires = true;
  showFasteners = true;

  constructor(bodyGroups: Record<string, THREE.Group>, botName: string) {
    this._bodyGroups = bodyGroups;
    this._botName = botName;
  }

  // -- Wire stubs --

  buildWireStubs(parts: ManifestPart[], designScene: DesignScene): void {
    const stubs = parts.filter((p) => p.category === 'wire' && p.wire_kind === 'stub' && p.position && p.direction);

    for (const stub of stubs) {
      const group = this._bodyGroups[stub.parent_body];
      if (!group) continue;

      const color = BUS_TYPE_COLORS[stub.bus_type ?? ''] ?? BUS_TYPE_FALLBACK;
      const radius = 0.0015; // 1.5mm
      const height = stub.length || 0.025;

      // Cylinder defaults to Y-axis; we orient it after creation
      const geo = new THREE.CylinderGeometry(radius, radius, height, 8);
      const mat = new THREE.MeshPhysicalMaterial({
        color,
        roughness: 0.5,
        metalness: 0.0,
        emissive: color,
        emissiveIntensity: 0.15,
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = false;
      mesh.receiveShadow = false;

      // Tag for raycasting / tooltip
      (mesh as any).overlayType = 'wire_stub';
      (mesh as any).overlayData = stub;

      // Orient: cylinder default axis is Y, we need direction vector
      const dir = new THREE.Vector3(stub.direction![0], stub.direction![1], stub.direction![2]).normalize();

      // Position at start + half-length along direction (cylinder centered at origin)
      const pos = new THREE.Vector3(stub.position![0], stub.position![1], stub.position![2]);
      pos.addScaledVector(dir, height / 2);
      mesh.position.copy(pos);

      // Rotate from Y-axis to direction
      const yAxis = new THREE.Vector3(0, 1, 0);
      const quat = new THREE.Quaternion().setFromUnitVectors(yAxis, dir);
      mesh.quaternion.copy(quat);

      group.add(mesh);
      this._wireStubMeshes.push(mesh);

      // Register with design scene for visibility sync
      const nodeId = `wire-group:${stub.parent_body}`;
      designScene.registerMesh(nodeId, mesh);
    }
  }

  // -- Fastener instances --

  async buildFasteners(parts: ManifestPart[], designScene: DesignScene): Promise<void> {
    const fasteners = parts.filter((p) => p.category === 'fastener' && p.axis);

    if (fasteners.length === 0) return;

    // Deduplicate mesh files
    const meshFiles = new Set(fasteners.map((f) => f.mesh));
    const geometryCache: Map<string, THREE.BufferGeometry> = new Map();

    // Load all unique STL geometries in parallel
    const loadPromises = [...meshFiles].map(async (meshFile) => {
      const geometry = await fetchSTL(this._botName, meshFile);
      if (geometry) {
        geometryCache.set(meshFile, geometry);
      }
    });
    await Promise.all(loadPromises);

    // Create mesh instances
    for (const fastener of fasteners) {
      const group = this._bodyGroups[fastener.parent_body];
      if (!group) continue;

      const geo = geometryCache.get(fastener.mesh);
      if (!geo) continue;

      const mat = createFastenerMaterial();
      const mesh = new THREE.Mesh(geo.clone(), mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      // Tag for raycasting / tooltip
      (mesh as any).overlayType = 'fastener';
      (mesh as any).overlayData = fastener;

      // Position -- use axis-based positioning when available (body-local),
      // otherwise fall back to pos/quat (world-frame)
      if (fastener.pos) {
        mesh.position.set(fastener.pos[0], fastener.pos[1], fastener.pos[2]);
      }

      // Orient: fastener STL has head at +Z, shank extends in -Z.
      // Manifest axis = insertion direction (into the hole).
      // Align +Z (head) opposite to insertion → head faces outward.
      const axis = new THREE.Vector3(fastener.axis![0], fastener.axis![1], fastener.axis![2]).normalize();
      const posZ = new THREE.Vector3(0, 0, 1);
      const quatFromAxis = new THREE.Quaternion().setFromUnitVectors(posZ, axis);
      mesh.quaternion.copy(quatFromAxis);

      group.add(mesh);
      this._fastenerMeshes.push(mesh);

      // Register with design scene -- use the same node ID pattern as the tree
      const nodeId = fastener.joint
        ? `fastener-group:${fastener.joint}:${fastener.name}`
        : `fastener-group:${fastener.id}:${fastener.name}`;
      designScene.registerMesh(nodeId, mesh);
    }
  }

  // -- Visibility toggles --

  setWiresVisible(visible: boolean): void {
    this.showWires = visible;
    for (const mesh of this._wireStubMeshes) {
      mesh.visible = visible;
    }
  }

  setFastenersVisible(visible: boolean): void {
    this.showFasteners = visible;
    for (const mesh of this._fastenerMeshes) {
      mesh.visible = visible;
    }
  }

  // -- Tooltip support --

  /** Get all overlay meshes for raycasting. */
  getRaycastTargets(): THREE.Mesh[] {
    const targets: THREE.Mesh[] = [];
    if (this.showWires) targets.push(...this._wireStubMeshes);
    if (this.showFasteners) targets.push(...this._fastenerMeshes);
    return targets;
  }

  /** Format a tooltip string for an overlay mesh hit. */
  static formatTooltip(mesh: THREE.Mesh): string | null {
    const data = (mesh as any).overlayData;
    const type = (mesh as any).overlayType;
    if (!data) return null;

    if (type === 'wire_stub') {
      return `${data.name} \u2014 ${data.bus_type}`;
    }
    if (type === 'fastener') {
      return `${data.name} \u2014 ${data.context || 'fastener'}`;
    }
    return null;
  }

  // -- Cleanup --

  dispose(): void {
    for (const mesh of [...this._wireStubMeshes, ...this._fastenerMeshes]) {
      mesh.parent?.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) mesh.material.dispose();
    }
    this._wireStubMeshes = [];
    this._fastenerMeshes = [];
  }
}
