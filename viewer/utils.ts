/**
 * Shared utilities for the MindSim viewer.
 */

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { errorMessage, timedFetch, warn } from './log.ts';
import type { ManifestMaterial, ManifestPart } from './manifest-types.ts';

// MuJoCo geom group constants
export const GEOM_GROUP_STRUCTURAL = 0;
export const GEOM_GROUP_DETAIL = 1; // servos, screws, components
export const GEOM_GROUP_WIRE = 2;

/**
 * Dispose all children of a Three.js group, cleaning up geometry and materials.
 * @param {THREE.Group} group
 */
export function clearGroup(group: THREE.Group): void {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    child.traverse((obj: any) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material))
          obj.material.forEach((m: THREE.Material) => {
            m.dispose();
          });
        else obj.material.dispose();
      }
    });
  }
}

/**
 * Convert radians to degrees, formatted to one decimal.
 * @param {number} rad
 * @returns {string}
 */
export function radToDegStr(rad: number): string {
  return ((rad * 180) / Math.PI).toFixed(1);
}

/**
 * Create a small sphere marker mesh.
 * @param {number} color - hex color
 * @returns {THREE.Mesh}
 */
export function createMarker(color: number): THREE.Mesh {
  const geo = new THREE.SphereGeometry(0.008, 12, 12);
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.visible = false;
  return mesh;
}

// ---------------------------------------------------------------------------
// Arc geometry helpers (extracted from joint-mode for reuse)
// ---------------------------------------------------------------------------
const _arcUp = new THREE.Vector3(0, 0, 1);
const _arcQuat = new THREE.Quaternion();

/**
 * Create a BufferGeometry arc in the XY plane.
 * @param {number} radius
 * @param {number} startAngle - radians
 * @param {number} endAngle - radians
 * @param {number} segments
 * @returns {THREE.BufferGeometry}
 */
export function createArcGeometry(
  radius: number,
  startAngle: number,
  endAngle: number,
  segments: number,
): THREE.BufferGeometry {
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const t = startAngle + (endAngle - startAngle) * (i / segments);
    points.push(new THREE.Vector3(Math.cos(t) * radius, Math.sin(t) * radius, 0));
  }
  return new THREE.BufferGeometry().setFromPoints(points);
}

/**
 * Orient a mesh/line from the Z-axis to a target axis direction.
 * @param {THREE.Object3D} obj
 * @param {THREE.Vector3} axisDir - normalized target direction
 */
export function orientToAxis(obj: THREE.Object3D, axisDir: THREE.Vector3): void {
  _arcQuat.setFromUnitVectors(_arcUp, axisDir);
  obj.quaternion.copy(_arcQuat);
}

// ---------------------------------------------------------------------------
// Text sprite for dimension / spec labels
// ---------------------------------------------------------------------------

/**
 * Create a billboard text sprite.
 * @param {string} text
 * @param {Object} opts
 * @param {number} [opts.fontSize=12]
 * @param {string} [opts.color='#ffffff']
 * @param {string} [opts.bgColor='rgba(0,0,0,0.5)']
 * @returns {THREE.Sprite}
 */
export function createTextSprite(
  text: string,
  {
    fontSize = 12,
    color = '#ffffff',
    bgColor = 'rgba(0,0,0,0.5)',
  }: { fontSize?: number; color?: string; bgColor?: string } = {},
): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const font = `${fontSize * 4}px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
  ctx.font = font;
  const metrics = ctx.measureText(text);
  const padding = fontSize * 2;
  canvas.width = Math.ceil(metrics.width + padding * 2);
  canvas.height = Math.ceil(fontSize * 6);

  // Background
  ctx.fillStyle = bgColor;
  ctx.roundRect(0, 0, canvas.width, canvas.height, fontSize);
  ctx.fill();

  // Text
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  return new THREE.Sprite(mat);
}

// ---------------------------------------------------------------------------
// Fastener grouping
// ---------------------------------------------------------------------------

export interface FastenerGroup {
  key: string;
  name: string;
  label: string;
  count: number;
  items: ManifestPart[];
}

/** Group identical fasteners: "4x M2 SHC" instead of 4 separate nodes. */
export function groupFasteners(fasteners: ManifestPart[]): FastenerGroup[] {
  const groups: Record<string, FastenerGroup> = {};
  for (const f of fasteners) {
    const key = f.name;
    if (!groups[key]) {
      groups[key] = { key, name: f.name, label: f.name, count: 0, items: [] };
    }
    groups[key].count++;
    groups[key].items.push(f);
  }
  return Object.values(groups).map((g) => ({
    ...g,
    label: g.count > 1 ? `${g.count}\u00d7 ${g.name}` : g.name,
  }));
}

// ---------------------------------------------------------------------------
// Joint name humanization
// ---------------------------------------------------------------------------

/** Humanize a joint name: "left_wheel" -> "Left Wheel" */
export function humanizeJointName(name: string): string {
  return name
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

// ---------------------------------------------------------------------------
// STL loading
// ---------------------------------------------------------------------------

const stlLoader = new STLLoader();

/** Fetch an STL and parse it into a BufferGeometry with vertex normals. */
export async function fetchSTL(botName: string, meshFile: string): Promise<THREE.BufferGeometry | null> {
  try {
    const resp = await timedFetch(`/api/bots/${botName}/meshes/${meshFile}`);
    if (!resp.ok) {
      warn('viewer', `failed to fetch ${meshFile}: ${resp.status}`);
      return null;
    }
    const buf = await resp.arrayBuffer();
    const geometry = stlLoader.parse(buf);
    geometry.computeVertexNormals();
    return geometry;
  } catch (err) {
    warn('viewer', `error loading ${meshFile}`, { error: errorMessage(err) });
    return null;
  }
}

/** Fetch an STL from a full URL and parse it into a BufferGeometry with vertex normals. */
export async function fetchSTLFromUrl(url: string): Promise<THREE.BufferGeometry | null> {
  try {
    const resp = await timedFetch(url);
    if (!resp.ok) {
      warn('viewer', `failed to fetch STL: ${url} (${resp.status})`);
      return null;
    }
    const buf = await resp.arrayBuffer();
    const geometry = stlLoader.parse(buf);
    geometry.computeVertexNormals();
    return geometry;
  } catch (err) {
    warn('viewer', `STL fetch error: ${url}`, { error: errorMessage(err) });
    return null;
  }
}

// ---------------------------------------------------------------------------
// Manifest quaternion conversion
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Positioned mesh with edge lines
// ---------------------------------------------------------------------------

/** Shared edge line material — consistent wireframe look across viewers. */
export const EDGE_LINE_MAT = new THREE.LineBasicMaterial({
  color: 0x000000,
  transparent: true,
  opacity: 0.35,
  linewidth: 2, // only effective on some platforms; WebGL often caps at 1px
});

/** Create a positioned mesh with edge lines, matching component browser style. */
export function createPositionedMesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  pos: number[],
  quat: number[],
): THREE.Mesh {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.position.set(pos[0], pos[1], pos[2]);
  mesh.quaternion.copy(manifestQuatToThree(quat));

  // Add wireframe edge lines for surface definition (same as component browser)
  const edges = new THREE.EdgesGeometry(geometry, 28);
  const lines = new THREE.LineSegments(edges, EDGE_LINE_MAT);
  lines.raycast = () => {}; // edges don't participate in picking
  mesh.add(lines);

  return mesh;
}

// ---------------------------------------------------------------------------
// Manifest quaternion conversion
// ---------------------------------------------------------------------------

/** Convert manifest wxyz quaternion to Three.js xyzw quaternion. */
export function manifestQuatToThree(q: number[]): THREE.Quaternion {
  // Manifest: [w, x, y, z] -> Three.js: new Quaternion(x, y, z, w)
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]);
}

// ---------------------------------------------------------------------------
// Material creation
// ---------------------------------------------------------------------------

/** Create a MeshPhysicalMaterial from a manifest material definition. */
export function makeMaterial(matDef: ManifestMaterial | undefined): THREE.MeshPhysicalMaterial {
  if (!matDef) {
    return new THREE.MeshPhysicalMaterial({ color: 0xb0b0b0, roughness: 0.7 });
  }
  const color = new THREE.Color(matDef.color[0], matDef.color[1], matDef.color[2]);
  return new THREE.MeshPhysicalMaterial({
    color,
    metalness: matDef.metallic,
    roughness: matDef.roughness,
    transparent: matDef.opacity < 1.0,
    opacity: matDef.opacity,
  });
}

// ---------------------------------------------------------------------------
// Part node ID resolution
// ---------------------------------------------------------------------------

/**
 * Resolve the SceneTree node ID for a manifest part.
 *
 * With the new model, parts[] only contains fasteners and wires.
 * Servos, horns, and wheels are in bodies[]; mounted components are in mounts[].
 */
export function resolvePartNodeId(part: ManifestPart): string {
  if (part.category === 'fastener') {
    // Fasteners are grouped — use the joint-scoped or mount-scoped group key
    return part.joint ? `fastener-group:${part.joint}:${part.name}` : `fastener-group:${part.id}:${part.name}`;
  }
  if (part.category === 'wire') {
    // Wire stubs/segments register under the wire group node for their body
    return `wire-group:${part.parent_body}`;
  }
  return `part:${part.id}`;
}
