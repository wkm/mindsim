/**
 * Shared utilities for the MindSim viewer.
 */

import * as THREE from 'three';

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
