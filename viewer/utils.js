/**
 * Shared utilities for the MindSim viewer.
 */

import * as THREE from 'three';

// MuJoCo geom group constants
export const GEOM_GROUP_STRUCTURAL = 0;
export const GEOM_GROUP_DETAIL = 1;     // servos, screws, components
export const GEOM_GROUP_WIRE = 2;

/**
 * Dispose all children of a Three.js group, cleaning up geometry and materials.
 * @param {THREE.Group} group
 */
export function clearGroup(group) {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    child.traverse(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
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
export function radToDegStr(rad) {
  return (rad * 180 / Math.PI).toFixed(1);
}

/**
 * Create a small sphere marker mesh.
 * @param {number} color - hex color
 * @returns {THREE.Mesh}
 */
export function createMarker(color) {
  const geo = new THREE.SphereGeometry(0.008, 12, 12);
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.visible = false;
  return mesh;
}
