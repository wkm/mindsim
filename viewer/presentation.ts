/**
 * Centralized 3D presentation layer for MindSim viewer.
 *
 * Single source of truth for how geometry is rendered: materials,
 * color tinting, and Blueprint palette constants.
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Blueprint.js palette (hex values matching CSS custom properties)
// ---------------------------------------------------------------------------
export const BP = {
  DARK_GRAY1: 0x182026,
  DARK_GRAY3: 0x293742,
  DARK_GRAY5: 0x394b59,
  GRAY1: 0x5c7080,
  GRAY3: 0x8a9ba8,
  GRAY4: 0xa7b6c2,
  GRAY5: 0xbfccd6,
  LIGHT_GRAY1: 0xced9e0,
  LIGHT_GRAY5: 0xf5f8fa,
  BLUE1: 0x0e5a8a,
  BLUE3: 0x137cbd,
  BLUE4: 0x2b95d6,
  BLUE5: 0x48aff0,
  GREEN3: 0x0f9960,
  GREEN4: 0x15b371,
  RED3: 0xdb3737,
  RED4: 0xf55656,
  GOLD3: 0xd99e0b,
};

// ---------------------------------------------------------------------------
// Render order — controls Three.js draw order for layered effects.
// ---------------------------------------------------------------------------
export const RENDER_ORDER = {
  SECTION_VIZ: -100,
  SECTION_CONTOUR: 9000,
  STENCIL_BACK: 0,
  STENCIL_FRONT: 1,
  STENCIL_CAP: 2,
};

// Stencil constants for per-body section caps.
// Each body gets a unique stencil ref (BASE + i) and render order band.
export const SECTION_STENCIL_BASE = 100;
export const SECTION_STENCIL_STRIDE = 10;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/** Convert a numeric hex color (e.g. 0xFF0000) to a CSS hex string ('#ff0000'). */
export function hexStr(n: number) {
  return `#${n.toString(16).padStart(6, '0')}`;
}

// ---------------------------------------------------------------------------
// Default material parameters
// ---------------------------------------------------------------------------
const DEFAULT_ROUGHNESS = 0.6;
const DEFAULT_METALNESS = 0.1;

// ---------------------------------------------------------------------------
// Color tinting
// ---------------------------------------------------------------------------

/**
 * Tint an entity color for the component browser.
 *
 * Blends `tintFraction` of the entity color onto a light base,
 * producing a pastel/technical-drawing look where edges remain visible.
 */
export function tintColor(entityColor: any, tintFraction = 0.5, baseHex = BP.LIGHT_GRAY5) {
  const base = new THREE.Color(baseHex);
  const entity = entityColor instanceof THREE.Color ? entityColor : new THREE.Color(entityColor);
  return base.lerp(entity, tintFraction);
}

// ---------------------------------------------------------------------------
// Material creation
// ---------------------------------------------------------------------------

export function createMaterial(color: any, opts: any = {}) {
  const params: any = {
    color,
    roughness: opts.roughness ?? DEFAULT_ROUGHNESS,
    metalness: opts.metalness ?? DEFAULT_METALNESS,
  };
  if (opts.transparent) {
    params.transparent = true;
    params.opacity = opts.opacity ?? 0.3;
    params.side = THREE.DoubleSide;
  }
  if (opts.wireframe) {
    params.wireframe = true;
  }
  return new THREE.MeshPhysicalMaterial(params);
}

export function createToolMaterial(color: any, opacity = 0.3) {
  return createMaterial(color, { transparent: true, opacity });
}

/** Create a mesh with material and add it to a group. */
export function addMeshWithEdges(geometry: THREE.BufferGeometry, color: any, group: THREE.Group, opts: any = {}) {
  const material = createMaterial(color, opts);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = !opts.wireframe;
  mesh.receiveShadow = !opts.wireframe;
  group.add(mesh);
  return mesh;
}
