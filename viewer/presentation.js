/**
 * Centralized 3D presentation layer for MindSim viewer.
 *
 * Single source of truth for how geometry is rendered: materials,
 * edge outlines, color tinting, and Blueprint palette constants.
 * All viewer modes import from here instead of hardcoding material params.
 */

import * as THREE from 'three';

// ---------------------------------------------------------------------------
// Blueprint.js palette (hex values matching CSS custom properties)
// ---------------------------------------------------------------------------
export const BP = {
  DARK_GRAY1:  0x182026,
  DARK_GRAY3:  0x293742,
  DARK_GRAY5:  0x394B59,
  GRAY1:       0x5C7080,
  GRAY3:       0x8A9BA8,
  GRAY4:       0xA7B6C2,
  GRAY5:       0xBFCCD6,
  LIGHT_GRAY1: 0xCED9E0,
  LIGHT_GRAY5: 0xF5F8FA,
  BLUE1:       0x0E5A8A,
  BLUE3:       0x137CBD,
  BLUE4:       0x2B95D6,
  BLUE5:       0x48AFF0,
  GREEN3:      0x0F9960,
  GREEN4:      0x15B371,
  RED3:        0xDB3737,
  RED4:        0xF55656,
  GOLD3:       0xD99E0B,
};

// ---------------------------------------------------------------------------
// Render order — controls Three.js draw order for layered effects.
//
// The rendering pipeline draws in this order:
//   1. Section plane visualizer (translucent, behind everything)
//   2. Stencil back-face pass (invisible, writes stencil)
//   3. Stencil front-face pass (invisible, clears stencil)
//   4. Section cap fill (colored cross-section, reads stencil)
//   5. Normal geometry (default renderOrder 0)
//   6. Section contour lines (thick outlines on top)
//
// Per-layer stencil passes use SECTION_STENCIL_BASE + layerIndex * SECTION_STENCIL_STRIDE
// to keep each layer's passes grouped and ordered.
// ---------------------------------------------------------------------------
export const RENDER_ORDER = {
  SECTION_VIZ:      -100,   // translucent cut plane indicator
  STENCIL_BACK:        0,   // offset within a layer's stencil group
  STENCIL_FRONT:       1,
  STENCIL_CAP:         2,
  SECTION_CONTOUR:  9000,   // contour lines drawn last
};
export const SECTION_STENCIL_BASE = 100;    // first layer starts here
export const SECTION_STENCIL_STRIDE = 10;   // spacing between layers

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/** Convert a numeric hex color (e.g. 0xFF0000) to a CSS hex string ('#ff0000'). */
export function hexStr(n) { return '#' + n.toString(16).padStart(6, '0'); }

// ---------------------------------------------------------------------------
// Edge rendering constants
// ---------------------------------------------------------------------------
export const EDGE_COLOR = BP.DARK_GRAY1;
export const EDGE_OPACITY = 0.7;
export const EDGE_THRESHOLD_DEG = 28;

// Shared edge material — immutable, reused across all viewers
const _edgeMaterial = new THREE.LineBasicMaterial({
  color: EDGE_COLOR,
  transparent: true,
  opacity: EDGE_OPACITY,
});

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
 *
 * @param {THREE.Color|number} entityColor — the "true" color of the part
 * @param {number} tintFraction — how much entity color to mix in (0–1, default 0.2)
 * @param {number} baseHex — base body color to tint onto (default: LIGHT_GRAY5)
 * @returns {THREE.Color}
 */
export function tintColor(entityColor, tintFraction = 0.5, baseHex = BP.LIGHT_GRAY5) {
  const base = new THREE.Color(baseHex);
  const entity = entityColor instanceof THREE.Color
    ? entityColor
    : new THREE.Color(entityColor);
  return base.lerp(entity, tintFraction);
}

// ---------------------------------------------------------------------------
// Material + edge creation
// ---------------------------------------------------------------------------

/**
 * Create a MeshPhysicalMaterial with project-standard defaults.
 *
 * @param {THREE.Color|number} color
 * @param {Object} [opts]
 * @param {boolean} [opts.transparent]
 * @param {number}  [opts.opacity]
 * @param {boolean} [opts.wireframe]
 * @returns {THREE.MeshPhysicalMaterial}
 */
export function createMaterial(color, opts = {}) {
  const params = {
    color,
    roughness: DEFAULT_ROUGHNESS,
    metalness: DEFAULT_METALNESS,
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

/**
 * Create a transparent tool/overlay material (for section cuts, boolean tools).
 *
 * @param {THREE.Color|number} color
 * @param {number} [opacity=0.3]
 * @returns {THREE.MeshPhysicalMaterial}
 */
export function createToolMaterial(color, opacity = 0.3) {
  return createMaterial(color, { transparent: true, opacity });
}

/**
 * Add edge outlines to a mesh's parent group.
 *
 * Uses the shared edge material and threshold. Edges are non-interactive
 * (raycast disabled) so they don't interfere with picking.
 *
 * @param {THREE.BufferGeometry} geometry
 * @param {THREE.Group|THREE.Object3D} parent — group to add edges to
 * @param {THREE.Mesh} [sourceMesh] — if provided, copies position/quaternion/scale
 */
export function addEdges(geometry, parent, sourceMesh = null) {
  const edges = new THREE.EdgesGeometry(geometry, EDGE_THRESHOLD_DEG);
  const lines = new THREE.LineSegments(edges, _edgeMaterial);
  lines.raycast = () => {};

  if (sourceMesh) {
    lines.position.copy(sourceMesh.position);
    lines.quaternion.copy(sourceMesh.quaternion);
    lines.scale.copy(sourceMesh.scale);
  }

  parent.add(lines);
}

/**
 * Create a mesh with material and edge outlines in one call.
 *
 * @param {THREE.BufferGeometry} geometry
 * @param {THREE.Color|number} color
 * @param {THREE.Group} group — parent group to add mesh + edges to
 * @param {Object} [opts] — passed to createMaterial
 * @returns {THREE.Mesh}
 */
export function addMeshWithEdges(geometry, color, group, opts = {}) {
  const material = createMaterial(color, opts);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = !opts.wireframe;
  mesh.receiveShadow = !opts.wireframe;
  group.add(mesh);

  // Edge outlines — skip for wireframe/transparent
  if (!opts.wireframe && !opts.transparent) {
    addEdges(geometry, group);
  }

  return mesh;
}
