/**
 * SceneSync — maps BotScene data model → Three.js scene graph.
 *
 * Single function that applies all visual state in one idempotent pass.
 * Only code that touches mesh visibility/opacity/emissive.
 *
 * Key rules:
 * - Body groups stay visible:true always (fixes body-0 parent cascading)
 * - Materials are cloned on first sync (no save/restore gymnastics)
 * - Call sync() after every BotScene mutation
 */

import type * as THREE from 'three';
import type { BotScene } from './bot-scene.ts';
import { GEOM_GROUP_STRUCTURAL } from './utils.ts';

// ---------------------------------------------------------------------------
// Material cloning — one-time per mesh
// ---------------------------------------------------------------------------

/** Tag to track whether we've cloned this mesh's material. */
const SYNCED = Symbol('botscene-synced');

/** Original opacity stored on first clone. */
const ORIG_OPACITY = Symbol('orig-opacity');

/** Original transparent flag stored on first clone. */
const ORIG_TRANSPARENT = Symbol('orig-transparent');

/** Original depthWrite flag stored on first clone. */
const ORIG_DEPTH_WRITE = Symbol('orig-depth-write');

/** Original alphaHash flag stored on first clone. */
const ORIG_ALPHA_HASH = Symbol('orig-alpha-hash');

interface SyncedMaterial extends THREE.Material {
  [SYNCED]?: boolean;
  [ORIG_OPACITY]?: number;
  [ORIG_TRANSPARENT]?: boolean;
  [ORIG_DEPTH_WRITE]?: boolean;
  [ORIG_ALPHA_HASH]?: boolean;
  opacity: number;
  transparent: boolean;
  depthWrite: boolean;
  alphaHash: boolean;
  emissive?: THREE.Color;
}

/**
 * Ensure this mesh has its own cloned material (first sync only).
 * Stores original opacity so we can reset without save/restore maps.
 */
function ensureCloned(mesh: THREE.Mesh): SyncedMaterial {
  let mat = mesh.material as SyncedMaterial;
  if (!mat[SYNCED]) {
    mat = (mesh.material as THREE.Material).clone() as SyncedMaterial;
    mat[SYNCED] = true;
    mat[ORIG_OPACITY] = mat.opacity;
    mat[ORIG_TRANSPARENT] = mat.transparent;
    mat[ORIG_DEPTH_WRITE] = mat.depthWrite;
    mat[ORIG_ALPHA_HASH] = mat.alphaHash;
    mesh.material = mat;
  }
  return mat;
}

// ---------------------------------------------------------------------------
// sync()
// ---------------------------------------------------------------------------

/**
 * Apply BotScene visual state to the Three.js scene graph.
 *
 * @param botScene - The data model
 * @param bodies - Map of body ID → THREE.Group (from bot-viewer)
 */
export function sync(botScene: BotScene, bodies: Record<number, THREE.Group>): void {
  for (const body of botScene.bodies) {
    const group = bodies[body.id];
    if (!group) continue;

    // Body groups always stay visible (fixes body-0 cascading bug)
    group.visible = true;

    const emissiveHex = botScene.bodyEmissive(body.id);

    group.traverse((child: any) => {
      if (!child.isMesh) return;

      // Original meshes replaced by multi-material sub-meshes stay hidden,
      // but sub-meshes themselves ARE synced (they don't have this flag).
      if (child._multiMaterialReplaced) {
        child.visible = false;
        return;
      }

      // Per-mesh opacity: uses geomGroup for transparent mode split
      const targetOpacity = botScene.bodyOpacity(body.id, child.geomGroup);
      const mat = ensureCloned(child);
      const origOpacity = mat[ORIG_OPACITY] ?? 1.0;

      if (targetOpacity <= 0) {
        // Hidden — make mesh invisible
        child.visible = false;
      } else {
        child.visible = true;

        if (targetOpacity < 1.0) {
          mat.opacity = targetOpacity;
          mat.transparent = true;
          mat.depthWrite = false;
        } else {
          // Normal — restore originals
          mat.opacity = origOpacity;
          mat.transparent = mat[ORIG_TRANSPARENT] ?? origOpacity < 1.0;
          mat.depthWrite = mat[ORIG_DEPTH_WRITE] ?? true;
        }

        // Emissive highlight (structural meshes only)
        if (mat.emissive && child.geomGroup === GEOM_GROUP_STRUCTURAL) {
          mat.emissive.setHex(emissiveHex);
        }
      }
    });
  }
}
