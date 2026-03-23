/**
 * Shared type definitions for the MindSim viewer.
 *
 * ViewerContext is the bag of references passed to every mode constructor.
 * MuJoCo types are opaque (`any`) because the WASM module has no type defs.
 */

import type * as THREE from 'three';
import type { BotScene } from './bot-scene.ts';
import type { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// MuJoCo — opaque WASM types (no upstream .d.ts)
// ---------------------------------------------------------------------------

/** MuJoCo WASM module (load_mujoco() return value). */
export type MujocoModule = any;

/** MuJoCo MjModel instance. */
export type MjModel = any;

/** MuJoCo MjData instance. */
export type MjData = any;

// ---------------------------------------------------------------------------
// Viewer context — passed to all mode constructors
// ---------------------------------------------------------------------------

/**
 * The shared context object built in `initBotViewer` and passed to every mode.
 *
 * MuJoCo fields are typed as opaque aliases because the WASM module doesn't
 * ship TypeScript declarations. Everything else is concrete.
 */
export interface ViewerContext {
  // ── MuJoCo (opaque WASM types) ──
  mujoco: MujocoModule;
  model: MjModel;
  data: MjData;

  // ── Three.js scene graph ──
  bodies: Record<number, THREE.Group>;
  mujocoRoot: THREE.Group;
  scene: THREE.Scene;
  camera: THREE.Camera;
  renderer: THREE.WebGLRenderer;
  controls: THREE.EventDispatcher & { target: THREE.Vector3; enabled: boolean; update(): void };
  viewport: Viewport3D;

  // ── Coordinate helpers ──
  syncTransforms: () => void;
  getPosition: (buffer: any, index: number, target: THREE.Vector3) => THREE.Vector3;
  getQuaternion: (buffer: any, index: number, target: THREE.Quaternion) => THREE.Quaternion;
  toMujocoPos: (v: THREE.Vector3) => THREE.Vector3;

  // ── Model metadata ──
  getMujocoName: (adrArray: any, index: number) => string;
  botName: string;

  // ── Data model + sync ──
  botScene: BotScene;
  syncScene: () => void;
}

// ---------------------------------------------------------------------------
// Mode interface — common shape for all viewer modes
// ---------------------------------------------------------------------------

/** Minimal interface shared by all viewer modes. */
export interface ViewerMode {
  activate(): void;
  deactivate(): void;
  update?(): void;
}

// ---------------------------------------------------------------------------
// Joint slider info — shared between JointMode and IK readout
// ---------------------------------------------------------------------------

export interface SliderInfo {
  jointIdx: number;
  qposAddr: number;
  rangeMin: number;
  rangeMax: number;
  sliderId: string;
  color: string;
  name: string;
}

// ---------------------------------------------------------------------------
// Angle arc overlay state — used by JointMode
// ---------------------------------------------------------------------------

export interface AngleArc {
  group: THREE.Group;
  curArcGroup: THREE.Group;
  axisDir: THREE.Vector3;
  rangeMin: number;
  rangeMax: number;
  arcRadius: number;
}

// ---------------------------------------------------------------------------
// Joint readout element — used by IKMode
// ---------------------------------------------------------------------------

export interface JointReadoutEl {
  jointIdx: number;
  addr: number;
}
