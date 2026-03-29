/**
 * BotScene — pure data model for the bot viewer's visual state.
 *
 * No Three.js imports. All visual state lives here as plain data.
 * Modes mutate BotScene, then call sync() to apply changes to Three.js.
 *
 * Key invariant: body groups stay visible:true always in Three.js.
 * Sync sets mesh.visible per-mesh, never group.visible.
 * This fixes the body-0-parent-cascading bug.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Opacity for ghosted (dimmed) bodies. */
export const GHOST_OPACITY = 0.06;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ViewerMode = 'explore' | 'joint' | 'ik';

export interface BodyState {
  id: number;
  name: string;
  visible: boolean;
  ghosted: boolean;
  hovered: boolean;
  selected: boolean;
}

// ---------------------------------------------------------------------------
// BotScene
// ---------------------------------------------------------------------------

export class BotScene {
  bodies: BodyState[];
  activeMode: ViewerMode;

  /** Which body IDs are isolated (empty = no isolation). */
  private _isolatedIds: Set<number>;

  constructor(bodyCount: number, bodyNames: string[]) {
    this.bodies = [];
    for (let i = 0; i < bodyCount; i++) {
      this.bodies.push({
        id: i,
        name: bodyNames[i] ?? `body_${i}`,
        visible: true,
        ghosted: false,
        hovered: false,
        selected: false,
      });
    }

    this.activeMode = 'explore';
    this._isolatedIds = new Set();
  }

  // ── Mutations ──

  setBodyVisible(bodyId: number, visible: boolean): void {
    const body = this.bodies[bodyId];
    if (body) body.visible = visible;
  }

  /**
   * Ghost all bodies except those in keepBodyIds.
   * Body 0 (world) is always ghosted unless explicitly kept.
   */
  ghostAllExcept(keepBodyIds: number[]): void {
    const keepSet = new Set(keepBodyIds);
    for (const body of this.bodies) {
      body.ghosted = !keepSet.has(body.id);
    }
  }

  /** Remove all ghosting. */
  unghost(): void {
    for (const body of this.bodies) {
      body.ghosted = false;
    }
  }

  /**
   * Isolate a body — only this body (and body 0) are visible.
   * Body 0 stays visible because it's the scene graph parent,
   * but its meshes should be hidden (handled by sync).
   */
  isolate(bodyId: number): void {
    this._isolatedIds.clear();
    this._isolatedIds.add(bodyId);
    for (const body of this.bodies) {
      body.visible = body.id === bodyId;
    }
    // Body 0 group must stay visible for children to render
    if (this.bodies[0]) this.bodies[0].visible = true;
  }

  /** Isolate multiple bodies (e.g., joint parent + child). */
  isolateMultiple(bodyIds: number[]): void {
    this._isolatedIds = new Set(bodyIds);
    const keepSet = new Set(bodyIds);
    for (const body of this.bodies) {
      body.visible = keepSet.has(body.id);
    }
    // Body 0 group must stay visible for children to render
    if (this.bodies[0]) this.bodies[0].visible = true;
  }

  /** Reset all visibility — show everything. */
  showAll(): void {
    this._isolatedIds.clear();
    for (const body of this.bodies) {
      body.visible = true;
      body.ghosted = false;
    }
  }

  /** Set which body is hovered (null to clear). */
  setHovered(bodyId: number | null): void {
    for (const body of this.bodies) {
      body.hovered = body.id === bodyId;
    }
  }

  /** Set which body is selected (null to clear). */
  setSelected(bodyId: number | null): void {
    for (const body of this.bodies) {
      body.selected = body.id === bodyId;
    }
  }

  // ── Queries ──

  /** IDs of all currently visible bodies. */
  visibleBodyIds(): number[] {
    return this.bodies.filter((b) => b.visible).map((b) => b.id);
  }

  /** Whether any body is currently isolated. */
  isIsolated(): boolean {
    return this._isolatedIds.size > 0;
  }

  /** The set of isolated body IDs (empty if not isolating). */
  isolatedIds(): ReadonlySet<number> {
    return this._isolatedIds;
  }

  /**
   * Compute the target opacity for a body's meshes.
   *
   * - Invisible → 0
   * - Ghosted → GHOST_OPACITY
   * - Normal → 1.0
   *
   * Body 0 during isolation: visible=true but not in isolatedIds
   * → its meshes should be hidden (opacity 0).
   */
  bodyOpacity(bodyId: number): number {
    const body = this.bodies[bodyId];
    if (!body) return 0;

    // Body 0 special case: during isolation, it's visible (for children)
    // but its own meshes should be hidden unless it's explicitly isolated
    if (body.id === 0 && this._isolatedIds.size > 0 && !this._isolatedIds.has(0)) {
      return 0;
    }

    if (!body.visible) return 0;
    if (body.ghosted) return GHOST_OPACITY;
    return 1.0;
  }

  /**
   * Compute the emissive intensity for hover/selection highlight.
   * Returns a hex color value (0 = no highlight).
   */
  bodyEmissive(bodyId: number): number {
    const body = this.bodies[bodyId];
    if (!body) return 0;
    if (body.hovered) return 0x666666;
    if (body.selected) return 0x333333;
    return 0;
  }
}
