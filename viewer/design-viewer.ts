/**
 * Design Viewer — thin wrapper that fetches a bot manifest and delegates
 * to ManifestViewer for the full viewing pipeline.
 *
 * Entry point: initDesignViewer(botName, viewport, treePanelEl)
 */

import type { ComponentTree } from './component-tree.ts';
import type { DesignScene } from './design-scene.ts';
import type { ViewerManifest } from './manifest-types.ts';
import { initManifestViewer } from './manifest-viewer.ts';
import type { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface DesignViewerContext {
  scene: DesignScene;
  tree: ComponentTree;
  viewport: Viewport3D;
  syncVisibility(): void;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function initDesignViewer(
  botName: string,
  viewport: Viewport3D,
  treePanelEl: HTMLElement,
): Promise<DesignViewerContext> {
  // Fetch manifest
  const resp = await fetch(`/api/bots/${botName}/viewer_manifest`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch viewer manifest: ${resp.status}`);
  }
  const manifest: ViewerManifest = await resp.json();

  // Delegate to ManifestViewer
  const ctx = await initManifestViewer({
    container: document.getElementById('canvas-container')!,
    treePanelEl,
    sidePanelEl: document.getElementById('side-panel')!,
    manifest,
    viewport,
    resolveStlUrl: (mesh) => `/api/bots/${botName}/meshes/${mesh}`,
    onNodeSelected: undefined,
  });

  return {
    scene: ctx.scene,
    tree: ctx.tree,
    viewport: ctx.viewport,
    syncVisibility: ctx.syncVisibility,
  };
}
