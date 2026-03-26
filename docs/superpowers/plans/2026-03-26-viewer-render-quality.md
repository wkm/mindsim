# Viewer Render Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add environment-based lighting, tone mapping, and soft shadows to the Three.js bot viewer for more realistic material rendering.

**Architecture:** New `environment.ts` module handles HDRI loading. Renderer and lighting changes go in `viewport3d.ts`. Edge detection composer in `presentation.ts` gets a tone mapping fix. Each layer is a separate commit.

**Tech Stack:** Three.js 0.183, `RGBELoader`, `PMREMGenerator`, Poly Haven CDN for HDRIs.

**Linear:** PER-63

**Worktree:** `.claude/worktrees/render-quality` (branch `exp/260326-render-quality`)

**Spec:** `docs/superpowers/specs/2026-03-26-viewer-render-quality-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `viewer/environment.ts` | Create | HDRI loading, PMREMGenerator, apply to scene |
| `viewer/viewport3d.ts` | Modify | Tone mapping, soft shadows, lighting rebalance, call environment loader |
| `viewer/presentation.ts` | Modify | Fix tone mapping double-application in edge composer |

---

### Task 1: Create environment.ts — HDRI loader

**Files:**
- Create: `viewer/environment.ts`

- [ ] **Step 1: Create environment.ts with HDRI loading function**

```typescript
import * as THREE from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

const HDRI_BASE = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k';

export const HDRI_PRESETS: Record<string, { file: string; exposure: number }> = {
  artist_workshop: { file: 'artist_workshop_1k.hdr', exposure: 1.0 },
  studio_small_09: { file: 'studio_small_09_1k.hdr', exposure: 0.8 },
  old_bus_depot: { file: 'old_bus_depot_1k.hdr', exposure: 1.0 },
};

/**
 * Load an HDRI environment map and apply it to the scene for IBL.
 * Background stays unchanged — the HDRI is only used for lighting/reflections.
 * Degrades gracefully if load fails (logs warning, scene uses existing lights).
 */
export async function loadEnvironment(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  preset: string = 'artist_workshop',
): Promise<void> {
  const entry = HDRI_PRESETS[preset];
  if (!entry) {
    console.warn(`[environment] Unknown HDRI preset "${preset}", skipping`);
    return;
  }

  const url = `${HDRI_BASE}/${entry.file}`;
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();

  try {
    const loader = new RGBELoader();
    const hdrTexture = await loader.loadAsync(url);
    const envMap = pmrem.fromEquirectangular(hdrTexture).texture;
    hdrTexture.dispose();
    scene.environment = envMap;
    scene.environmentIntensity = 1.0;
    renderer.toneMappingExposure = entry.exposure;
  } catch (err) {
    console.warn('[environment] Failed to load HDRI, falling back to default lighting:', err);
  } finally {
    pmrem.dispose();
  }
}
```

- [ ] **Step 2: Verify lint passes**

Run: `pnpm exec biome check viewer/environment.ts`
Expected: No errors

- [ ] **Step 3: Verify typecheck passes**

Run: `pnpm exec tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add viewer/environment.ts
git commit -m "feat(viewer): add environment.ts — HDRI loader for IBL"
```

---

### Task 2: Wire environment into Viewport3D + add tone mapping

**Files:**
- Modify: `viewer/viewport3d.ts:12-19` (imports)
- Modify: `viewer/viewport3d.ts:642-665` (renderer config + lighting)

- [ ] **Step 1: Add import to viewport3d.ts**

At line 19, after the existing imports, add:

```typescript
import { loadEnvironment } from './environment.ts';
```

- [ ] **Step 2: Add tone mapping after renderer creation**

In the constructor, after line 649 (`this._ren.localClippingEnabled = true;`), add:

```typescript
this._ren.toneMapping = THREE.ACESFilmicToneMapping;
this._ren.toneMappingExposure = 1.0;
```

- [ ] **Step 3: Rebalance lighting for IBL**

Replace the lighting block (lines 659-665):

```typescript
// Before:
this._scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const dir = new THREE.DirectionalLight(0xffffff, 1.6);
dir.position.set(0.3, 0.5, 0.4);
this._scene.add(dir);
const fill = new THREE.DirectionalLight(0xffffff, 0.5);
fill.position.set(-0.3, -0.2, -0.4);
this._scene.add(fill);
```

```typescript
// After:
this._scene.add(new THREE.AmbientLight(0xffffff, 0.3));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(0.3, 0.5, 0.4);
this._scene.add(dir);
this._dirLight = dir;
const fill = new THREE.DirectionalLight(0xffffff, 0.3);
fill.position.set(-0.3, -0.2, -0.4);
this._scene.add(fill);
```

Note: Ambient reduced from 1.0→0.3 (IBL provides ambient), key light from 1.6→1.2, fill from 0.5→0.3. Store `dir` as `this._dirLight` for shadow camera fitting in Task 4.

Add the field declaration near the other `_` fields (around line 134-140):

```typescript
_dirLight: THREE.DirectionalLight | null;
```

- [ ] **Step 4: Call environment loader after construction**

At the end of the constructor (after the grid helper setup, around line 673), add:

```typescript
loadEnvironment(this._ren, this._scene);
```

Note: This is fire-and-forget async — the constructor is synchronous. The scene renders immediately with rebalanced direct lights; the HDRI pops in when loaded (~1-2s). If CDN is unreachable, the fallback lighting still works.

- [ ] **Step 5: Verify lint + typecheck**

Run: `make lint`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add viewer/viewport3d.ts
git commit -m "feat(viewer): wire IBL environment + tone mapping into Viewport3D

Adds ACES Filmic tone mapping, loads workshop HDRI for IBL,
rebalances ambient/directional lights to work with environment map."
```

---

### Task 3: Fix tone mapping double-application in edge composer

**Files:**
- Modify: `viewer/presentation.ts:1-5` (imports)
- Modify: `viewer/presentation.ts:237-246` (composer pass setup)
- Modify: `viewer/presentation.ts:251-296` (edge composer render function)

The `EffectComposer` pipeline double-applies tone mapping: once in `RenderPass` and once when the final pass writes to screen. The canonical Three.js fix is to use `OutputPass` as the final composer pass and disable tone mapping on the renderer during the entire composer render.

- [ ] **Step 1: Add OutputPass import**

In `presentation.ts`, add to the imports:

```typescript
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
```

- [ ] **Step 2: Add OutputPass as final composer pass**

After the edge pass is added to the composer (around line 246, after `composer.addPass(edgePass);`), add:

```typescript
const outputPass = new OutputPass();
composer.addPass(outputPass);
```

The `OutputPass` handles tone mapping and color space conversion as the final step, so we disable it on the renderer during the entire composer render.

- [ ] **Step 3: Disable tone mapping on renderer during all composer passes**

Replace the entire `render()` body (lines 251-296) with:

```typescript
render() {
  // Disable tone mapping on renderer — OutputPass handles it as the final step.
  // This prevents double-application (RenderPass + screen output).
  const origToneMapping = renderer.toneMapping;
  renderer.toneMapping = THREE.NoToneMapping;

  // Hide non-mesh objects and the section viz plane for edge passes.
  // Hide non-mesh objects AND section cap geometry from edge passes.
  // Cap triangles create false edges in the normal/depth detection.
  const hidden: THREE.Object3D[] = [];
  scene.traverse((child: any) => {
    if (
      child.visible &&
      (child.isGridHelper ||
        child.isLineSegments ||
        child.isLine ||
        child.isSprite ||
        child.isPoints ||
        child.constructor.name === 'LineSegments2' ||
        child.userData._vpSec ||
        child.userData._vpCap)
    ) {
      child.visible = false;
      hidden.push(child);
    }
  });

  // 1. Render normals
  const origOverride = scene.overrideMaterial;
  const origBg = scene.background;
  scene.overrideMaterial = normalMaterial;
  scene.background = null;
  renderer.setRenderTarget(normalTarget);
  renderer.render(scene, camera);

  // 2. Render depth
  scene.overrideMaterial = depthMaterial;
  renderer.setRenderTarget(depthTarget);
  renderer.render(scene, camera);

  // 3. Restore and composite
  scene.overrideMaterial = origOverride;
  scene.background = origBg;
  for (const obj of hidden) obj.visible = true;
  renderer.setRenderTarget(null);

  edgePass.uniforms.tNormal.value = normalTarget.texture;
  edgePass.uniforms.tDepth.value = depthTarget.texture;

  // OutputPass (final) applies tone mapping + color space once.
  composer.render();
  renderer.toneMapping = origToneMapping;
},
```

Key: tone mapping stays disabled on the renderer for the entire `composer.render()` call. The `OutputPass` at the end of the composer chain applies tone mapping exactly once.

- [ ] **Step 2: Verify lint + typecheck**

Run: `make lint`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add viewer/presentation.ts
git commit -m "fix(viewer): prevent double tone mapping in edge detection composer

Save/restore renderer.toneMapping around normal+depth passes so
tone mapping is only applied once via the RenderPass color pass."
```

---

### Task 4: Enable soft shadows with fitted frustum

**Files:**
- Modify: `viewer/viewport3d.ts:647-648` (shadow map type)
- Modify: `viewer/viewport3d.ts:660-661` (directional light shadow setup)

- [ ] **Step 1: Switch to PCFSoftShadowMap**

In `viewport3d.ts`, change line 648:

```typescript
// Before:
this._ren.shadowMap.type = THREE.PCFShadowMap;

// After:
this._ren.shadowMap.type = THREE.PCFSoftShadowMap;
```

- [ ] **Step 2: Configure shadow camera on the directional light**

After the directional light creation (after the line `this._dirLight = dir;` added in Task 2), add:

```typescript
dir.castShadow = true;
dir.shadow.mapSize.width = 1024;
dir.shadow.mapSize.height = 1024;
dir.shadow.bias = -0.0005;
// Shadow camera frustum — sized for typical bot (~0.3m).
// Covers a 0.4m box centered on origin, 0.5m tall.
dir.shadow.camera.left = -0.2;
dir.shadow.camera.right = 0.2;
dir.shadow.camera.top = 0.5;
dir.shadow.camera.bottom = -0.05;
dir.shadow.camera.near = 0.01;
dir.shadow.camera.far = 1.5;
```

This is a reasonable static frustum for the typical bot scale. Dynamic fitting from `bot-viewer.ts` after geometry loads (as mentioned in the spec) is deferred — the static frustum covers the common case and avoids coupling viewport to bot-specific loading logic.

- [ ] **Step 3: Verify lint + typecheck**

Run: `make lint`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add viewer/viewport3d.ts
git commit -m "feat(viewer): enable soft shadows with fitted shadow camera

Switch to PCFSoftShadowMap, configure 1024 shadow map, set bias
and shadow camera frustum sized for typical bot geometry."
```

---

### Task 5: Visual verification and iteration

This task is interactive — load the viewer with a bot and evaluate the render quality with the user.

- [ ] **Step 1: Start the dev server and load viewer**

Run: `make serve` (or however the viewer dev server starts)
Open the bot viewer in browser.

- [ ] **Step 2: Check baseline render**

Verify:
- Environment map loaded (check console for errors, check specular highlights on materials)
- Tone mapping active (colors should look more natural/cinematic)
- Soft shadows visible under the bot
- Edge detection still works correctly (no visual artifacts from tone mapping fix)

- [ ] **Step 3: Compare HDRI presets**

Try each preset by editing the `loadEnvironment` call in viewport3d.ts:
- `artist_workshop` — warm workshop tones
- `studio_small_09` — neutral studio
- `old_bus_depot` — industrial/garage

Get user feedback on preferred preset.

- [ ] **Step 4: Compare tone mapping modes**

Try swapping in viewport3d.ts:
- `THREE.ACESFilmicToneMapping` — warm, cinematic
- `THREE.AgXToneMapping` — accurate, neutral

Get user feedback.

- [ ] **Step 5: Fine-tune parameters**

Adjust based on feedback:
- `scene.environmentIntensity` — IBL strength
- `renderer.toneMappingExposure` — overall brightness
- Ambient light intensity — fill lighting
- Shadow bias — acne vs peter-panning

- [ ] **Step 6: Ablation**

Disable each layer one at a time to show the user what each contributes:
1. Remove environment map (`scene.environment = null`)
2. Remove tone mapping (`renderer.toneMapping = NoToneMapping`)
3. Revert to `PCFShadowMap`

- [ ] **Step 7: Final commit with tuned values**

```bash
git add viewer/
git commit -m "tune: finalize render quality parameters after visual review"
```

---

### Task 6: Run validation

- [ ] **Step 1: Run full lint + typecheck**

Run: `make lint`
Expected: All pass

- [ ] **Step 2: Run test suite**

Run: `make validate`
Expected: All pass

- [ ] **Step 3: Final commit if any fixes needed**
