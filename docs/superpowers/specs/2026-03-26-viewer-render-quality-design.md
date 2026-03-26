# Viewer Render Quality: Environment Map + Tone Mapping + Soft Shadows

**Date:** 2026-03-26
**Status:** Draft

## Goal

Improve the Three.js bot viewer's visual quality and realism by adding environment-based lighting (IBL), tone mapping, and soft shadows. The viewer should make robot designs easier to understand by giving materials realistic depth, specular highlights, and grounding shadows.

## Current State

- **Materials:** `MeshPhysicalMaterial` with roughness 0.6, metalness 0.1 — physically-based but uniform
- **Lighting:** Ambient (1.0) + key directional (1.6) + fill directional (0.5) — flat, no environment
- **Shadows:** `PCFShadowMap` enabled on key light, default 512 map size
- **Post-processing:** Roberts cross edge detection (normals + depth) — kept as-is
- **No environment map, no tone mapping, no AO**
- **Three.js version:** 0.183

## Design

### 1. Environment Map (IBL)

New module `viewer/environment.ts` that loads an HDRI and applies it as image-based lighting.

**HDRI source:** Poly Haven CDN — free CC0 HDRIs via direct URL:
```
https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/{resolution}/{name}_{resolution}.hdr
```

**Loading pipeline:**
1. `RGBELoader` (from `three/addons/loaders/RGBELoader.js`) fetches the .hdr file
2. `PMREMGenerator` converts to prefiltered mipmap cubemap for IBL
3. Assign result to `scene.environment` — all `MeshPhysicalMaterial` meshes pick it up automatically
4. `scene.background` stays as current solid color (`0xf5f8fa`) — no visible skybox

**Default HDRI:** Workshop/garage style at 1k resolution (~1-2MB). Warm tones to make mechanical parts feel tangible. We'll try 2-3 alternatives during iteration.

**Caching:** Browser HTTP cache handles repeat loads. Loading is async during scene init.

### 2. Tone Mapping

Set on the `WebGLRenderer` after creation:
- `renderer.toneMapping` — try `THREE.ACESFilmicToneMapping` and `THREE.AgXToneMapping`
- `renderer.toneMappingExposure` — default 1.0, may need tuning per HDRI

**ACES Filmic:** Industry standard, slightly warm/cinematic feel.
**AgX:** Newer, better color preservation in highlights, more perceptually accurate.

We'll compare both with the user and pick one.

### 3. Soft Shadows

Three changes to the existing shadow setup in `Viewport3D`:

1. **Shadow type:** `THREE.PCFShadowMap` → `THREE.PCFSoftShadowMap` (softer penumbra)
2. **Shadow map size:** Default (512) → 1024 or 2048 for crisper shadows
3. **Shadow camera fitting:** After bot geometry loads, fit the directional light's shadow camera frustum to the bot's bounding box so shadow resolution isn't wasted on empty space
4. **Bias tuning:** Set `light.shadow.bias` (typically -0.0001 to -0.001) to eliminate shadow acne

### 4. Iteration Plan

Each change is a separate commit. After all three are in place:
- Try 2-3 different HDRIs (workshop, studio, garage)
- Compare ACES vs AgX tone mapping
- Get user feedback on preferred combination
- Ablate layers at the end to understand each one's contribution

## Aesthetic Direction

**Workshop/garage** — warm-toned environment with subtle color variation in reflections. The bot should look like it's sitting on a workbench, not in an abstract void.

**Edge detection stays on** — the Roberts cross outlines serve design comprehension. They coexist with realistic materials as a hybrid style.

## Files Changed

- **New:** `viewer/environment.ts` — HDRI loading + PMREMGenerator pipeline
- **Modified:** `viewer/viewport3d.ts` — tone mapping config, soft shadow config, call environment loader
- **Modified:** `viewer/bot-viewer.ts` — shadow camera fitting after geometry loads (needs bounding box)

## Non-Goals

- No material differentiation (metal vs plastic) in this work — that's a follow-up
- No SSAO or additional post-processing passes
- No runtime settings UI — we'll ablate by editing code
- No HDRI as visible background/skybox
