/**
 * Centralized 3D presentation layer for MindSim viewer.
 *
 * Single source of truth for how geometry is rendered: materials,
 * edge outlines, color tinting, and Blueprint palette constants.
 *
 * Edge rendering uses post-processing normal+depth edge detection
 * (the same technique used by Fusion 360, OnShape, etc.) instead of
 * geometric edge extraction. This correctly handles silhouettes,
 * fillets, and arbitrary smooth geometry.
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

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
// ---------------------------------------------------------------------------
export const RENDER_ORDER = {
  SECTION_VIZ:      -100,
  STENCIL_BACK:        0,
  STENCIL_FRONT:       1,
  STENCIL_CAP:         2,
  SECTION_CONTOUR:  9000,
};
export const SECTION_STENCIL_BASE = 100;
export const SECTION_STENCIL_STRIDE = 10;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/** Convert a numeric hex color (e.g. 0xFF0000) to a CSS hex string ('#ff0000'). */
export function hexStr(n) { return '#' + n.toString(16).padStart(6, '0'); }

// ---------------------------------------------------------------------------
// Edge rendering constants
// ---------------------------------------------------------------------------
export const EDGE_COLOR = BP.DARK_GRAY1;
export const EDGE_OPACITY = 0.85;
export const EDGE_THICKNESS = 1.0;  // edge line thickness (shader parameter)

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
export function tintColor(entityColor, tintFraction = 0.5, baseHex = BP.LIGHT_GRAY5) {
  const base = new THREE.Color(baseHex);
  const entity = entityColor instanceof THREE.Color
    ? entityColor
    : new THREE.Color(entityColor);
  return base.lerp(entity, tintFraction);
}

// ---------------------------------------------------------------------------
// Material creation
// ---------------------------------------------------------------------------

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

export function createToolMaterial(color, opacity = 0.3) {
  return createMaterial(color, { transparent: true, opacity });
}

/**
 * Create a mesh with material. Edge rendering is handled by the
 * post-processing pipeline, not per-mesh geometry.
 */
export function addMeshWithEdges(geometry, color, group, opts = {}) {
  const material = createMaterial(color, opts);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = !opts.wireframe;
  mesh.receiveShadow = !opts.wireframe;
  group.add(mesh);
  return mesh;
}

// ---------------------------------------------------------------------------
// Post-processing edge detection (normal + depth)
//
// Technique: render scene normals and depth to a texture, then apply a
// screen-space edge detection kernel (Roberts cross) that finds
// discontinuities. This catches feature edges, silhouettes, creases,
// and any geometry boundary — the same approach used by Fusion 360,
// OnShape, and other CAD web viewers.
// ---------------------------------------------------------------------------

/** Edge detection shader — operates on normal+depth render target. */
const EdgeDetectShader = {
  uniforms: {
    tDiffuse:  { value: null },     // color pass
    tNormal:   { value: null },     // normal pass
    tDepth:    { value: null },     // depth pass
    resolution: { value: new THREE.Vector2() },
    edgeColor: { value: new THREE.Color(EDGE_COLOR) },
    edgeOpacity: { value: EDGE_OPACITY },
    normalThreshold: { value: 0.3 },
    depthThreshold: { value: 0.002 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform sampler2D tNormal;
    uniform sampler2D tDepth;
    uniform vec2 resolution;
    uniform vec3 edgeColor;
    uniform float edgeOpacity;
    uniform float normalThreshold;
    uniform float depthThreshold;

    varying vec2 vUv;

    void main() {
      vec2 texel = 1.0 / resolution;

      // Sample normals in a 2x2 neighborhood (Roberts cross)
      vec3 n00 = texture2D(tNormal, vUv).rgb;
      vec3 n10 = texture2D(tNormal, vUv + vec2(texel.x, 0.0)).rgb;
      vec3 n01 = texture2D(tNormal, vUv + vec2(0.0, texel.y)).rgb;
      vec3 n11 = texture2D(tNormal, vUv + vec2(texel.x, texel.y)).rgb;

      // Roberts cross on normals
      float normalEdge = length(n00 - n11) + length(n10 - n01);

      // Sample depth
      float d00 = texture2D(tDepth, vUv).r;
      float d10 = texture2D(tDepth, vUv + vec2(texel.x, 0.0)).r;
      float d01 = texture2D(tDepth, vUv + vec2(0.0, texel.y)).r;
      float d11 = texture2D(tDepth, vUv + vec2(texel.x, texel.y)).r;

      // Roberts cross on depth
      float depthEdge = abs(d00 - d11) + abs(d10 - d01);

      // Combine edge signals
      float edge = 0.0;
      if (normalEdge > normalThreshold) edge = 1.0;
      if (depthEdge > depthThreshold) edge = 1.0;

      // Composite edge on top of color
      vec4 color = texture2D(tDiffuse, vUv);
      vec3 result = mix(color.rgb, edgeColor, edge * edgeOpacity);
      gl_FragColor = vec4(result, color.a);
    }
  `,
};

/**
 * Create a post-processing edge detection pipeline.
 *
 * @param {THREE.WebGLRenderer} renderer
 * @param {THREE.Scene} scene
 * @param {THREE.Camera} camera
 * @returns {{ composer: EffectComposer, resize: (w,h) => void, render: () => void }}
 */
export function createEdgeComposer(renderer, scene, camera) {
  const size = renderer.getSize(new THREE.Vector2());

  // Normal render target
  const normalTarget = new THREE.WebGLRenderTarget(size.x, size.y, {
    type: THREE.FloatType,
  });
  const normalMaterial = new THREE.MeshNormalMaterial();

  // Depth render target
  const depthTarget = new THREE.WebGLRenderTarget(size.x, size.y, {
    type: THREE.FloatType,
  });
  const depthMaterial = new THREE.MeshDepthMaterial({
    depthPacking: THREE.BasicDepthPacking,
  });

  // Composer: color pass + edge detection pass
  const composer = new EffectComposer(renderer);
  const renderPass = new RenderPass(scene, camera);
  composer.addPass(renderPass);

  const edgePass = new ShaderPass(EdgeDetectShader);
  edgePass.uniforms.resolution.value.copy(size);
  edgePass.uniforms.edgeColor.value.set(EDGE_COLOR);
  composer.addPass(edgePass);

  return {
    composer,

    render() {
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
      renderer.setRenderTarget(null);

      edgePass.uniforms.tNormal.value = normalTarget.texture;
      edgePass.uniforms.tDepth.value = depthTarget.texture;

      composer.render();
    },

    resize(w, h) {
      normalTarget.setSize(w, h);
      depthTarget.setSize(w, h);
      composer.setSize(w, h);
      edgePass.uniforms.resolution.value.set(w, h);
    },
  };
}
