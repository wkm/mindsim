import * as THREE from 'three';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

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
    const loader = new HDRLoader();
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
