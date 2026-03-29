/**
 * Central icon module — imports tabler SVGs via Vite's ?raw suffix
 * and exports sized icon collections for toolbar, tree, UI, and DFM.
 */

// ── DFM severity icons ──
import alertCircleSvg from '@tabler/icons/outline/alert-circle.svg?raw';
import alertTriangleSvg from '@tabler/icons/outline/alert-triangle.svg?raw';
// ── Tree node icons ──
import assemblySvg from '@tabler/icons/outline/assembly.svg?raw';
import batterySvg from '@tabler/icons/outline/battery.svg?raw';
import boltSvg from '@tabler/icons/outline/bolt.svg?raw';
import cameraSvg from '@tabler/icons/outline/camera.svg?raw';
// ── UI icons ──
import chevronRightSvg from '@tabler/icons/outline/chevron-right.svg?raw';
import circleDotSvg from '@tabler/icons/outline/circle-dot.svg?raw';
import circuitMotorSvg from '@tabler/icons/outline/circuit-motor.svg?raw';
import codeSvg from '@tabler/icons/outline/code.svg?raw';
import cpuSvg from '@tabler/icons/outline/cpu.svg?raw';
import cubeSvg from '@tabler/icons/outline/cube.svg?raw';
import eyeSvg from '@tabler/icons/outline/eye.svg?raw';
import eyeOffSvg from '@tabler/icons/outline/eye-off.svg?raw';
import ghostSvg from '@tabler/icons/outline/ghost.svg?raw';
import infoCircleSvg from '@tabler/icons/outline/info-circle.svg?raw';
import layersLinkedSvg from '@tabler/icons/outline/layers-linked.svg?raw';
import layersSubtractSvg from '@tabler/icons/outline/layers-subtract.svg?raw';
import packageSvg from '@tabler/icons/outline/package.svg?raw';
import plugConnectedSvg from '@tabler/icons/outline/plug-connected.svg?raw';
// ── Toolbar icons ──
import pointerSvg from '@tabler/icons/outline/pointer.svg?raw';
import propellerSvg from '@tabler/icons/outline/propeller.svg?raw';
import rulerMeasureSvg from '@tabler/icons/outline/ruler-measure.svg?raw';
import sectionSvg from '@tabler/icons/outline/section.svg?raw';
import settingsSvg from '@tabler/icons/outline/settings.svg?raw';
import wheelSvg from '@tabler/icons/outline/wheel.svg?raw';
import xSvg from '@tabler/icons/outline/x.svg?raw';

/** Resize a raw tabler SVG string to the given pixel dimensions. */
export function sizedIcon(rawSvg: string, size: number): string {
  return rawSvg.replace(/width="\d+"/, `width="${size}"`).replace(/height="\d+"/, `height="${size}"`);
}

// ── Toolbar ──

export const TOOLBAR_ICONS = {
  select: sizedIcon(pointerSvg, 20),
  measure: sizedIcon(rulerMeasureSvg, 20),
  section: sizedIcon(sectionSvg, 20),
  transparent: sizedIcon(ghostSvg, 20),
  settings: sizedIcon(settingsSvg, 20),
};

// ── Tree nodes ──

export const TREE_ICONS: Record<string, { svg: string; color: string }> = {
  assembly: { svg: sizedIcon(assemblySvg, 14), color: '#5C7080' },
  body: { svg: sizedIcon(cubeSvg, 14), color: '#2B95D6' },
  servo: { svg: sizedIcon(circuitMotorSvg, 14), color: '#182026' },
  horn: { svg: sizedIcon(propellerSvg, 14), color: '#E8E8E8' },
  mount: { svg: sizedIcon(packageSvg, 14), color: '#0F9960' },
  battery: { svg: sizedIcon(batterySvg, 14), color: '#0F9960' },
  camera: { svg: sizedIcon(cameraSvg, 14), color: '#0F9960' },
  compute: { svg: sizedIcon(cpuSvg, 14), color: '#0F9960' },
  component: { svg: sizedIcon(packageSvg, 14), color: '#0F9960' },
  wheel: { svg: sizedIcon(wheelSvg, 14), color: '#0F9960' },
  fastener: { svg: sizedIcon(boltSvg, 14), color: '#D4A843' },
  wire: { svg: sizedIcon(plugConnectedSvg, 14), color: '#9179F2' },
  joint: { svg: sizedIcon(circleDotSvg, 14), color: '#0A6640' },
  design_layer: { svg: sizedIcon(layersLinkedSvg, 14), color: '#CED9E0' },
  clearance: { svg: sizedIcon(layersSubtractSvg, 14), color: '#F55656' },
};

// ── UI icons ──

// ── Category chip icons (10px, for filter chips) ──

export const CHIP_ICONS: Record<string, string> = {
  body: sizedIcon(cubeSvg, 10),
  servo: sizedIcon(circuitMotorSvg, 10),
  mount: sizedIcon(packageSvg, 10),
  fastener: sizedIcon(boltSvg, 10),
  wire: sizedIcon(plugConnectedSvg, 10),
  design_layer: sizedIcon(layersLinkedSvg, 10),
  clearance: sizedIcon(layersSubtractSvg, 10),
};

export const CHEVRON_RIGHT = sizedIcon(chevronRightSvg, 10);
export const EYE_ICON = sizedIcon(eyeSvg, 12);
export const EYE_OFF_ICON = sizedIcon(eyeOffSvg, 12);
export const CODE_ICON = sizedIcon(codeSvg, 12);
export const CLEAR_ICON = sizedIcon(xSvg, 14);

// ── DFM severity ──

export const SEVERITY_ICONS: Record<string, string> = {
  error: sizedIcon(alertCircleSvg, 14),
  warning: sizedIcon(alertTriangleSvg, 14),
  info: sizedIcon(infoCircleSvg, 14),
};

export const SEVERITY_COLORS: Record<string, string> = {
  error: '#F55656',
  warning: '#D4A843',
  info: '#137CBD',
};
