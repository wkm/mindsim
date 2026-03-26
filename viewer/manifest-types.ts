/**
 * Shared manifest types — single source of truth for all viewer manifest interfaces.
 *
 * These mirror the Python output from botcad/emit/viewer.py.
 * Import from here instead of redeclaring in individual modules.
 */

// ---------------------------------------------------------------------------
// Manifest sub-types
// ---------------------------------------------------------------------------

export interface ManifestMaterial {
  color: [number, number, number];
  metallic: number;
  roughness: number;
  opacity: number;
}

export interface ManifestMesh {
  file: string;
  material: string;
}

export interface ManifestBodyMount {
  label: string;
  component_name: string;
  component_type: string;
}

export interface ManifestBody {
  name: string;
  mesh: string;
  role: 'structure' | 'component';
  parent: string | null;
  pos: number[];
  quat: number[]; // wxyz
  color?: [number, number, number];
  component?: string;
  category?: string;
  joint?: string;
  mounts?: ManifestBodyMount[];
  shapescript_component?: string;
}

export interface ManifestMount {
  body: string;
  label: string;
  component: string;
  category: string;
  mesh: string;
  pos: number[];
  quat: number[]; // wxyz
  meshes?: ManifestMesh[];
  shapescript_component?: string;
  color?: [number, number, number, number];
}

export interface ManifestPart {
  id: string;
  name: string;
  category: string;
  parent_body: string;
  joint?: string;
  mount_label?: string;
  mesh: string;
  pos: number[];
  quat: number[]; // wxyz
  meshes?: ManifestMesh[];
}

export interface ManifestDesignLayer {
  kind: string;
  mesh: string;
  parent_body: string;
}

export interface ManifestJoint {
  name: string;
  parent_body: string;
  child_body: string;
  servo?: string;
  design_layers?: ManifestDesignLayer[];
}

export interface ManifestAssembly {
  name: string;
  path: string;
  bodies: string[];
  sub_assemblies: ManifestAssembly[];
}

export interface ViewerManifest {
  bot_name: string;
  bodies: ManifestBody[];
  joints: ManifestJoint[];
  mounts?: ManifestMount[];
  parts?: ManifestPart[];
  materials?: Record<string, ManifestMaterial>;
  assemblies?: ManifestAssembly[];
}

// ---------------------------------------------------------------------------
// Manifest indexing — shared lookup tables built once from a manifest
// ---------------------------------------------------------------------------

export interface ManifestIndex {
  bodiesByName: Record<string, ManifestBody>;
  childJointsOf: Record<string, ManifestJoint[]>;
  servosByJoint: Record<string, ManifestBody[]>;
  hornsByJoint: Record<string, ManifestBody[]>;
  mountsByBody: Record<string, ManifestMount[]>;
  partsByBody: Record<string, ManifestPart[]>;
  partsByJoint: Record<string, ManifestPart[]>;
}

/** Build lookup indices from a viewer manifest. */
export function indexManifest(manifest: ViewerManifest): ManifestIndex {
  const mounts = manifest.mounts ?? [];
  const parts = manifest.parts ?? [];

  const bodiesByName: Record<string, ManifestBody> = {};
  for (const b of manifest.bodies) bodiesByName[b.name] = b;

  const childJointsOf: Record<string, ManifestJoint[]> = {};
  for (const j of manifest.joints) {
    if (!childJointsOf[j.parent_body]) childJointsOf[j.parent_body] = [];
    childJointsOf[j.parent_body].push(j);
  }

  const servosByJoint: Record<string, ManifestBody[]> = {};
  const hornsByJoint: Record<string, ManifestBody[]> = {};
  for (const b of manifest.bodies) {
    if (b.role !== 'component' || !b.joint) continue;
    if (b.category === 'servo') {
      if (!servosByJoint[b.joint]) servosByJoint[b.joint] = [];
      servosByJoint[b.joint].push(b);
    } else if (b.category === 'horn') {
      if (!hornsByJoint[b.joint]) hornsByJoint[b.joint] = [];
      hornsByJoint[b.joint].push(b);
    }
  }

  const mountsByBody: Record<string, ManifestMount[]> = {};
  for (const m of mounts) {
    if (!mountsByBody[m.body]) mountsByBody[m.body] = [];
    mountsByBody[m.body].push(m);
  }

  const partsByBody: Record<string, ManifestPart[]> = {};
  const partsByJoint: Record<string, ManifestPart[]> = {};
  for (const p of parts) {
    if (!partsByBody[p.parent_body]) partsByBody[p.parent_body] = [];
    partsByBody[p.parent_body].push(p);
    if (p.joint) {
      if (!partsByJoint[p.joint]) partsByJoint[p.joint] = [];
      partsByJoint[p.joint].push(p);
    }
  }

  return { bodiesByName, childJointsOf, servosByJoint, hornsByJoint, mountsByBody, partsByBody, partsByJoint };
}
