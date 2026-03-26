"""Emit viewer_manifest.json for the 3D web viewer.

Walks the kinematic tree and produces structured assembly steps,
joint metadata, IK chain definitions, and enriched body/joint/mount data
for the CAD-app-style explore mode.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot, Joint

import logging

from botcad.component import ComponentKind

log = logging.getLogger(__name__)


def _material_to_dict(mat) -> dict:
    """Convert a Material to a viewer-serializable dict."""
    return {
        "color": list(mat.color),
        "metallic": mat.metallic,
        "roughness": mat.roughness,
        "opacity": mat.opacity,
    }


def _component_specs(comp) -> dict:
    """Extract type-specific specs from a Component."""
    specs: dict = {}
    specs["component_type"] = comp.kind.value
    if comp.kind == ComponentKind.CAMERA:
        specs["fov_deg"] = comp.fov_deg
        specs["resolution"] = list(comp.resolution)
    elif comp.kind == ComponentKind.BATTERY:
        specs["chemistry"] = comp.chemistry
        specs["voltage"] = comp.voltage
        specs["cells_s"] = comp.cells_s
    elif comp.kind == ComponentKind.SERVO:
        specs["stall_torque_nm"] = round(comp.stall_torque, 4)
        specs["no_load_speed_rad_s"] = round(comp.no_load_speed, 3)
        specs["voltage"] = comp.voltage
        specs["gear_ratio"] = comp.gear_ratio
    return specs


def _round_vec(v):
    """Round a tuple/list of floats to 6 decimal places (µm precision for meters)."""
    return [round(x, 6) for x in v]


def _build_assembly_tree(bot: Bot) -> list[dict]:
    """Build the assembly hierarchy for the viewer manifest."""
    from botcad.skeleton import Assembly

    def _assembly_dict(asm: Assembly) -> dict:
        # Collect body names belonging to this assembly
        body_names = [b.name for b in bot.all_bodies if b.assembly is asm]
        sub_assemblies = [_assembly_dict(sub) for sub in asm._sub_assemblies.values()]
        return {
            "name": asm.name,
            "path": asm.path,
            "bodies": body_names,
            "sub_assemblies": sub_assemblies,
        }

    return [_assembly_dict(asm) for asm in bot._assemblies.values()]


def _get_multi_material_result(comp, cache: dict):
    """Get multi-material emitter result for a component, with caching by kind.

    Returns the emitter result or None if unavailable/failed.
    """
    from botcad.component import get_component_meta

    kind_key = comp.kind.value
    if kind_key in cache:
        return cache[kind_key]

    meta = get_component_meta(comp.kind)
    if meta.multi_material_emitter is None:
        cache[kind_key] = None
        return None

    try:
        result = meta.multi_material_emitter(comp)
        cache[kind_key] = result
        return result
    except Exception:
        log.debug(
            "Multi-material emitter failed for %s",
            kind_key,
            exc_info=True,
        )
        cache[kind_key] = None
        return None


def _build_materials_dict(bot: Bot, mm_cache: dict) -> dict:
    """Build the materials dictionary for the manifest.

    Collects all unique materials from component default_materials and the
    material catalog, mapping name -> visual properties for the viewer.
    """
    materials: dict[str, dict] = {}

    # Collect from component default_materials
    for body in bot.all_bodies:
        for mount in body.mounts:
            mat = mount.component.default_material
            if mat is not None and mat.name not in materials:
                materials[mat.name] = _material_to_dict(mat)

    # Collect from multi-material emitters
    seen_kinds: set[str] = set()
    for body in bot.all_bodies:
        for mount in body.mounts:
            comp = mount.component
            if comp.kind.value in seen_kinds:
                continue
            seen_kinds.add(comp.kind.value)
            mm_result = _get_multi_material_result(comp, mm_cache)
            if mm_result is not None:
                for mp in mm_result.material_programs:
                    mat = mp.material
                    if mat.name not in materials:
                        materials[mat.name] = _material_to_dict(mat)

    return materials


def emit_viewer_manifest(bot: Bot, output_dir: Path) -> None:
    """Generate viewer_manifest.json in the bot's output directory."""
    from botcad.fasteners import fastener_key, fastener_stl_stem

    # Cache multi-material emitter results (keyed by ComponentKind value)
    # so we call each emitter at most once across materials + parts.
    mm_cache: dict = {}

    manifest = {
        "bot_name": bot.name,
        "assemblies": _build_assembly_tree(bot),
        "bodies": [],
        "joints": [],
        "parts": [],
        "assembly_steps": [],
        "ik_chains": [],
        "materials": _build_materials_dict(bot, mm_cache),
    }

    # Walk tree to build body/joint lists and assembly steps
    step_num = [0]

    def _walk_body(body: Body, parent_name: str | None, joint: Joint | None) -> None:
        # Body entry — enriched with shape, dimensions, mass, mounts
        body_entry = {
            "name": body.name,
            "mesh": f"{body.name}.stl",
            "kind": "fabricated",
            "parent": parent_name,
            "shape": str(body.shape),
            "dimensions": _round_vec(body.dimensions),
            "mass": round(body.solved_mass, 4),
        }
        if joint:
            body_entry["joint"] = joint.name

        # Mounts with full component metadata
        mounts = []
        for mount in body.mounts:
            comp = mount.component
            mount_entry = {
                "label": mount.label,
                "component_name": comp.name,
                "dimensions": _round_vec(comp.dimensions),
                "mass": round(comp.mass, 4),
            }
            mount_entry.update(_component_specs(comp))
            mounts.append(mount_entry)
        if mounts:
            body_entry["mounts"] = mounts

        manifest["bodies"].append(body_entry)

        # Joint entry — enriched with servo specs
        if joint:
            lo, hi = joint.effective_range
            joint_entry = {
                "name": joint.name,
                "parent_body": parent_name,
                "child_body": body.name,
                "axis": list(joint.axis),
                "range": [lo, hi],
                "range_deg": [
                    round(math.degrees(lo), 1),
                    round(math.degrees(hi), 1),
                ],
                "pos": list(joint.pos),
                "servo": joint.servo.name,
                "continuous": joint.servo.continuous,
                "servo_specs": {
                    "stall_torque_nm": round(joint.servo.stall_torque, 4),
                    "no_load_speed_rad_s": round(joint.servo.no_load_speed, 3),
                    "voltage": joint.servo.voltage,
                    "gear_ratio": joint.servo.gear_ratio,
                    "mass": round(joint.servo.mass, 4),
                },
            }
            manifest["joints"].append(joint_entry)

        # Assembly step
        step_num[0] += 1
        step = {
            "step": step_num[0],
            "title": body.name,
            "body": body.name,
        }

        # Components mounted in this body
        components = []
        for mount in body.mounts:
            components.append(
                {
                    "label": mount.label,
                    "component": mount.component.name,
                }
            )
        if components:
            step["components"] = components

        # Servo for the joint connecting this body
        if joint:
            step["servo"] = {
                "joint": joint.name,
                "model": joint.servo.name,
            }
            step["description"] = f"Attach via {joint.servo.name} at joint {joint.name}"
        else:
            step["description"] = "Base structure"
            if components:
                comp_names = ", ".join(c["label"] for c in components)
                step["description"] += f" — mount {comp_names}"

        manifest["assembly_steps"].append(step)

        # Recurse into child joints
        for child_joint in body.joints:
            if child_joint.child is not None:
                _walk_body(child_joint.child, body.name, child_joint)

    if bot.root:
        _walk_body(bot.root, None, None)

    # --- Build parts list: every physical object that isn't a structural body ---

    # 1 & 2. Servos, horns, and mounted components from purchased bodies
    from botcad.skeleton import BodyKind

    for body in bot.all_bodies:
        if body.kind != BodyKind.PURCHASED:
            continue

        # Derive category and extra fields from body name / component
        if body.name.startswith("servo_"):
            joint_name = body.name[len("servo_") :]
            comp = body.component
            part_entry = {
                "id": body.name,
                "name": comp.name if comp else body.name,
                "kind": "purchased",
                "category": "servo",
                "parent_body": body.parent_body_name,
                "joint": joint_name,
                "mesh": body.mesh_file,
                "pos": _round_vec(body.world_pos),
                "quat": _round_vec(body.world_quat),
                "mass": round(comp.mass, 4) if comp else 0.0,
                "shapescript_component": comp.name if comp else body.name,
            }
            manifest["parts"].append(part_entry)
        elif body.name.startswith("horn_"):
            joint_name = body.name[len("horn_") :]
            comp = body.component
            part_entry = {
                "id": body.name,
                "name": "Horn disc",
                "kind": "purchased",
                "category": "horn",
                "parent_body": body.parent_body_name,
                "joint": joint_name,
                "mesh": body.mesh_file,
                "shapescript_component": f"horn:{comp.name}" if comp else body.name,
            }
            manifest["parts"].append(part_entry)
        elif body.name.startswith("comp_"):
            comp = body.component
            # Extract mount_label: body name is "comp_{parent}_{label}"
            # parent_body_name is known, so strip "comp_{parent}_" prefix
            prefix = f"comp_{body.parent_body_name}_"
            mount_label = (
                body.name[len(prefix) :] if body.name.startswith(prefix) else body.name
            )
            category = (
                _component_specs(comp).get("component_type", "component")
                if comp
                else "component"
            )
            part_entry = {
                "id": body.name,
                "name": comp.name if comp else body.name,
                "kind": "purchased",
                "category": category,
                "parent_body": body.parent_body_name,
                "mount_label": mount_label,
                "mesh": body.mesh_file,
                "pos": _round_vec(body.world_pos),
                "mass": round(comp.mass, 4) if comp else 0.0,
                "shapescript_component": comp.name if comp else body.name,
            }

            # Multi-material meshes: if this component has per-material
            # programs, add a `meshes` array alongside the legacy `mesh`.
            if comp is not None:
                mm_result = _get_multi_material_result(comp, mm_cache)
                if mm_result is not None:
                    meshes = []
                    for mp in mm_result.material_programs:
                        meshes.append(
                            {
                                "file": f"{body.name}__{mp.material.name}.stl",
                                "material": mp.material.name,
                            }
                        )
                    if meshes:
                        part_entry["meshes"] = meshes

            manifest["parts"].append(part_entry)

    # 3. Fasteners at each joint (bracket screws + horn screws + rear horn screws)
    for body in bot.all_bodies:
        for joint in body.joints:
            servo = joint.servo
            for i, ear in enumerate(servo.mounting_ears):
                manifest["parts"].append(
                    {
                        "id": f"fastener_{joint.name}_ear_{i}",
                        "name": f"{fastener_key(ear)[0]} {fastener_key(ear)[1] or 'SHC'}",
                        "kind": "purchased",
                        "category": "fastener",
                        "parent_body": body.name,
                        "joint": joint.name,
                        "mesh": f"{fastener_stl_stem(ear)}.stl",
                    }
                )
            for i, mp in enumerate(servo.horn_mounting_points):
                manifest["parts"].append(
                    {
                        "id": f"fastener_{joint.name}_horn_{i}",
                        "name": f"{fastener_key(mp)[0]} {fastener_key(mp)[1] or 'SHC'}",
                        "kind": "purchased",
                        "category": "fastener",
                        "parent_body": body.name,
                        "joint": joint.name,
                        "mesh": f"{fastener_stl_stem(mp)}.stl",
                    }
                )
            for i, mp in enumerate(servo.rear_horn_mounting_points):
                manifest["parts"].append(
                    {
                        "id": f"fastener_{joint.name}_rear_{i}",
                        "name": f"{fastener_key(mp)[0]} {fastener_key(mp)[1] or 'SHC'}",
                        "kind": "purchased",
                        "category": "fastener",
                        "parent_body": body.name,
                        "joint": joint.name,
                        "mesh": f"{fastener_stl_stem(mp)}.stl",
                    }
                )

    # 3b. Fasteners for mounted components
    for body in bot.all_bodies:
        for mount in body.mounts:
            for i, mp in enumerate(mount.component.mounting_points):
                manifest["parts"].append(
                    {
                        "id": f"fastener_{body.name}_{mount.label}_{i}",
                        "name": f"{fastener_key(mp)[0]} {fastener_key(mp)[1] or 'SHC'}",
                        "kind": "purchased",
                        "category": "fastener",
                        "parent_body": body.name,
                        "mount_label": mount.label,
                        "mesh": f"{fastener_stl_stem(mp)}.stl",
                    }
                )

    # 4. Wire segments
    for route in bot.wire_routes:
        for i, seg in enumerate(route.segments):
            if seg.straight_length < 0.001:
                continue
            manifest["parts"].append(
                {
                    "id": f"wire_{route.label}_{seg.body_name}_{i}",
                    "name": route.label,
                    "kind": "fabricated",
                    "category": "wire",
                    "parent_body": seg.body_name,
                    "bus_type": str(route.bus_type),
                    "mesh": f"wire_{route.label}_{seg.body_name}_{i}.stl",
                }
            )

    # Build IK chains — find longest serial chains of hinge joints
    _build_ik_chains(bot, manifest)

    out_path = output_dir / "viewer_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _build_ik_chains(bot: Bot, manifest: dict) -> None:
    """Identify serial kinematic chains for IK mode."""
    if not bot.root:
        return

    # Pre-build set of continuous joint names for O(1) lookup
    continuous_joints = {
        j.name for j in bot.all_joints if getattr(j.servo, "continuous", False)
    }

    def _find_chains(body: Body, chain: list[dict]) -> list[list[dict]]:
        chains = []
        for joint in body.joints:
            if joint.child is not None:
                entry = {"joint": joint.name, "body": joint.child.name}
                child_chains = _find_chains(joint.child, [*chain, entry])
                chains.extend(child_chains)
        if not body.joints and len(chain) >= 2:
            chains.append(chain)
        return chains

    for i, chain in enumerate(_find_chains(bot.root, [])):
        filtered = [e for e in chain if e["joint"] not in continuous_joints]
        if len(filtered) >= 2:
            manifest["ik_chains"].append(
                {
                    "name": f"chain_{i}",
                    "joints": [e["joint"] for e in filtered],
                    "end_effector": filtered[-1]["body"],
                }
            )
