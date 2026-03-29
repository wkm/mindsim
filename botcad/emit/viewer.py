"""Emit viewer_manifest.json for the 3D web viewer.

Walks the kinematic tree and produces joint metadata, IK chain definitions,
and enriched body/joint/mount data for the CAD-app-style explore mode.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from botcad.component import ComponentKind
from botcad.geometry import PackingResult, Pose
from botcad.ids import BodyId
from botcad.skeleton import Body, Bot, Joint

log = logging.getLogger(__name__)

# Wire stub colors by bus type (RGBA, 0-1 range)
_BUS_TYPE_COLORS: dict[str, list[float]] = {
    "uart_half_duplex": [0.20, 0.60, 0.86, 1.0],  # blue — servo bus
    "csi": [0.40, 0.73, 0.42, 1.0],  # green — camera ribbon
    "power": [0.90, 0.30, 0.25, 1.0],  # red — power
}


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
        specs["fov_deg"] = comp.fov
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


def _fastener_entry(
    owner_name: str,
    tag: str,
    index: int,
    mp,
    fp: Pose,
    body_name: BodyId,
) -> dict:
    """Build a manifest entry for a fastener at a computed Pose.

    Works for both joint-mounted fasteners (bracket/horn/rear) and
    mount-attached fasteners.  The Pose is computed by fastener_pose().
    """
    from botcad.fasteners import fastener_key, fastener_stl_stem

    # Context labels for fastener categories
    context_labels = {
        "ear": "bracket ear -> servo case",
        "horn": "coupler -> horn",
        "rear": "rear horn -> servo case",
        "mount": f"{owner_name} mount -> {body_name}",
    }

    entry: dict = {
        "id": f"fastener_{owner_name}_{tag}_{index}",
        "name": f"{fastener_key(mp)[0]} {fastener_key(mp)[1] or 'SHC'}",
        "kind": "purchased",
        "category": "fastener",
        "parent_body": body_name,
        "mesh": f"{fastener_stl_stem(mp)}.stl",
        "pos": _round_vec(fp.pos),
        "quat": _round_vec(fp.quat),
        "material": "steel",
        "context": context_labels.get(tag, "fastener"),
    }
    # For joint fasteners, include the joint reference
    if tag in ("ear", "horn", "rear"):
        entry["joint"] = owner_name
    else:
        entry["mount_label"] = owner_name
    return entry


def build_viewer_manifest(bot: Bot, packing: PackingResult | None = None) -> dict:
    """Build the viewer manifest dict from a Bot object.

    Returns the manifest as a plain dict, ready for JSON serialization
    or direct use as an API response.
    """

    if packing is None:
        packing = bot.packing_result

    # Cache multi-material emitter results (keyed by ComponentKind value)
    # so we call each emitter at most once across materials + parts.
    mm_cache: dict = {}

    manifest = {
        "bot_name": bot.name,
        "bodies": [],
        "joints": [],
        "mounts": [],
        "parts": [],
        "ik_chains": [],
        "materials": _build_materials_dict(bot, mm_cache),
    }

    # Walk tree to build body/joint lists
    def _walk_body(body: Body, parent_name: BodyId | None, joint: Joint | None) -> None:
        # Body entry — enriched with shape, dimensions, mass
        body_entry = {
            "name": body.name,
            "mesh": f"{body.name}.stl",
            "role": "structure",
            "parent": parent_name,
            "shape": str(body.shape),
            "dimensions": _round_vec(body.dimensions),
            "mass": round(body.solved_mass, 4),
            "pos": _round_vec(body.world_pos),
            "quat": _round_vec(body.world_quat),
        }
        # Emit body material color for rendering
        if body.material is not None:
            body_entry["color"] = list(body.material.color)
        if joint:
            body_entry["joint"] = joint.name

        # Inline mount metadata on the body (for component details)
        body_mounts = []
        for mount in body.mounts:
            comp = mount.component
            mount_meta = {
                "label": mount.label,
                "component_name": comp.name,
                "dimensions": _round_vec(comp.dimensions),
                "mass": round(comp.mass, 4),
            }
            mount_meta.update(_component_specs(comp))
            body_mounts.append(mount_meta)
        if body_mounts:
            body_entry["mounts"] = body_mounts

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
                "design_layers": [
                    {
                        "kind": "bracket",
                        "mesh": f"bracket_{joint.name}.stl",
                        "parent_body": parent_name,
                    },
                    {
                        "kind": "coupler",
                        "mesh": f"coupler_{joint.name}.stl",
                        "parent_body": parent_name,
                    },
                    {
                        "kind": "clearance",
                        "mesh": f"clearance_{joint.name}.stl",
                        "parent_body": parent_name,
                    },
                    {
                        "kind": "insertion",
                        "mesh": f"insertion_{joint.name}.stl",
                        "parent_body": parent_name,
                    },
                ],
            }
            manifest["joints"].append(joint_entry)

        # Recurse into child joints
        for child_joint in body.joints:
            if child_joint.child is not None:
                _walk_body(child_joint.child, body.name, child_joint)

    if bot.root:
        _walk_body(bot.root, None, None)

    # --- Build component bodies and mounts ---

    from botcad.skeleton import BodyKind

    # 1. Component bodies: servos, horns, and purchased bodies that are their
    #    own kinematic node (e.g. wheels).  These go into bodies[] with
    #    role="component" instead of the old parts[] array.
    # 2. Mounts: purchased bodies that share a structural body's kinematic node
    #    (camera, battery, compute mounted ON a structural body).  These go into
    #    the new mounts[] array.

    for body in bot.all_bodies:
        if body.kind != BodyKind.PURCHASED:
            continue

        comp = body.component

        if body.name.startswith("servo_"):
            joint_name = body.name[len("servo_") :]
            manifest["bodies"].append(
                {
                    "name": body.name,
                    "mesh": body.mesh_file,
                    "role": "component",
                    "component": comp.name if comp else body.name,
                    "parent": body.parent_body_name,
                    "category": "servo",
                    "joint": joint_name,
                    "pos": _round_vec(body.world_pos),
                    "quat": _round_vec(body.world_quat),
                    "mass": round(comp.mass, 4) if comp else 0.0,
                    "color": list(comp.default_material.color)
                    if comp and comp.default_material
                    else None,
                    "shapescript_component": comp.name if comp else body.name,
                }
            )
        elif body.name.startswith("horn_"):
            joint_name = body.name[len("horn_") :]
            manifest["bodies"].append(
                {
                    "name": body.name,
                    "mesh": body.mesh_file,
                    "role": "component",
                    "component": "Horn disc",
                    "parent": body.parent_body_name,
                    "category": "horn",
                    "joint": joint_name,
                    "pos": _round_vec(body.world_pos),
                    "quat": _round_vec(body.world_quat),
                    "color": list(comp.default_material.color)
                    if comp and comp.default_material
                    else None,
                    "shapescript_component": f"horn:{comp.name}" if comp else body.name,
                }
            )
        elif body.name.startswith("comp_"):
            # Mounted component — goes into mounts[]
            prefix = f"comp_{body.parent_body_name}_"
            mount_label = (
                body.name[len(prefix) :] if body.name.startswith(prefix) else body.name
            )
            category = (
                _component_specs(comp).get("component_type", "component")
                if comp
                else "component"
            )

            # Look up mount orientation from PackingResult
            mount_quat = [1.0, 0.0, 0.0, 0.0]
            if packing is not None:
                # Find the mount matching this label on the parent body
                parent_body = next(
                    (b for b in bot.all_bodies if b.name == body.parent_body_name),
                    None,
                )
                if parent_body is not None:
                    for m in parent_body.mounts:
                        if m.label == mount_label and m in packing.placements:
                            mount_quat = list(packing.placements[m].pose.quat)
                            break

            mount_entry = {
                "body": body.parent_body_name,
                "label": mount_label,
                "component": comp.name if comp else body.name,
                "category": category,
                "mesh": body.mesh_file,
                "pos": _round_vec(body.world_pos),
                "quat": _round_vec(mount_quat),
                "mass": round(comp.mass, 4) if comp else 0.0,
                "color": list(comp.default_material.color)
                if comp and comp.default_material
                else None,
                "shapescript_component": comp.name if comp else body.name,
            }

            # Multi-material meshes
            if comp is not None:
                mm_result = _get_multi_material_result(comp, mm_cache)
                if mm_result is not None:
                    meshes = [
                        {
                            "file": f"{body.name}__{mp.material.name}.stl",
                            "material": mp.material.name,
                        }
                        for mp in mm_result.material_programs
                    ]
                    if meshes:
                        mount_entry["meshes"] = meshes

            manifest["mounts"].append(mount_entry)

    # 3. Fasteners at each joint (bracket screws + horn screws + rear horn screws)
    if packing is not None:
        from botcad.geometry import fastener_pose

        for body in bot.all_bodies:
            for joint in body.joints:
                if joint not in packing.placements:
                    continue
                servo = joint.servo
                servo_pose = packing.placements[joint].pose
                # Servo pose is in body-local frame; shift to world frame
                bwp = body.world_pos
                from botcad.geometry import Pose

                world_servo_pose = Pose(
                    pos=(
                        bwp[0] + servo_pose.pos[0],
                        bwp[1] + servo_pose.pos[1],
                        bwp[2] + servo_pose.pos[2],
                    ),
                    quat=servo_pose.quat,
                )
                for i, ear in enumerate(servo.mounting_ears):
                    fp = fastener_pose(world_servo_pose, ear)
                    manifest["parts"].append(
                        _fastener_entry(joint.name, "ear", i, ear, fp, body.name)
                    )
                for i, mp in enumerate(servo.horn_mounting_points):
                    fp = fastener_pose(world_servo_pose, mp)
                    manifest["parts"].append(
                        _fastener_entry(joint.name, "horn", i, mp, fp, body.name)
                    )
                for i, mp in enumerate(servo.rear_horn_mounting_points):
                    fp = fastener_pose(world_servo_pose, mp)
                    manifest["parts"].append(
                        _fastener_entry(joint.name, "rear", i, mp, fp, body.name)
                    )

        # 3b. Fasteners for mounted components
        for body in bot.all_bodies:
            bwp = body.world_pos
            for mount in body.mounts:
                if mount not in packing.placements:
                    continue
                mount_pose = packing.placements[mount].pose
                # Mount pose is in body-local frame; shift to world frame
                world_mount_pose = Pose(
                    pos=(
                        bwp[0] + mount_pose.pos[0],
                        bwp[1] + mount_pose.pos[1],
                        bwp[2] + mount_pose.pos[2],
                    ),
                    quat=mount_pose.quat,
                )
                for i, mp in enumerate(mount.component.mounting_points):
                    fp = fastener_pose(world_mount_pose, mp)
                    manifest["parts"].append(
                        _fastener_entry(mount.label, "mount", i, mp, fp, body.name)
                    )

    # 4. Wire routes (one entry per route, world coords, parented to root)
    net_labels = {net.label for net in bot.wire_nets}
    root_body_name = str(bot.root.name) if bot.root else ""
    for route in bot.wire_routes:
        if route.label not in net_labels:
            continue
        if not route.segments:
            continue
        bus_color = _BUS_TYPE_COLORS.get(str(route.bus_type), [0.53, 0.53, 0.53, 1.0])
        manifest["parts"].append(
            {
                "id": f"wire_{route.label}",
                "name": route.label,
                "kind": "fabricated",
                "category": "wire",
                "bus_type": str(route.bus_type),
                "route_label": route.label,
                "parent_body": root_body_name,
                "mesh": f"wire_{route.label}.stl",
                "color": bus_color,
            }
        )

    # Build IK chains — find longest serial chains of hinge joints
    _build_ik_chains(bot, manifest)

    return manifest


def emit_viewer_manifest(bot: Bot, output_dir: Path) -> None:
    """Generate viewer_manifest.json in the bot's output directory."""
    manifest = build_viewer_manifest(bot)
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
