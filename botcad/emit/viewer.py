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

# Wire stub colors by bus type (RGBA, 0-1 range)
_BUS_TYPE_COLORS: dict[str, list[float]] = {
    "uart_half_duplex": [0.20, 0.60, 0.86, 1.0],  # blue — servo bus
    "csi": [0.40, 0.73, 0.42, 1.0],  # green — camera ribbon
    "power": [0.90, 0.30, 0.25, 1.0],  # red — power
    "usb": [0.55, 0.35, 0.75, 1.0],  # purple — USB data/power
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


def _transform_fastener_pos(
    mp_pos: tuple, servo_quat: tuple, servo_center: tuple, body_world_pos: tuple
) -> list:
    """Transform a fastener mount-point position to world frame.

    mp_pos is in the servo's local frame. We rotate by the servo's solved
    quaternion, add the servo center (body-local), then add body world pos.
    """
    from botcad.geometry import rotate_vec

    rotated = rotate_vec(servo_quat, mp_pos)
    return _round_vec(
        (
            body_world_pos[0] + servo_center[0] + rotated[0],
            body_world_pos[1] + servo_center[1] + rotated[1],
            body_world_pos[2] + servo_center[2] + rotated[2],
        )
    )


# Context labels for fastener categories
_FASTENER_CONTEXT = {
    "ear": "bracket ear -> servo case",
    "horn": "coupler -> horn",
    "rear": "rear horn -> servo case",
}


def _fastener_total_length(mp) -> float:
    """Total screw length (head + shank) for a mount point's fastener."""
    from botcad.fasteners import resolve_fastener

    try:
        spec = resolve_fastener(mp)
        head_h = spec.head_height
    except KeyError:
        head_h = 0.003
    shank = 0.004  # default shank length (matches screw_solid default)
    return head_h + shank


def _build_joint_fastener_entry(
    joint_name: str,
    tag: str,
    index: int,
    mp,
    servo_quat: tuple,
    servo_center: tuple,
    body_world_pos: tuple,
    body_name: str,
) -> dict:
    """Build a manifest entry for a joint-mounted fastener (bracket/horn/rear)."""
    from botcad.fasteners import fastener_key, fastener_stl_stem
    from botcad.geometry import quat_multiply, rotate_vec, rotation_between

    # Align fastener +Z (head top face) to the mount point axis direction,
    # then compose with servo orientation to get world-frame quaternion.
    axis_align = rotation_between(
        (0.0, 0.0, 1.0), mp.axis
    )  # plint: disable=no-rotation-between-in-emitters
    final_quat = quat_multiply(servo_quat, axis_align)

    # Position: mount hole center, then offset so screw tip sits at hole depth
    # and head protrudes outward. The screw STL extends from Z=0 (head) to
    # Z=-total_length (tip). After quat rotation, the head faces along the
    # mount axis. Offset along the axis by total_length so the tip aligns
    # with the original mount hole position.
    base_pos = _transform_fastener_pos(mp.pos, servo_quat, servo_center, body_world_pos)
    total_len = _fastener_total_length(mp)
    # The mount axis in world frame = servo_quat applied to mp.axis
    world_axis = rotate_vec(servo_quat, mp.axis)
    offset_pos = (
        base_pos[0] + world_axis[0] * total_len,
        base_pos[1] + world_axis[1] * total_len,
        base_pos[2] + world_axis[2] * total_len,
    )

    return {
        "id": f"fastener_{joint_name}_{tag}_{index}",
        "name": f"{fastener_key(mp)[0]} {fastener_key(mp)[1] or 'SHC'}",
        "kind": "purchased",
        "category": "fastener",
        "parent_body": body_name,
        "joint": joint_name,
        "mesh": f"{fastener_stl_stem(mp)}.stl",
        "pos": _round_vec(offset_pos),
        "quat": _round_vec(final_quat),
        "material": "steel",
        "context": _FASTENER_CONTEXT.get(tag, "fastener"),
    }


def _build_mount_fastener_entry(
    body_name: str,
    mount_label: str,
    index: int,
    mp,
    fastener_pos: list,
    fastener_axis: tuple,
) -> dict:
    """Build a manifest entry for a mount-attached fastener."""
    from botcad.fasteners import fastener_key, fastener_stl_stem
    from botcad.geometry import rotation_between

    # MountPoint.axis = insertion direction (where shank goes).
    # Screw STL head faces +Z, so align +Z with head direction = -axis.
    # (Same convention as MuJoCo emitter, see mujoco.py line 665-668.)
    head_axis = (-fastener_axis[0], -fastener_axis[1], -fastener_axis[2])

    return {
        "id": f"fastener_{body_name}_{mount_label}_{index}",
        "name": f"{fastener_key(mp)[0]} {fastener_key(mp)[1] or 'SHC'}",
        "kind": "purchased",
        "category": "fastener",
        "parent_body": body_name,
        "mount_label": mount_label,
        "mesh": f"{fastener_stl_stem(mp)}.stl",
        "pos": fastener_pos,
        "quat": _round_vec(
            rotation_between((0.0, 0.0, 1.0), head_axis)
        ),  # plint: disable=no-rotation-between-in-emitters
        "material": "steel",
        "context": f"{mount_label} mount -> {body_name}",
    }


def build_viewer_manifest(bot: Bot) -> dict:
    """Build the viewer manifest dict from a Bot object.

    Returns the manifest as a plain dict, ready for JSON serialization
    or direct use as an API response.
    """

    # Cache multi-material emitter results (keyed by ComponentKind value)
    # so we call each emitter at most once across materials + parts.
    mm_cache: dict = {}

    manifest = {
        "bot_name": bot.name,
        "assemblies": _build_assembly_tree(bot),
        "bodies": [],
        "joints": [],
        "mounts": [],
        "parts": [],
        "assembly_steps": [],
        "ik_chains": [],
        "materials": _build_materials_dict(bot, mm_cache),
    }

    # Walk tree to build body/joint lists and assembly steps
    step_num = [0]

    def _walk_body(body: Body, parent_name: str | None, joint: Joint | None) -> None:
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

        # Assembly step
        step_num[0] += 1
        step = {
            "step": step_num[0],
            "title": body.name,
            "body": body.name,
        }

        # Components mounted in this body
        components = [
            {
                "label": mount.label,
                "component": mount.component.name,
            }
            for mount in body.mounts
        ]
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
            mount_entry = {
                "body": body.parent_body_name,
                "label": mount_label,
                "component": comp.name if comp else body.name,
                "category": category,
                "mesh": body.mesh_file,
                "pos": _round_vec(body.world_pos),
                "quat": [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],  # plint: disable=no-hardcoded-identity-quat
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
    for body in bot.all_bodies:
        for joint in body.joints:
            servo = joint.servo
            sq = joint.solved_servo_quat
            sc = joint.solved_servo_center
            bwp = body.world_pos
            for i, ear in enumerate(servo.mounting_ears):
                manifest["parts"].append(
                    _build_joint_fastener_entry(
                        joint.name, "ear", i, ear, sq, sc, bwp, body.name
                    )
                )
            for i, mp in enumerate(servo.horn_mounting_points):
                manifest["parts"].append(
                    _build_joint_fastener_entry(
                        joint.name, "horn", i, mp, sq, sc, bwp, body.name
                    )
                )
            for i, mp in enumerate(servo.rear_horn_mounting_points):
                manifest["parts"].append(
                    _build_joint_fastener_entry(
                        joint.name, "rear", i, mp, sq, sc, bwp, body.name
                    )
                )

    # 3b. Fasteners for mounted components
    for body in bot.all_bodies:
        bwp = body.world_pos
        for mount in body.mounts:
            rp = mount.resolved_pos
            for i, mp in enumerate(mount.component.mounting_points):
                # Rotate mount point from component-local to body frame
                rotated = mount.rotate_point(mp.pos)
                fastener_pos = _round_vec(
                    (
                        bwp[0] + rp[0] + rotated[0],
                        bwp[1] + rp[1] + rotated[1],
                        bwp[2] + rp[2] + rotated[2],
                    )
                )
                fastener_axis = tuple(
                    mount.rotate_point(mp.axis)
                )  # plint: disable=no-rotate-point-for-axes
                manifest["parts"].append(
                    _build_mount_fastener_entry(
                        body.name, mount.label, i, mp, fastener_pos, fastener_axis
                    )
                )

    # 4. Wire segments — identity pos/quat because geometry is already
    # in body-local frame in the per-segment STL. The viewer parents
    # each part under its body group via parent_body.
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
                    "wire_kind": "segment",
                    "parent_body": seg.body_name,
                    "bus_type": str(route.bus_type),
                    "mesh": f"wire_{route.label}_{seg.body_name}_{i}.stl",
                    "pos": [0.0, 0.0, 0.0],
                    "quat": [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ],  # plint: disable=no-hardcoded-identity-quat
                    "color": _BUS_TYPE_COLORS.get(
                        str(route.bus_type), [0.53, 0.53, 0.53, 1.0]
                    ),
                }
            )

    # 5. Wire stubs at connector sockets
    _emit_wire_stubs(bot, manifest)

    # Build IK chains — find longest serial chains of hinge joints
    _build_ik_chains(bot, manifest)

    return manifest


def emit_viewer_manifest(bot: Bot, output_dir: Path) -> None:
    """Generate viewer_manifest.json in the bot's output directory."""
    manifest = build_viewer_manifest(bot)
    out_path = output_dir / "viewer_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _emit_wire_stubs(bot: Bot, manifest: dict) -> None:
    """Emit short wire stub entries at each component's WirePort.

    Wire stubs are short cylinders (~25mm) extending from connector sockets
    along the wire exit direction. They let the viewer show "there's a wire
    here, does it collide?" without full cable routing.

    Each stub is emitted as a proper part with pos/quat/mesh so the viewer
    can render it the same way as any other part.
    """
    from botcad.connectors import connector_spec
    from botcad.geometry import rotate_vec, rotation_between

    for body in bot.all_bodies:
        for mount in body.mounts:
            comp = mount.component
            for wp in comp.wire_ports:
                if not wp.connector_type:
                    continue

                # Look up connector spec for wire exit direction
                try:
                    cspec = connector_spec(wp.connector_type)
                except KeyError:
                    log.debug(
                        "Unknown connector_type %r on %s/%s, skipping wire stub",
                        wp.connector_type,
                        comp.name,
                        wp.label,
                    )
                    continue

                # Transform port position and exit direction from component-local
                # to body-local frame:
                #   1. Apply mount.rotate_point (handles rotate_z + face rotation)
                #   2. Translate by mount.resolved_pos
                rotated_pos = mount.rotate_point(wp.pos)
                rp = mount.resolved_pos
                body_pos = (
                    rotated_pos[0] + rp[0],
                    rotated_pos[1] + rp[1],
                    rotated_pos[2] + rp[2],
                )

                # Direction is a vector (no translation), only rotation
                body_dir = mount.rotate_point(
                    cspec.wire_exit_direction
                )  # plint: disable=no-rotate-point-for-axes

                # Quaternion aligning cylinder +Z to the wire exit direction
                stub_quat = rotation_between(
                    (0.0, 0.0, 1.0), tuple(body_dir)
                )  # plint: disable=no-rotation-between-in-emitters

                # Position the stub center halfway along the 25mm length
                half_len = 0.0125  # 25mm / 2
                stub_center = (
                    body_pos[0] + body_dir[0] * half_len,
                    body_pos[1] + body_dir[1] * half_len,
                    body_pos[2] + body_dir[2] * half_len,
                )

                bus_color = _BUS_TYPE_COLORS.get(
                    str(wp.bus_type), [0.53, 0.53, 0.53, 1.0]
                )

                manifest["parts"].append(
                    {
                        "id": f"wire_stub_{body.name}_{mount.label}_{wp.label}",
                        "name": wp.label,
                        "kind": "fabricated",
                        "category": "wire",
                        "wire_kind": "stub",
                        "parent_body": body.name,
                        "mesh": "wire_stub.stl",
                        "pos": _round_vec(stub_center),
                        "quat": _round_vec(stub_quat),
                        "bus_type": str(wp.bus_type),
                        "connector_type": wp.connector_type,
                        "color": bus_color,
                    }
                )

                # Connector housing at wire port position
                conn_quat = rotation_between(
                    (0.0, 0.0, 1.0), cspec.mating_direction
                )  # plint: disable=no-rotation-between-in-emitters
                # Apply mount rotation to connector orientation
                body_mate_dir = mount.rotate_point(
                    cspec.mating_direction
                )  # plint: disable=no-rotate-point-for-axes
                conn_quat = rotation_between(
                    (0.0, 0.0, 1.0), tuple(body_mate_dir)
                )  # plint: disable=no-rotation-between-in-emitters

                manifest["parts"].append(
                    {
                        "id": f"wire_conn_{body.name}_{mount.label}_{wp.label}",
                        "name": f"{wp.label} connector",
                        "kind": "fabricated",
                        "category": "wire",
                        "wire_kind": "connector",
                        "parent_body": body.name,
                        "mesh": f"connector_{wp.connector_type}.stl",
                        "pos": _round_vec(body_pos),
                        "quat": _round_vec(conn_quat),
                        "bus_type": str(wp.bus_type),
                        "connector_type": wp.connector_type,
                        "color": bus_color,
                    }
                )

        # --- Servo wire ports (on joints, not mounts) ---
        for joint in body.joints:
            servo = joint.servo
            for wp in servo.wire_ports:
                if not wp.connector_type:
                    continue

                try:
                    cspec = connector_spec(wp.connector_type)
                except KeyError:
                    log.debug(
                        "Unknown connector_type %r on servo %s/%s, skipping",
                        wp.connector_type,
                        servo.name,
                        wp.label,
                    )
                    continue

                # Transform from servo-local to body-local frame
                rotated_pos = rotate_vec(joint.solved_servo_quat, wp.pos)
                sc = joint.solved_servo_center
                body_pos = (
                    rotated_pos[0] + sc[0],
                    rotated_pos[1] + sc[1],
                    rotated_pos[2] + sc[2],
                )

                body_dir = rotate_vec(
                    joint.solved_servo_quat, cspec.wire_exit_direction
                )
                stub_quat = rotation_between(
                    (0.0, 0.0, 1.0), tuple(body_dir)
                )  # plint: disable=no-rotation-between-in-emitters

                half_len = 0.0125
                stub_center = (
                    body_pos[0] + body_dir[0] * half_len,
                    body_pos[1] + body_dir[1] * half_len,
                    body_pos[2] + body_dir[2] * half_len,
                )

                bus_color = _BUS_TYPE_COLORS.get(
                    str(wp.bus_type), [0.53, 0.53, 0.53, 1.0]
                )

                manifest["parts"].append(
                    {
                        "id": f"wire_stub_{body.name}_{joint.name}_{wp.label}",
                        "name": wp.label,
                        "kind": "fabricated",
                        "category": "wire",
                        "wire_kind": "stub",
                        "parent_body": body.name,
                        "mesh": "wire_stub.stl",
                        "pos": _round_vec(stub_center),
                        "quat": _round_vec(stub_quat),
                        "bus_type": str(wp.bus_type),
                        "connector_type": wp.connector_type,
                        "color": bus_color,
                    }
                )

                # Connector housing at wire port position
                body_mate_dir = rotate_vec(
                    joint.solved_servo_quat, cspec.mating_direction
                )
                conn_quat = rotation_between(
                    (0.0, 0.0, 1.0), tuple(body_mate_dir)
                )  # plint: disable=no-rotation-between-in-emitters

                manifest["parts"].append(
                    {
                        "id": f"wire_conn_{body.name}_{joint.name}_{wp.label}",
                        "name": f"{wp.label} connector",
                        "kind": "fabricated",
                        "category": "wire",
                        "wire_kind": "connector",
                        "parent_body": body.name,
                        "mesh": f"connector_{wp.connector_type}.stl",
                        "pos": _round_vec(body_pos),
                        "quat": _round_vec(conn_quat),
                        "bus_type": str(wp.bus_type),
                        "connector_type": wp.connector_type,
                        "color": bus_color,
                    }
                )


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
