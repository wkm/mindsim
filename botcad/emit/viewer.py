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

from botcad.component import BatterySpec, CameraSpec, ServoSpec


def _component_specs(comp) -> dict:
    """Extract type-specific specs from a Component."""
    specs: dict = {}
    if isinstance(comp, CameraSpec):
        specs["component_type"] = "camera"
        specs["fov_deg"] = comp.fov_deg
        specs["resolution"] = list(comp.resolution)
    elif isinstance(comp, BatterySpec):
        specs["component_type"] = "battery"
        specs["chemistry"] = comp.chemistry
        specs["voltage"] = comp.voltage
        specs["cells_s"] = comp.cells_s
    elif isinstance(comp, ServoSpec):
        specs["component_type"] = "servo"
        specs["stall_torque_nm"] = round(comp.stall_torque, 4)
        specs["no_load_speed_rad_s"] = round(comp.no_load_speed, 3)
        specs["voltage"] = comp.voltage
        specs["gear_ratio"] = comp.gear_ratio
    elif comp.is_wheel:
        specs["component_type"] = "wheel"
    else:
        specs["component_type"] = "component"
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


def emit_viewer_manifest(bot: Bot, output_dir: Path) -> None:
    """Generate viewer_manifest.json in the bot's output directory."""
    from botcad.fasteners import fastener_key, fastener_stl_stem

    manifest = {
        "bot_name": bot.name,
        "assemblies": _build_assembly_tree(bot),
        "bodies": [],
        "joints": [],
        "parts": [],
        "assembly_steps": [],
        "ik_chains": [],
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

    # 1. Servos and horns from all joints
    for body in bot.all_bodies:
        for joint in body.joints:
            servo = joint.servo
            # Servo part
            manifest["parts"].append(
                {
                    "id": f"servo_{joint.name}",
                    "name": servo.name,
                    "kind": "purchased",
                    "category": "servo",
                    "parent_body": body.name,
                    "joint": joint.name,
                    "mesh": f"servo_{servo.name}.stl",
                    "pos": _round_vec(joint.solved_servo_center),
                    "quat": _round_vec(joint.solved_servo_quat),
                    "mass": round(servo.mass, 4),
                    "shapescript_component": servo.name,
                }
            )

            # Horn disc part (all joints — wheels attach via horn too)
            from botcad.bracket import horn_disc_params

            if horn_disc_params(servo) is not None:
                manifest["parts"].append(
                    {
                        "id": f"horn_{joint.name}",
                        "name": "Horn disc",
                        "kind": "purchased",
                        "category": "horn",
                        "parent_body": body.name,
                        "joint": joint.name,
                        "mesh": f"horn_{joint.name}.stl",
                        "shapescript_component": f"horn:{servo.name}",
                    }
                )

    # 2. Mounted components (battery, camera, Pi, etc.)
    for body in bot.all_bodies:
        for mount in body.mounts:
            comp = mount.component
            part_entry = {
                "id": f"comp_{body.name}_{mount.label}",
                "name": comp.name,
                "kind": "purchased",
                "category": _component_specs(comp).get("component_type", "component"),
                "parent_body": body.name,
                "mount_label": mount.label,
                "mesh": f"comp_{body.name}_{mount.label}.stl",
                "pos": _round_vec(mount.resolved_pos),
                "mass": round(comp.mass, 4),
                "shapescript_component": comp.name,
            }
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
