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


def _round3(v):
    """Round a tuple/list of floats to 6 decimal places (µm precision)."""
    return [round(x, 6) for x in v]


def emit_viewer_manifest(bot: Bot, output_dir: Path) -> None:
    """Generate viewer_manifest.json in the bot's output directory."""
    manifest = {
        "bot_name": bot.name,
        "bodies": [],
        "joints": [],
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
            "parent": parent_name,
            "shape": str(body.shape),
            "dimensions": _round3(body.dimensions),
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
                "dimensions": _round3(comp.dimensions),
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
