"""
Bot validation script — automated actuator sweep and geometry check.

For each bot, loads the scene and:
1. Reports model summary (bodies, joints, actuators, sensors, mass)
2. Checks rest state (all ctrl=0): body positions, contacts
3. Sweeps each actuator to its min and max ctrlrange (one at a time,
   others at 0), recording target vs actual joint angle, body positions,
   contacts, and key site positions
4. Writes results to bots/<name>/validation.md

Usage:
    uv run python validate_bot.py                          # All bots
    uv run python validate_bot.py --bot simplearm          # Single bot
    uv run python validate_bot.py --scene bots/X/scene.xml # By path
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import mujoco
import numpy as np


SETTLE_STEPS = 3000  # 6s at 0.002s timestep — plenty for position actuators


def _fmt_pos(pos: np.ndarray) -> str:
    """Format a position as '0.000 | 0.000 | 0.000'."""
    return f"{pos[0]:.3f} | {pos[1]:.3f} | {pos[2]:.3f}"


def _body_name(model: mujoco.MjModel, i: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    return name if name else f"body_{i}"


def _joint_name(model: mujoco.MjModel, i: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    return name if name else f"joint_{i}"


def _actuator_name(model: mujoco.MjModel, i: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    return name if name else f"actuator_{i}"


def _site_name(model: mujoco.MjModel, i: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    return name if name else f"site_{i}"


def _geom_name(model: mujoco.MjModel, i: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    return name if name else f"geom_{i}"


def _joint_type_str(jnt_type: int) -> str:
    return {0: "free", 1: "ball", 2: "slide", 3: "hinge"}.get(jnt_type, f"type_{jnt_type}")


def _format_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Return a list of contact description strings."""
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = _geom_name(model, c.geom1)
        g2 = _geom_name(model, c.geom2)
        depth = c.dist
        contacts.append(f"{g1} <-> {g2} (depth={depth:.4f})")
    return contacts


def _body_positions_table(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Generate markdown table rows for all body positions."""
    rows = []
    for i in range(model.nbody):
        name = _body_name(model, i)
        if name == "world":
            continue
        pos = data.xpos[i]
        rows.append(f"| {name} | {pos[0]:.3f} | {pos[1]:.3f} | {pos[2]:.3f} |")
    return rows


def _site_positions(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Generate site position lines (if any sites exist)."""
    if model.nsite == 0:
        return []
    rows = []
    for i in range(model.nsite):
        name = _site_name(model, i)
        pos = data.site_xpos[i]
        rows.append(f"| {name} | {pos[0]:.3f} | {pos[1]:.3f} | {pos[2]:.3f} |")
    return rows


def _model_summary(model: mujoco.MjModel) -> list[str]:
    """Generate the Model Summary section."""
    total_mass = sum(float(model.body_mass[i]) for i in range(model.nbody))
    lines = [
        "## Model Summary",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Bodies | {model.nbody} |",
        f"| Joints | {model.njnt} |",
        f"| Actuators | {model.nu} |",
        f"| Sensors | {model.nsensor} |",
        f"| Total mass | {total_mass:.2f} kg |",
        f"| Timestep | {model.opt.timestep} s |",
    ]
    return lines


def _rest_state(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Generate the Rest State section."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    # Step for a short settle to let gravity act
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)

    lines = [
        "## Rest State (all ctrl = 0)",
        "",
        "### Body Positions",
        "",
        "| Body | X | Y | Z |",
        "|------|------|------|------|",
    ]
    lines.extend(_body_positions_table(model, data))

    # Sites
    site_rows = _site_positions(model, data)
    if site_rows:
        lines.extend([
            "",
            "### Site Positions",
            "",
            "| Site | X | Y | Z |",
            "|------|------|------|------|",
        ])
        lines.extend(site_rows)

    # Contacts
    contacts = _format_contacts(model, data)
    lines.extend(["", "### Contacts", ""])
    if contacts:
        for c in contacts:
            lines.append(f"- {c}")
    else:
        lines.append("(none)")

    return lines


def _get_joint_info(model: mujoco.MjModel, actuator_idx: int) -> dict:
    """Get joint info for an actuator's transmission target."""
    # Get the joint that this actuator drives
    trntype = model.actuator_trntype[actuator_idx]
    trnid = model.actuator_trnid[actuator_idx][0]

    info = {"joint_idx": trnid, "joint_name": "?", "joint_type": "?", "axis": "?"}

    if trntype == 0:  # mjTRN_JOINT
        info["joint_name"] = _joint_name(model, trnid)
        jnt_type = model.jnt_type[trnid]
        info["joint_type"] = _joint_type_str(jnt_type)
        if jnt_type == 3:  # hinge
            info["axis"] = f"[{model.jnt_axis[trnid][0]:.0f}, {model.jnt_axis[trnid][1]:.0f}, {model.jnt_axis[trnid][2]:.0f}]"
        elif jnt_type == 2:  # slide
            info["axis"] = f"[{model.jnt_axis[trnid][0]:.0f}, {model.jnt_axis[trnid][1]:.0f}, {model.jnt_axis[trnid][2]:.0f}]"

    return info


def _get_joint_position(model: mujoco.MjModel, data: mujoco.MjData, joint_idx: int) -> float:
    """Get the current position of a joint."""
    adr = model.jnt_qposadr[joint_idx]
    jnt_type = model.jnt_type[joint_idx]
    if jnt_type == 0:  # free — skip
        return 0.0
    return float(data.qpos[adr])


def _actuator_sweep(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Generate the Actuator Sweep section."""
    lines = ["## Actuator Sweep", ""]

    for act_i in range(model.nu):
        act_name = _actuator_name(model, act_i)
        joint_info = _get_joint_info(model, act_i)
        ctrlrange = model.actuator_ctrlrange[act_i]
        ctrl_min, ctrl_max = float(ctrlrange[0]), float(ctrlrange[1])

        # Get actuator gain (kp for position actuators)
        kp = float(model.actuator_gainprm[act_i][0])

        lines.append(f"### {act_name}")
        lines.append(f"- **Joint**: {joint_info['joint_name']} ({joint_info['joint_type']}, axis={joint_info['axis']})")
        lines.append(f"- **Range**: [{ctrl_min:.3f}, {ctrl_max:.3f}] | **kp**: {kp:.0f}")
        lines.append("")

        # Sweep to MIN and MAX
        results = {}
        for label, target_val in [("MIN", ctrl_min), ("MAX", ctrl_max)]:
            mujoco.mj_resetData(model, data)
            data.ctrl[:] = 0
            data.ctrl[act_i] = target_val
            mujoco.mj_forward(model, data)

            for _ in range(SETTLE_STEPS):
                mujoco.mj_step(model, data)

            actual = _get_joint_position(model, data, joint_info["joint_idx"])
            error = abs(target_val - actual)

            results[label] = {
                "target": target_val,
                "actual": actual,
                "error": error,
                "contacts": _format_contacts(model, data),
                "body_positions": _body_positions_table(model, data),
                "site_positions": _site_positions(model, data),
            }

        # Tracking accuracy table
        lines.extend([
            "| | Target | Actual | Error |",
            "|-----|--------|--------|-------|",
        ])
        for label in ["MIN", "MAX"]:
            r = results[label]
            lines.append(
                f"| {label} | {r['target']:+.3f} | {r['actual']:+.3f} | {r['error']:.3f} |"
            )
        lines.append("")

        # Body positions and contacts at MAX (most informative extreme)
        r_max = results["MAX"]
        lines.append("**Body positions at MAX:**")
        lines.append("")
        lines.extend([
            "| Body | X | Y | Z |",
            "|------|------|------|------|",
        ])
        lines.extend(r_max["body_positions"])
        lines.append("")

        # Site positions at MAX
        if r_max["site_positions"]:
            lines.append("**Site positions at MAX:**")
            lines.append("")
            lines.extend([
                "| Site | X | Y | Z |",
                "|------|------|------|------|",
            ])
            lines.extend(r_max["site_positions"])
            lines.append("")

        # Contacts at MAX
        if r_max["contacts"]:
            lines.append("**Contacts at MAX:**")
            for c in r_max["contacts"]:
                lines.append(f"- {c}")
        else:
            lines.append("**Contacts at MAX:** (none)")
        lines.append("")

        # Also note contacts at MIN if any
        r_min = results["MIN"]
        if r_min["contacts"]:
            lines.append("**Contacts at MIN:**")
            for c in r_min["contacts"]:
                lines.append(f"- {c}")
            lines.append("")

    return lines


def validate_bot(scene_path: str) -> str:
    """Run validation on a bot, return markdown report string."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    bot_name = Path(scene_path).parent.name

    report = []
    report.append(f"# Bot Validation: {bot_name}")
    report.append(f"Generated: {date.today().isoformat()}")
    report.append("")

    # 1. Model summary
    report.extend(_model_summary(model))
    report.append("")

    # 2. Rest state
    report.extend(_rest_state(model, data))
    report.append("")

    # 3. Actuator sweep
    report.extend(_actuator_sweep(model, data))

    return "\n".join(report)


def _discover_bots() -> list[dict]:
    """Scan bots/*/scene.xml and return info about each bot."""
    bots_dir = Path("bots")
    results = []
    if bots_dir.is_dir():
        for scene in sorted(bots_dir.glob("*/scene.xml")):
            name = scene.parent.name
            results.append({"name": name, "scene_path": str(scene)})
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Bot validation — automated actuator sweep and geometry check"
    )
    parser.add_argument("--bot", type=str, default=None,
                        help="Bot name (e.g. simplearm)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Path to scene.xml (overrides --bot)")
    args = parser.parse_args()

    if args.scene:
        # Single scene by path
        scene_path = args.scene
        bot_name = Path(scene_path).parent.name
        print(f"Validating {bot_name} ({scene_path})...")
        report = validate_bot(scene_path)
        out_path = Path(scene_path).parent / "validation.md"
        out_path.write_text(report + "\n")
        print(f"  -> {out_path}")
    elif args.bot:
        # Single bot by name
        bots = _discover_bots()
        match = [b for b in bots if b["name"] == args.bot]
        if not match:
            available = ", ".join(b["name"] for b in bots)
            print(f"Error: Unknown bot '{args.bot}'. Available: {available}")
            raise SystemExit(1)
        bot = match[0]
        print(f"Validating {bot['name']}...")
        report = validate_bot(bot["scene_path"])
        out_path = Path(bot["scene_path"]).parent / "validation.md"
        out_path.write_text(report + "\n")
        print(f"  -> {out_path}")
    else:
        # All bots
        bots = _discover_bots()
        if not bots:
            print("No bots found in bots/*/scene.xml")
            raise SystemExit(1)
        for bot in bots:
            print(f"Validating {bot['name']}...")
            report = validate_bot(bot["scene_path"])
            out_path = Path(bot["scene_path"]).parent / "validation.md"
            out_path.write_text(report + "\n")
            print(f"  -> {out_path}")
        print(f"\nValidated {len(bots)} bots.")


if __name__ == "__main__":
    main()
