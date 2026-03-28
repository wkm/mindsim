"""Simplified structural analysis for robot joints.

Calculates the safety factor of each joint based on servo stall torque
and bracket geometry. Identifies the "weakest link" in the robot's
fabricated structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botcad.skeleton import Body, Bot, Joint


@dataclass(frozen=True)
class StressResult:
    joint_name: str
    body_name: str
    side: str  # "parent" or "child"
    max_torque_nm: float
    stall_torque_nm: float
    safety_factor: float
    failure_mode: str


def analyze_joint_stresses(bot: Bot) -> list[StressResult]:
    """Analyze the safety factor of all fabricated joints in the bot."""
    from botcad.skeleton import BodyKind

    results = []

    # Ensure bot is solved so joints are populated
    if not bot.all_joints:
        bot._collect_tree()

    for joint in bot.all_joints:
        # --- Check Parent Side (Bracket) ---
        parent = None
        for body in bot.all_bodies:
            if joint in body.joints:
                parent = body
                break

        if parent and parent.kind == BodyKind.FABRICATED:
            results.append(_analyze_side(joint, parent, "parent"))

        # --- Check Child Side (Coupler or Attachment) ---
        child = joint.child
        if child and child.kind == BodyKind.FABRICATED:
            results.append(_analyze_side(joint, child, "child"))

    return sorted([r for r in results if r is not None], key=lambda r: r.safety_factor)


def _analyze_side(joint: Joint, body: Body, side: str) -> StressResult | None:
    """Analyze stress on one side of a joint."""
    from botcad.skeleton import BracketStyle

    torque = joint.servo.stall_torque
    if torque <= 0:
        return None

    mat = body.material
    yield_strength = mat.effective_yield_strength

    if mat and mat.process:
        wall_thick = mat.process.wall_layers * mat.process.nozzle_width
    else:
        wall_thick = 0.0008

    servo_l = joint.servo.effective_body_dims[0]
    servo_w = joint.servo.effective_body_dims[1]

    wall = 0.003
    tol = 0.0003

    if side == "parent":
        # Bracket neck stress
        bracket_l = servo_l + 2 * (tol + wall)
        bracket_w = servo_w + 2 * (tol + wall)
        perimeter = 2 * (bracket_l + bracket_w)
        area = perimeter * wall_thick
        radius = max(bracket_l, bracket_w) / 2
        failure_mode = "Shell-Bracket Interface Shear"
    else:
        # Child side: Coupler or direct attachment
        if joint.bracket_style == BracketStyle.COUPLER:
            # Weakest point is where the coupler arms meet the child body shell
            # Approximate coupler arm width as horn diameter
            from botcad.bracket import horn_disc_params

            hp = horn_disc_params(joint.servo)
            horn_d = (hp.radius * 2) if hp else 0.020

            # Neck area where coupler meets body
            area = horn_d * wall_thick * 2  # two arms
            radius = horn_d / 2
            failure_mode = "Coupler-Body Interface Shear"
        else:
            # POCKET style: child is attached to the horn via screws.
            # Stress is on the screw bosses in the child body.
            area = 0.003 * wall_thick * 4  # 4x M2 bosses
            radius = 0.007  # typical bolt circle
            failure_mode = "Screw Boss Yield"

    force = torque / radius
    stress = force / area
    stress *= 3.0  # concentration factor

    sf = yield_strength / stress

    return StressResult(
        joint_name=joint.name,
        body_name=body.name,
        side=side,
        max_torque_nm=torque * sf,
        stall_torque_nm=torque,
        safety_factor=sf,
        failure_mode=failure_mode,
    )


def print_fea_report(bot: Bot):
    """Print a human-readable FEA report for the bot."""
    results = analyze_joint_stresses(bot)

    print(f"\n--- Structural Analysis Report: {bot.name} ---")
    print(
        f"{'Joint':<20} {'Body':<15} {'Stall T':<10} {'Safety Factor':<15} {'Status'}"
    )
    print("-" * 75)

    for r in results:
        status = (
            "OK"
            if r.safety_factor > 1.5
            else "WARNING"
            if r.safety_factor > 1.0
            else "DANGER"
        )
        print(
            f"{r.joint_name:<20} {r.body_name:<15} {r.stall_torque_nm:<10.2f} {r.safety_factor:<15.2f} {status}"
        )

    if results:
        weakest = results[0]
        print("-" * 75)
        print(
            f"Weakest point: Joint '{weakest.joint_name}' on Body '{weakest.body_name}'"
        )
        print(f"Safety Factor: {weakest.safety_factor:.2f}")
        if weakest.safety_factor < 1.0:
            print(
                f"CRITICAL: The {weakest.body_name} body might crack under full stall torque!"
            )
        else:
            print(
                f"The design should withstand full stall torque with a safety factor of {weakest.safety_factor:.2f}."
            )
    else:
        print("No joints found for analysis.")
