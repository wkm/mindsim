#!/usr/bin/env python3
"""SO-101 — 6-DOF open-source robot arm (LeRobot / TheRobotStudio).

Parametric recreation of the SO-ARM100/SO-101 from first principles.
6x Feetech STS3215 servos connected by 3D-printed structural brackets,
with a parallel-jaw gripper end effector. Fixed to a table.

Kinematic chain (from official URDF so101_new_calib.urdf):
    Base plate (fixed)
    └── shoulder_pan  (Z-axis, ±110°)  — rotates entire arm
        └── shoulder_lift (Y-axis, ±100°)  — tilts arm up/down
            └── elbow_flex (Y-axis, ±97°)  — bends forearm
                └── wrist_flex (Y-axis, ±95°)  — tilts hand
                    └── wrist_roll (Z-axis, ±157/163°)  — rotates hand
                        └── gripper (Y-axis, -10° to +100°) — opens/closes jaw

Reference dimensions extracted from the official URDF + MuJoCo files:
    github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101

Run:
    python bots/so101_arm/design.py
    uv run mjpython main.py view --bot so101_arm
"""

from pathlib import Path

from botcad.components import STS3215, WaveshareSerialBus
from botcad.skeleton import Bot


def build() -> Bot:
    """Define the SO-101 robot arm.

    Joint positions derived from the official URDF (so101_new_calib.urdf).
    The URDF uses rotated frames at each joint; here we convert to botcad's
    convention where joint pos is in the parent body's local frame and axis
    is expressed directly.

    Link lengths from URDF (meters):
        base height:      0.0624  (base_link → shoulder_link Z)
        shoulder offset:  0.0542  (shoulder → upper_arm, mostly Z after frame rotation)
        upper arm:        0.1126  (upper_arm → lower_arm, along arm axis)
        forearm:          0.1349  (lower_arm → wrist, along arm axis)
        wrist-to-gripper: 0.0611  (wrist → gripper, along arm axis)
        gripper jaw:      0.0234  (gripper → moving jaw)

    Masses from official MuJoCo XML (kg):
        base: 0.147, shoulder: 0.100, upper_arm: 0.103
        lower_arm: 0.104, wrist: 0.079, gripper: 0.087, jaw: 0.012
        Total: 0.632 kg
    """
    bot = Bot("so101_arm", base_type="fixed")

    # ── Base plate ──────────────────────────────────────────────────
    # Houses the Waveshare controller board. Bolted to table.
    # URDF base mass: 0.147 kg (includes base bracket + first servo)
    base = bot.body("base", shape="box", dimensions=(0.075, 0.075, 0.020))
    base.mount(WaveshareSerialBus(), position="center", label="controller")

    # ── Joint 1: Shoulder Pan ───────────────────────────────────────
    # Rotates around Z (vertical). Servo sits on top of base plate.
    # URDF origin: xyz=(0.0388, 0, 0.0624) — 62.4mm above base origin.
    # This includes base plate (20mm) + servo height (35mm) + bracket (~7mm).
    shoulder_pan = base.joint(
        "shoulder_pan",
        servo=STS3215(),
        axis="z",
        pos=(0.0, 0.0, 0.0624),  # match URDF: 62.4mm above base origin
        range=(-1.92, 1.92),  # ±110° from URDF
    )
    # Shoulder body: the bracket connecting pan servo to lift servo
    # URDF shoulder mass: 0.100 kg
    shoulder = shoulder_pan.body(
        "shoulder", shape="box", dimensions=(0.050, 0.040, 0.054)
    )

    # ── Joint 2: Shoulder Lift ──────────────────────────────────────
    # Tilts the arm up/down. Axis = Y in world frame (X in botcad).
    # URDF origin: xyz=(-0.0304, -0.0183, -0.0542) with rpy=(-pi/2, -pi/2, 0)
    # Effective offset after frame rotation: ~54.2mm along parent Z
    shoulder_lift = shoulder.joint(
        "shoulder_lift",
        servo=STS3215(),
        axis="y",
        pos=(0.0, 0.0, 0.054),  # top of shoulder body
        range=(-1.75, 1.75),  # ±100° from URDF
    )
    # Upper arm link: 112.6mm long tube
    # URDF upper_arm mass: 0.103 kg
    upper_arm = shoulder_lift.body(
        "upper_arm", shape="tube", length=0.1126, outer_r=0.018
    )

    # ── Joint 3: Elbow Flex ─────────────────────────────────────────
    # Bends the forearm. Same axis as shoulder lift.
    # URDF origin: xyz=(-0.1126, -0.028, 0) — 112.6mm along arm
    elbow_flex = upper_arm.joint(
        "elbow_flex",
        servo=STS3215(),
        axis="y",
        pos=(0.0, 0.0, 0.1126),  # end of upper arm tube
        range=(-1.69, 1.69),  # ±97° from URDF
    )
    # Forearm link: 134.9mm long tube
    # URDF lower_arm mass: 0.104 kg
    forearm = elbow_flex.body("forearm", shape="tube", length=0.1349, outer_r=0.016)

    # ── Joint 4: Wrist Flex ─────────────────────────────────────────
    # Tilts the hand up/down.
    # URDF origin: xyz=(-0.1349, 0.0052, 0) — 134.9mm along arm
    wrist_flex = forearm.joint(
        "wrist_flex",
        servo=STS3215(),
        axis="y",
        pos=(0.0, 0.0, 0.1349),  # end of forearm tube
        range=(-1.66, 1.66),  # ±95° from URDF
    )
    # Wrist body: compact bracket
    # URDF wrist mass: 0.079 kg
    wrist = wrist_flex.body("wrist", shape="box", dimensions=(0.040, 0.035, 0.035))

    # ── Joint 5: Wrist Roll ─────────────────────────────────────────
    # Rotates the hand/gripper around the arm axis.
    # URDF origin: xyz=(0, -0.0611, 0.0181) with rpy=(pi/2, 0.049, pi)
    # Effective offset: ~61.1mm along arm axis from wrist origin
    wrist_roll = wrist.joint(
        "wrist_roll",
        servo=STS3215(),
        axis="z",
        pos=(0.0, 0.0, 0.0611),  # match URDF: 61.1mm along arm axis
        range=(-2.74, 2.84),  # asymmetric range from URDF
    )
    # Gripper body: houses the gripper mechanism
    # URDF gripper mass: 0.087 kg
    hand = wrist_roll.body("hand", shape="box", dimensions=(0.045, 0.040, 0.030))

    # ── Joint 6: Gripper ────────────────────────────────────────────
    # Opens/closes the parallel jaw.
    # URDF origin: xyz=(0.0202, 0.0188, -0.0234) with rpy=(pi/2, 0, 0)
    # Effective offset ~23.4mm
    # URDF limits: -0.175 to 1.745 rad (-10° to 100°)
    gripper = hand.joint(
        "gripper",
        servo=STS3215(),
        axis="y",
        pos=(0.0, 0.0, 0.0234),  # match URDF: 23.4mm
        range=(-0.175, 1.745),  # from URDF
    )
    # Moving jaw: small finger piece
    # URDF moving_jaw mass: 0.012 kg
    gripper.body("jaw", shape="box", dimensions=(0.060, 0.020, 0.010))

    return bot


def main() -> None:
    bot = build()
    bot.solve()

    output_dir = Path(__file__).parent
    bot.emit(str(output_dir))

    print(f"\nGenerated so101_arm bot in {output_dir}")
    print("  bot.xml       — MuJoCo robot definition")
    print("  scene.xml     — Scene wrapper (bot + tabletop)")
    print("  meshes/       — Per-body STL meshes")
    print("  bom.md        — Bill of materials")
    print("  assembly_guide.md — Assembly instructions")
    print("\nTo view in MuJoCo:")
    print("  uv run mjpython main.py view --bot so101_arm")
    print("\nValidation:")
    print(f"  Total joints: {len(bot.all_joints)}")
    print(f"  Total bodies: {len(bot.all_bodies)}")
    total_mass = sum(b.solved_mass for b in bot.all_bodies)
    print(f"  Total mass:   {total_mass:.3f} kg (reference: 0.632 kg)")


if __name__ == "__main__":
    main()
