# SO-101 Arm: Research & Implementation Status

## What is the SO-101?

The SO-101 (successor to SO-100) is an open-source, 3D-printable 6-DOF robot arm
from TheRobotStudio / Hugging Face's LeRobot ecosystem. 6x Feetech STS3215 servos
connected by 3D-printed structural brackets, with a parallel-jaw gripper end effector.
Cost: ~$130-220 for non-printed parts.

## Kinematic Structure

```
Base plate (fixed to table)
└── shoulder_pan  (Z-axis, ±110°)
    └── shoulder_lift (Y-axis, ±100°)
        └── elbow_flex (Y-axis, ±97°)
            └── wrist_flex (Y-axis, ±95°)
                └── wrist_roll (Z-axis, -157°/+163°)
                    └── gripper (Y-axis, -10° to +100°)
                        └── jaw (moving finger)
```

## Validation Against Reference

All kinematic parameters extracted from the official URDF (`so101_new_calib.urdf`)
and MuJoCo XML (`so101_new_calib.xml`) from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101).

| Metric | Ours | Reference | Error |
|--------|------|-----------|-------|
| Arm reach (rest) | 448.4mm | 448.6mm | 0.2mm |
| Body Z positions | Match | Cumulative URDF | <0.2mm |
| Joint ranges | Match | URDF specs | <0.3° |
| Stability (fixed base) | Zero KE | — | Perfect |
| Total mass | 436g | 632g | 69% (see below) |

### Mass Gap

Our structural mass uses thin-wall PLA estimate (1mm wall, 1200 kg/m³).
The reference includes actual 3D-printed bracket mass (15-20% infill, thicker walls).
This is a packing solver calibration issue, not a kinematic error.

## Implementation Status

### Done
- **Fixed-base support**: `base_type="fixed"` on Bot, no freejoint, correct placement
- **Waveshare controller**: Component in catalog, mounted on base, in BOM
- **Tabletop scene**: `worlds/tabletop.xml`, 200Hz timestep, auto-selected for fixed bots
- **Full kinematic chain**: 6 joints, 6 actuators, all ranges from URDF
- **Assembly guide**: Correct wiring (USB→Waveshare→servo bus), full tree assembly steps
- **BOM**: Correct parts list and power budget (no phantom Pi)
- **Gripper**: Modeled as revolute joint + jaw body (matches reference URDF approach)

### Remaining Work (future branches)
- **Inter-servo bracket geometry** (`bracket.py`): STL output is simple shapes, not
  accurate bracket CAD. Sim kinematics unaffected.
- **Arm manipulation training rewards**: Reach/grasp reward functions in sim_env
- **Packing solver calibration**: Increase structural mass estimate to match real brackets

## References

- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- [Official URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf)
- [Official MuJoCo XML](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.xml)
