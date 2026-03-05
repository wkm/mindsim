# SO-101 Arm: Research & Gap Analysis

## What is the SO-101?

The SO-101 (successor to SO-100) is an open-source, 3D-printable 6-DOF robot arm from TheRobotStudio / Hugging Face's LeRobot ecosystem. It uses 6x Feetech STS3215 servos connected by 3D-printed structural brackets, with a parallel-jaw gripper as the end effector. Cost: ~$130-220 for non-printed parts.

**Goal:** Recreate this arm from first principles using botcad — define the kinematic skeleton in the DSL, generate MuJoCo sim XML, printable STLs, and assembly instructions automatically.

## SO-101 Kinematic Structure

```
Base plate (fixed to table)
└── Joint 1: Shoulder Pan (Z-axis, STS3215 1/345)
    └── Joint 2: Shoulder Lift (X-axis, STS3215 1/345)
        └── Joint 3: Elbow Flex (X-axis, STS3215 1/345)
            └── Joint 4: Wrist Flex (X-axis, STS3215 1/147)
                └── Joint 5: Wrist Roll (Z-axis, STS3215 1/147)
                    └── Joint 6: Gripper (X-axis, STS3215 1/147)
                        ├── Left finger
                        └── Right finger
```

All joints revolute. Servos connect via 3D-printed bracket links.

## What BotCad Already Has

- STS3215 servo fully specified (dimensions, mounting ears, shaft offset, wire ports)
- Parametric bracket generation
- MuJoCo XML emission (position actuators, contact excludes, sensors)
- CAD emission (STEP + per-body STLs)
- BOM and assembly guide generation
- The full pipeline works end-to-end (wheeler_arm proves this)

## Gap Analysis: Extensions Needed

### 1. Fixed-Base Support
**Status:** Missing. `emit/mujoco.py` always adds `<freejoint>` to root.
**Fix:** Add `base_type: Literal["free", "fixed"]` to `Bot`. Skip freejoint when fixed.
**Scope:** Small — ~10 lines across `skeleton.py` + `emit/mujoco.py`.

### 2. Inter-Servo Bracket Links
**Status:** Missing. Current brackets are enclosure pockets. SO-101 brackets connect one servo's horn to the next servo's ears.
**Fix:** New `inter_servo_bracket()` in `bracket.py` generating L/U-shaped connectors.
**Scope:** Medium — new bracket geometry function + CAD emitter updates.

### 3. Gripper Mechanism
**Status:** Missing. No gripper primitive.
**Fix:** Model as revolute joint + finger body geometry. In MuJoCo: hinge joint with finger geoms and contact.
**Scope:** Medium — new body type or joint annotation, MuJoCo emitter updates.

### 4. Controller Board Component
**Status:** Missing Waveshare serial bus controller.
**Fix:** Add `WaveshareSerialBus()` component — dimensions, mass, ports.
**Scope:** Small — catalog entry similar to existing RPi component.

### 5. Tabletop Scene
**Status:** Missing. Current arena is a floor with curbs for wheeled bots.
**Fix:** Add `worlds/tabletop.xml` with table surface + target objects.
**Scope:** Small — new XML file, scene type selection in emitter.

### 6. Arm Training Rewards
**Status:** Missing. Current rewards likely for locomotion.
**Fix:** Add reach/grasp reward functions. Future work, not blocking design.
**Scope:** Large — separate effort.

## Implementation Phases

### Phase 1: Express the Design
1. `base_type` on Bot (fixed base)
2. Waveshare controller component
3. First-pass `design.py` using existing primitives

### Phase 2: Bracket Geometry
4. Inter-servo bracket generation
5. CAD emitter updates for serial arm chains

### Phase 3: Gripper
6. Gripper mechanism (joint + finger geometry)
7. MuJoCo emitter for gripper

### Phase 4: Training
8. Tabletop scene
9. Arm manipulation rewards
10. Full pipeline test

## References

- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- [Feetech STS3215 specs](https://shop.wowrobo.com/products/feetech-sts3215-c001-servo)
