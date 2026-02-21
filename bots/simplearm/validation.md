# Bot Validation: simplearm
Generated: 2026-02-21

## Model Summary

| Property | Value |
|----------|-------|
| Bodies | 16 |
| Joints | 8 |
| Actuators | 6 |
| Sensors | 14 |
| Total mass | 15.60 kg |
| Timestep | 0.002 s |

## Rest State (all ctrl = 0)

### Body Positions

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.100 | 0.000 |
| shoulder | 0.000 | -0.100 | 0.620 |
| upper_arm | 0.000 | -0.100 | 0.620 |
| forearm | 0.000 | -0.100 | 0.370 |
| wrist | 0.000 | -0.100 | 0.170 |
| wrist_roll_link | 0.000 | -0.100 | 0.170 |
| gripper_base | 0.000 | -0.100 | 0.145 |
| finger_left | 0.000 | -0.105 | 0.137 |
| finger_right | 0.000 | -0.095 | 0.137 |
| cup | 0.000 | 0.250 | 0.439 |
| target | 0.000 | 0.250 | 0.650 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

### Site Positions

| Site | X | Y | Z |
|------|------|------|------|
| gripper_site | 0.000 | -0.100 | 0.095 |
| finger_left_touch | 0.000 | -0.100 | 0.102 |
| finger_right_touch | 0.000 | -0.100 | 0.102 |

### Contacts

- cup_geom <-> table_surface (depth=-0.0006)

## Actuator Sweep

### shoulder_yaw_motor
- **Joint**: shoulder_yaw (hinge, axis=[0, 0, 1])
- **Range**: [-3.142, 3.142] | **kp**: 200

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -3.142 | -3.142 | 0.000 |
| MAX | +3.142 | +3.142 | 0.000 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.100 | 0.000 |
| shoulder | 0.000 | -0.100 | 0.620 |
| upper_arm | 0.000 | -0.100 | 0.620 |
| forearm | 0.000 | -0.100 | 0.370 |
| wrist | 0.000 | -0.100 | 0.170 |
| wrist_roll_link | 0.000 | -0.100 | 0.170 |
| gripper_base | 0.000 | -0.100 | 0.145 |
| finger_left | 0.000 | -0.095 | 0.137 |
| finger_right | -0.000 | -0.105 | 0.137 |
| cup | 0.000 | 0.250 | 0.439 |
| target | 0.000 | 0.250 | 0.650 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| gripper_site | 0.000 | -0.100 | 0.095 |
| finger_left_touch | 0.000 | -0.100 | 0.102 |
| finger_right_touch | 0.000 | -0.100 | 0.102 |

**Contacts at MAX:**
- cup_geom <-> table_surface (depth=-0.0006)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)

**Contacts at MIN:**
- cup_geom <-> table_surface (depth=-0.0006)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)

### shoulder_pitch_motor
- **Joint**: shoulder_pitch (hinge, axis=[1, 0, 0])
- **Range**: [-1.571, 2.618] | **kp**: 200

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.571 | -1.526 | 0.045 |
| MAX | +2.618 | +2.594 | 0.024 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.100 | 0.000 |
| shoulder | 0.000 | -0.100 | 0.620 |
| upper_arm | 0.000 | -0.100 | 0.620 |
| forearm | -0.000 | 0.030 | 0.833 |
| wrist | -0.000 | 0.136 | 1.003 |
| wrist_roll_link | -0.000 | 0.136 | 1.003 |
| gripper_base | -0.000 | 0.150 | 1.024 |
| finger_left | -0.000 | 0.159 | 1.028 |
| finger_right | -0.000 | 0.150 | 1.033 |
| cup | 2.264 | 5.147 | 0.024 |
| target | 0.000 | 0.250 | 0.650 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| gripper_site | -0.000 | 0.176 | 1.066 |
| finger_left_touch | -0.000 | 0.173 | 1.060 |
| finger_right_touch | -0.000 | 0.173 | 1.060 |

**Contacts at MAX:**
- floor <-> cup_geom (depth=-0.0005)
- floor <-> cup_geom (depth=-0.0005)

**Contacts at MIN:**
- cup_geom <-> table_surface (depth=-0.0006)

### elbow_motor
- **Joint**: elbow (hinge, axis=[1, 0, 0])
- **Range**: [-2.356, 0.000] | **kp**: 100

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -2.356 | -2.338 | 0.018 |
| MAX | +0.000 | -0.000 | 0.000 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.100 | 0.000 |
| shoulder | 0.000 | -0.100 | 0.620 |
| upper_arm | 0.000 | -0.100 | 0.620 |
| forearm | 0.000 | -0.100 | 0.370 |
| wrist | 0.000 | -0.100 | 0.170 |
| wrist_roll_link | 0.000 | -0.100 | 0.170 |
| gripper_base | 0.000 | -0.100 | 0.145 |
| finger_left | 0.000 | -0.105 | 0.137 |
| finger_right | 0.000 | -0.095 | 0.137 |
| cup | 0.000 | 0.250 | 0.439 |
| target | 0.000 | 0.250 | 0.650 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| gripper_site | 0.000 | -0.100 | 0.095 |
| finger_left_touch | 0.000 | -0.100 | 0.102 |
| finger_right_touch | 0.000 | -0.100 | 0.102 |

**Contacts at MAX:**
- cup_geom <-> table_surface (depth=-0.0006)

**Contacts at MIN:**
- cup_geom <-> table_surface (depth=-0.0006)

### wrist_pitch_motor
- **Joint**: wrist_pitch (hinge, axis=[1, 0, 0])
- **Range**: [-1.571, 1.571] | **kp**: 50

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.571 | -1.569 | 0.002 |
| MAX | +1.571 | +1.569 | 0.002 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.100 | 0.000 |
| shoulder | 0.000 | -0.100 | 0.620 |
| upper_arm | 0.000 | -0.100 | 0.620 |
| forearm | 0.000 | -0.100 | 0.370 |
| wrist | 0.000 | -0.100 | 0.170 |
| wrist_roll_link | 0.000 | -0.100 | 0.170 |
| gripper_base | 0.000 | -0.075 | 0.170 |
| finger_left | 0.000 | -0.068 | 0.164 |
| finger_right | 0.000 | -0.067 | 0.175 |
| cup | 0.000 | 0.250 | 0.439 |
| target | 0.000 | 0.250 | 0.650 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| gripper_site | 0.000 | -0.025 | 0.170 |
| finger_left_touch | 0.000 | -0.032 | 0.169 |
| finger_right_touch | 0.000 | -0.032 | 0.170 |

**Contacts at MAX:**
- cup_geom <-> table_surface (depth=-0.0006)

**Contacts at MIN:**
- cup_geom <-> table_surface (depth=-0.0006)

### wrist_roll_motor
- **Joint**: wrist_roll (hinge, axis=[0, 0, 1])
- **Range**: [-3.142, 3.142] | **kp**: 20

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -3.142 | -3.142 | 0.000 |
| MAX | +3.142 | +3.142 | 0.000 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.100 | 0.000 |
| shoulder | 0.000 | -0.100 | 0.620 |
| upper_arm | 0.000 | -0.100 | 0.620 |
| forearm | 0.000 | -0.100 | 0.370 |
| wrist | 0.000 | -0.100 | 0.170 |
| wrist_roll_link | 0.000 | -0.100 | 0.170 |
| gripper_base | 0.000 | -0.100 | 0.145 |
| finger_left | 0.000 | -0.095 | 0.137 |
| finger_right | -0.000 | -0.105 | 0.137 |
| cup | 0.000 | 0.250 | 0.439 |
| target | 0.000 | 0.250 | 0.650 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| gripper_site | 0.000 | -0.100 | 0.095 |
| finger_left_touch | 0.000 | -0.100 | 0.102 |
| finger_right_touch | -0.000 | -0.100 | 0.102 |

**Contacts at MAX:**
- cup_geom <-> table_surface (depth=-0.0006)

**Contacts at MIN:**
- cup_geom <-> table_surface (depth=-0.0006)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)
- finger_left_geom <-> finger_right_geom (depth=-0.0000)

### gripper_motor
- **Joint**: finger_left (slide, axis=[0, -1, 0])
- **Range**: [0.000, 0.025] | **kp**: 40

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | +0.000 | -0.000 | 0.000 |
| MAX | +0.025 | +0.025 | 0.000 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.100 | 0.000 |
| shoulder | 0.000 | -0.100 | 0.620 |
| upper_arm | 0.000 | -0.100 | 0.620 |
| forearm | 0.000 | -0.100 | 0.370 |
| wrist | 0.000 | -0.100 | 0.170 |
| wrist_roll_link | 0.000 | -0.100 | 0.170 |
| gripper_base | 0.000 | -0.100 | 0.145 |
| finger_left | 0.000 | -0.130 | 0.137 |
| finger_right | 0.000 | -0.070 | 0.137 |
| cup | 0.000 | 0.250 | 0.439 |
| target | 0.000 | 0.250 | 0.650 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| gripper_site | 0.000 | -0.100 | 0.095 |
| finger_left_touch | 0.000 | -0.125 | 0.102 |
| finger_right_touch | 0.000 | -0.075 | 0.102 |

**Contacts at MAX:**
- cup_geom <-> table_surface (depth=-0.0006)

**Contacts at MIN:**
- cup_geom <-> table_surface (depth=-0.0006)

