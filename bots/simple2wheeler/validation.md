# Bot Validation: simple2wheeler
Generated: 2026-02-21

## Model Summary

| Property | Value |
|----------|-------|
| Bodies | 11 |
| Joints | 3 |
| Actuators | 2 |
| Sensors | 4 |
| Total mass | 14.95 kg |
| Timestep | 0.02 s |

## Rest State (all ctrl = 0)

### Body Positions

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.004 | -0.001 |
| body_1 | 0.000 | -0.004 | -0.001 |
| l_wheel_1 | -0.000 | -0.002 | -0.001 |
| r_wheel_1 | -0.000 | 0.004 | -0.001 |
| camera_1 | 0.000 | -0.004 | -0.001 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

### Contacts

- floor <-> geom_6 (depth=-0.0009)
- floor <-> geom_7 (depth=-0.0009)

## Actuator Sweep

### Revolute_1_motor
- **Joint**: Revolute_1 (hinge, axis=[-1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -363.635 | 362.635 |
| MAX | +1.000 | +360.780 | 359.780 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | -0.054 | 0.179 | -0.004 |
| body_1 | -0.054 | 0.179 | -0.004 |
| l_wheel_1 | -0.029 | 0.147 | 0.134 |
| r_wheel_1 | -0.063 | 0.193 | 0.145 |
| camera_1 | -0.054 | 0.179 | -0.004 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Contacts at MAX:**
- floor <-> geom_6 (depth=-0.0045)
- floor <-> geom_7 (depth=-0.0019)

**Contacts at MIN:**
- floor <-> geom_6 (depth=-0.0010)
- floor <-> geom_7 (depth=-0.0010)

### Revolute_2_motor
- **Joint**: Revolute_2 (hinge, axis=[1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -363.138 | 362.138 |
| MAX | +1.000 | +363.666 | 362.666 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.055 | 0.023 | -0.001 |
| body_1 | 0.055 | 0.023 | -0.001 |
| l_wheel_1 | 0.114 | -0.017 | 0.052 |
| r_wheel_1 | 0.103 | -0.010 | 0.029 |
| camera_1 | 0.055 | 0.023 | -0.001 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Contacts at MAX:**
- floor <-> geom_6 (depth=-0.0009)
- floor <-> geom_7 (depth=-0.0007)

**Contacts at MIN:**
- floor <-> geom_6 (depth=-0.0009)
- floor <-> geom_7 (depth=-0.0014)

