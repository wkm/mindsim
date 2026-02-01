# Fusion 360 to MuJoCo Exporter - Architecture

## Overview

This Fusion 360 add-in exports CAD assemblies with joints to MuJoCo-compatible MJCF (MuJoCo Modeling Format) XML files. It extracts the kinematic structure, joint definitions, inertial properties, and mesh geometry from Fusion 360 designs.

## References

- **Fusion 360 API**: https://help.autodesk.com/view/fusion360/ENU/?guid=GUID-A92A4B10-3781-4925-94C6-47DA85A4F65A
- **MuJoCo MJCF Reference**: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- **Inspiration**: https://github.com/runtimerobotics/fusion360-urdf-ros2

## Mapping: Fusion 360 to MuJoCo

### Joint Type Mapping

| Fusion 360 Joint  | MuJoCo Joint | Notes |
|-------------------|--------------|-------|
| Rigid             | (none)       | Body welded to parent (no joint element) |
| Revolute          | hinge        | 1-DOF rotation around axis |
| Slider            | slide        | 1-DOF translation along axis |
| Cylindrical       | hinge+slide  | 2-DOF: rotation + translation (composite) |
| Ball              | ball         | 3-DOF spherical joint |
| Planar            | slide+slide  | 2-DOF translation (composite) |

### Structural Mapping

| Fusion 360 Concept | MuJoCo Concept |
|--------------------|----------------|
| Component          | body           |
| Component bodies   | geom (mesh)    |
| Joint              | joint (inside child body) |
| Physical properties| inertial       |
| Root component     | worldbody      |

### Coordinate Conventions

- **Fusion 360**: Uses centimeters internally, Y-up or Z-up depending on design
- **MuJoCo**: Uses meters, Z-up by default
- **Conversion**: All positions scaled by 0.01 (cm to m), meshes exported in mm and scaled by 0.001

## MJCF Output Structure

```xml
<mujoco model="robot_name">
  <compiler angle="radian" meshdir="meshes"/>

  <option gravity="0 0 -9.81"/>

  <default>
    <joint damping="0.1"/>
    <geom friction="1 0.005 0.0001"/>
  </default>

  <asset>
    <mesh name="link1" file="meshes/link1.stl" scale="0.001 0.001 0.001"/>
    <mesh name="link2" file="meshes/link2.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <body name="base_link" pos="0 0 0">
      <inertial pos="x y z" mass="m" diaginertia="Ixx Iyy Izz"/>
      <geom type="mesh" mesh="base_link"/>

      <body name="link1" pos="x y z" quat="w x y z">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <inertial pos="x y z" mass="m" diaginertia="Ixx Iyy Izz"/>
        <geom type="mesh" mesh="link1"/>

        <!-- Nested child bodies -->
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="joint1" gear="1"/>
  </actuator>
</mujoco>
```

## Module Design

```
fusion360_mujoco_exporter/
├── Fusion360MuJoCoExporter.py     # Main add-in entry point
├── Fusion360MuJoCoExporter.manifest
├── core/
│   ├── __init__.py
│   ├── joint_extractor.py         # Extract joints from Fusion 360
│   ├── body_extractor.py          # Extract bodies/components with inertial props
│   ├── mesh_exporter.py           # Export STL meshes
│   ├── mjcf_generator.py          # Generate MJCF XML
│   └── transforms.py              # Coordinate transforms and utilities
└── ARCHITECTURE.md
```

### Module Responsibilities

#### `Fusion360MuJoCoExporter.py`
- Add-in lifecycle (run/stop)
- UI command creation (button in toolbar)
- Dialog for export options (output folder, model name)
- Orchestrates the export pipeline

#### `core/joint_extractor.py`
- Iterate over `root.joints`
- Map Fusion joint types to MuJoCo types
- Extract axis vectors, limits, parent/child occurrences
- Calculate joint origins in local coordinates

#### `core/body_extractor.py`
- Traverse `root.occurrences` to build kinematic tree
- Extract physical properties (mass, center of mass, inertia tensor)
- Transform inertia to body-local frame
- Identify base_link (root body)

#### `core/mesh_exporter.py`
- Use `design.exportManager` with `createSTLExportOptions()`
- Export each component's mesh as separate STL file
- Apply mesh refinement settings

#### `core/mjcf_generator.py`
- Build XML tree using `xml.etree.ElementTree`
- Generate `<asset>` section with mesh references
- Generate `<worldbody>` with nested body hierarchy
- Generate `<actuator>` section for motorized joints
- Write formatted XML to file

#### `core/transforms.py`
- Matrix operations for coordinate transforms
- Quaternion utilities
- Unit conversions (cm to m, kg*cm² to kg*m²)

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Fusion 360 Design                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Joint Extractor                                                     │
│     - Iterates root.joints                                              │
│     - Returns: Dict[joint_name, JointData]                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. Body Extractor                                                      │
│     - Iterates root.occurrences                                         │
│     - Builds kinematic tree from joint parent/child relationships       │
│     - Returns: Dict[body_name, BodyData], tree structure                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. Mesh Exporter                                                       │
│     - Exports STL for each body                                         │
│     - Returns: Dict[body_name, mesh_filepath]                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. MJCF Generator                                                      │
│     - Combines all data into MJCF XML                                   │
│     - Writes: model.xml + meshes/*.stl                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Fusion 360 API Patterns

### Accessing Design
```python
import adsk.core
import adsk.fusion

app = adsk.core.Application.get()
design = adsk.fusion.Design.cast(app.activeProduct)
root = design.rootComponent
```

### Iterating Joints
```python
for joint in root.joints:
    joint_type = joint.jointMotion.jointType  # RevoluteJointType, etc.
    parent = joint.occurrenceTwo  # Parent component occurrence
    child = joint.occurrenceOne   # Child component occurrence

    # For revolute joints
    if joint_type == adsk.fusion.JointTypes.RevoluteJointType:
        axis = joint.jointMotion.rotationAxisVector.asArray()
        limits = joint.jointMotion.rotationLimits
```

### Iterating Occurrences (Bodies)
```python
for occ in root.allOccurrences:
    name = occ.name
    transform = occ.transform  # Matrix3D

    # Get physical properties
    props = occ.getPhysicalProperties(
        adsk.fusion.CalculationAccuracy.HighCalculationAccuracy
    )
    mass = props.mass  # kg
    com = props.centerOfMass  # Point3D in cm
    (ok, ixx, iyy, izz, ixy, iyz, ixz) = props.getXYZMomentsOfInertia()
```

### Exporting STL
```python
export_mgr = design.exportManager
stl_options = export_mgr.createSTLExportOptions(occurrence)
stl_options.filename = "/path/to/mesh.stl"
stl_options.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementMedium
export_mgr.execute(stl_options)
```

## Design Constraints & Requirements

### From User's Model
1. Must have a component named "base_link" as the root
2. Joints must connect components (not internal bodies)
3. Kinematic tree must be acyclic (tree structure, no loops)
4. Component names should be unique and alphanumeric

### From MuJoCo
1. Bodies are nested (parent contains child in XML)
2. Joints are defined inside the child body
3. Joint axis is in child body's local frame
4. Inertial frame is at body origin (we compute offset)

## Future Enhancements

1. **Collision geometry simplification**: Option to use convex hull or primitive shapes instead of full mesh
2. **Visual/Collision separation**: Different meshes for rendering vs physics
3. **Actuator configuration**: UI to specify gear ratios, force limits
4. **Sensor placement**: Add IMU, force/torque sensors
5. **Material properties**: Map Fusion materials to MuJoCo friction/density
