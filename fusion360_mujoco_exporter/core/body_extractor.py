"""
Body/component extraction from Fusion 360 designs.

Extracts component information including inertial properties for MuJoCo bodies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from .transforms import (
    Vector3,
    Transform,
    matrix3d_to_transform,
    convert_inertia_to_body_frame,
    sanitize_name,
    CM_TO_M,
    KG_CM2_TO_KG_M2,
)
from .joint_extractor import JointData, build_kinematic_tree, get_joint_for_body

if TYPE_CHECKING:
    pass


@dataclass
class InertialData:
    """Inertial properties for a body."""

    mass: float  # kg
    center_of_mass: Vector3  # meters, in body-local frame
    inertia: Vector3  # diagonal inertia (Ixx, Iyy, Izz) in kg*m²

    def to_mjcf_diaginertia(self) -> str:
        """Format inertia for MuJoCo diaginertia attribute."""
        return f"{self.inertia.x:.6g} {self.inertia.y:.6g} {self.inertia.z:.6g}"


@dataclass
class BodyData:
    """Extracted body/component data."""

    name: str
    transform: Transform  # Position/orientation in parent frame
    inertial: Optional[InertialData] = None
    mesh_name: Optional[str] = None  # Reference to mesh asset
    parent_name: Optional[str] = None
    children: List[str] = None  # Child body names

    def __post_init__(self):
        if self.children is None:
            self.children = []


def extract_bodies(
    root_component,
    joints: Dict[str, JointData],
) -> Dict[str, BodyData]:
    """
    Extract all bodies from a Fusion 360 root component.

    Args:
        root_component: adsk.fusion.Component - The root component
        joints: Dictionary of extracted joints (for kinematic tree)

    Returns:
        Dictionary mapping body names to BodyData objects
    """

    bodies_dict: Dict[str, BodyData] = {}

    # Build kinematic tree from joints
    kinematic_tree = build_kinematic_tree(joints)

    # Track which occurrences we've processed
    processed_names: Set[str] = set()

    # First, add the root/base_link
    base_name = "base_link"
    base_body = _create_base_body(root_component, base_name)
    bodies_dict[base_name] = base_body
    processed_names.add(base_name)

    # Process all occurrences
    for occ in root_component.allOccurrences:
        try:
            body_name = sanitize_name(occ.name)

            if body_name in processed_names:
                continue

            body_data = _extract_single_body(occ, joints)
            if body_data:
                bodies_dict[body_name] = body_data
                processed_names.add(body_name)
        except Exception as e:
            print(f"Warning: Failed to extract body from '{occ.name}': {e}")

    # Build parent-child relationships
    _build_hierarchy(bodies_dict, kinematic_tree)

    return bodies_dict


def _create_base_body(root_component, name: str) -> BodyData:
    """Create the base/root body data."""
    import adsk.fusion

    # Get physical properties of root component
    inertial = None
    try:
        props = root_component.getPhysicalProperties(
            adsk.fusion.CalculationAccuracy.HighCalculationAccuracy
        )
        if props:
            inertial = _extract_inertial(props, Transform.identity())
    except Exception:
        # Physical properties may not be available for empty components
        # or components without solid bodies - proceed without inertial data
        pass

    return BodyData(
        name=name,
        transform=Transform.identity(),
        inertial=inertial,
        mesh_name=name,
        parent_name=None,
        children=[],
    )


def _extract_single_body(
    occurrence, joints: Dict[str, JointData]
) -> Optional[BodyData]:
    """Extract data from a single Fusion 360 occurrence."""
    import adsk.fusion

    body_name = sanitize_name(occurrence.name)

    # Get the joint connecting this body to its parent
    joint = get_joint_for_body(body_name, joints)

    # Get transform
    if joint:
        # Use joint origin for transform
        transform = joint.origin
        parent_name = joint.parent_name
    else:
        # Use occurrence transform
        matrix = occurrence.transform.asArray()
        transform = matrix3d_to_transform(matrix)
        parent_name = "base_link"  # Default parent

    # Get physical properties
    inertial = None
    try:
        props = occurrence.getPhysicalProperties(
            adsk.fusion.CalculationAccuracy.HighCalculationAccuracy
        )
        if props:
            inertial = _extract_inertial(props, transform)
    except Exception:
        # Physical properties calculation can fail for occurrences without
        # solid geometry or when the component is suppressed
        pass

    return BodyData(
        name=body_name,
        transform=transform,
        inertial=inertial,
        mesh_name=body_name,
        parent_name=parent_name,
        children=[],
    )


def _extract_inertial(
    physical_props, body_transform: Transform
) -> Optional[InertialData]:
    """Extract inertial properties from Fusion 360 PhysicalProperties."""
    try:
        # Mass in kg (Fusion uses kg)
        mass = physical_props.mass

        if mass <= 0:
            return None

        # Center of mass (Fusion returns in cm, world coordinates)
        com = physical_props.centerOfMass
        com_world = Vector3(
            com.x * CM_TO_M,
            com.y * CM_TO_M,
            com.z * CM_TO_M,
        )

        # Get moments of inertia (kg*cm², world coordinates)
        success, ixx, iyy, izz, ixy, iyz, ixz = physical_props.getXYZMomentsOfInertia()

        if not success:
            # Use default inertia based on mass
            default_inertia = mass * 0.001  # Small default value
            return InertialData(
                mass=mass,
                center_of_mass=com_world,
                inertia=Vector3(default_inertia, default_inertia, default_inertia),
            )

        # Convert to kg*m²
        ixx *= KG_CM2_TO_KG_M2
        iyy *= KG_CM2_TO_KG_M2
        izz *= KG_CM2_TO_KG_M2
        ixy *= KG_CM2_TO_KG_M2
        iyz *= KG_CM2_TO_KG_M2
        ixz *= KG_CM2_TO_KG_M2

        # Convert to body-local frame at center of mass
        ixx_local, iyy_local, izz_local = convert_inertia_to_body_frame(
            ixx, iyy, izz, ixy, iyz, ixz, com_world, mass
        )

        # Transform center of mass to body-local frame
        # For now, we keep it in world frame and let MuJoCo handle it
        # A more complete implementation would transform it

        return InertialData(
            mass=mass,
            center_of_mass=com_world,
            inertia=Vector3(ixx_local, iyy_local, izz_local),
        )

    except Exception as e:
        print(f"Warning: Failed to extract inertial properties: {e}")
        return None


def _build_hierarchy(
    bodies: Dict[str, BodyData],
    kinematic_tree: Dict[str, List[str]],
) -> None:
    """Build parent-child relationships in bodies dictionary."""
    for parent_name, child_names in kinematic_tree.items():
        if parent_name in bodies:
            bodies[parent_name].children = child_names

        for child_name in child_names:
            if child_name in bodies:
                bodies[child_name].parent_name = parent_name


def has_base_link_component(root_component) -> bool:
    """
    Check if the design has a component that can serve as base_link.

    Returns True if:
    - The root component is named "base_link", OR
    - There's an occurrence named "base_link", OR
    - There are joints that reference "base_link" as parent

    This validates that the user has properly set up their design with
    a base_link before export.
    """
    # Check root component name
    root_name = sanitize_name(root_component.name)
    if root_name == "base_link":
        return True

    # Check occurrences
    for occ in root_component.allOccurrences:
        if sanitize_name(occ.name) == "base_link":
            return True

    # Check if any joint references base_link (occurs when joint connects to root)
    for joint in root_component.joints:
        # If parent occurrence is None, it's connected to root (our base_link)
        if joint.occurrenceTwo is None:
            return True

    # No explicit base_link found
    return False


def get_body_hierarchy(
    bodies: Dict[str, BodyData],
    root_name: str = "base_link",
) -> List[str]:
    """
    Get bodies in hierarchical order (parents before children).

    Uses depth-first traversal from root.
    """
    result = []
    visited = set()

    def visit(name: str):
        if name in visited or name not in bodies:
            return
        visited.add(name)
        result.append(name)

        body = bodies[name]
        for child_name in body.children:
            visit(child_name)

    visit(root_name)

    # Add any orphaned bodies (not connected to root)
    for name in bodies:
        if name not in visited:
            result.append(name)

    return result
