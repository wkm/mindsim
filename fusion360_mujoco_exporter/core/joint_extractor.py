"""
Joint extraction from Fusion 360 designs.

Extracts joint information and maps Fusion 360 joint types to MuJoCo joint types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .transforms import (
    Vector3,
    Transform,
    Quaternion,
    matrix3d_to_transform,
    invert_matrix3d,
    multiply_matrices,
    sanitize_name,
    CM_TO_M,
)

if TYPE_CHECKING:
    import adsk.fusion


class MuJoCoJointType(Enum):
    """MuJoCo joint types."""
    HINGE = "hinge"      # 1-DOF rotation
    SLIDE = "slide"      # 1-DOF translation
    BALL = "ball"        # 3-DOF spherical
    FREE = "free"        # 6-DOF floating
    FIXED = "fixed"      # No motion (welded) - not actually a joint in MuJoCo


@dataclass
class JointLimits:
    """Joint motion limits."""
    lower: float  # radians for hinge, meters for slide
    upper: float
    limited: bool = True


@dataclass
class JointData:
    """Extracted joint data."""
    name: str
    joint_type: MuJoCoJointType
    parent_name: str
    child_name: str
    axis: Vector3
    origin: Transform  # Position/orientation of joint in parent frame
    limits: Optional[JointLimits] = None
    damping: float = 0.1
    friction: float = 0.0

    # For composite joints (cylindrical, planar)
    secondary_joint: Optional["JointData"] = None


# Mapping from Fusion 360 JointTypes enum values to MuJoCo types
# Fusion 360 joint types (from API):
# RigidJointType = 0
# RevoluteJointType = 1
# SliderJointType = 2
# CylindricalJointType = 3
# PinSlotJointType = 4
# PlanarJointType = 5
# BallJointType = 6

FUSION_TO_MUJOCO_JOINT_MAP = {
    0: MuJoCoJointType.FIXED,      # Rigid
    1: MuJoCoJointType.HINGE,      # Revolute
    2: MuJoCoJointType.SLIDE,      # Slider
    3: MuJoCoJointType.HINGE,      # Cylindrical (primary: rotation)
    4: MuJoCoJointType.HINGE,      # PinSlot (primary: rotation)
    5: MuJoCoJointType.SLIDE,      # Planar (primary: slide X)
    6: MuJoCoJointType.BALL,       # Ball
}


def extract_joints(root_component) -> Dict[str, JointData]:
    """
    Extract all joints from a Fusion 360 root component.

    Args:
        root_component: adsk.fusion.Component - The root component of the design

    Returns:
        Dictionary mapping joint names to JointData objects
    """
    import adsk.fusion

    joints_dict: Dict[str, JointData] = {}

    for joint in root_component.joints:
        try:
            joint_data = _extract_single_joint(joint)
            if joint_data:
                joints_dict[joint_data.name] = joint_data
        except Exception as e:
            # Log error but continue with other joints
            print(f"Warning: Failed to extract joint '{joint.name}': {e}")

    return joints_dict


def _extract_single_joint(joint) -> Optional[JointData]:
    """Extract data from a single Fusion 360 joint."""
    import adsk.fusion

    # Get joint motion type
    joint_motion = joint.jointMotion
    fusion_joint_type = joint_motion.jointType

    # Map to MuJoCo type
    mujoco_type = FUSION_TO_MUJOCO_JOINT_MAP.get(fusion_joint_type)
    if mujoco_type is None:
        print(f"Warning: Unknown joint type {fusion_joint_type} for joint '{joint.name}'")
        return None

    # Skip fixed joints (they don't need a joint element in MuJoCo)
    # But we still track them for the kinematic tree
    if mujoco_type == MuJoCoJointType.FIXED:
        # Return minimal data for tree building
        parent_name = _get_occurrence_name(joint.occurrenceTwo)
        child_name = _get_occurrence_name(joint.occurrenceOne)

        return JointData(
            name=sanitize_name(joint.name),
            joint_type=MuJoCoJointType.FIXED,
            parent_name=parent_name,
            child_name=child_name,
            axis=Vector3(0, 0, 1),  # Dummy axis
            origin=_get_joint_origin(joint),
        )

    # Get parent and child occurrences
    # In Fusion 360: occurrenceTwo is the parent, occurrenceOne is the child
    parent_occ = joint.occurrenceTwo
    child_occ = joint.occurrenceOne

    parent_name = _get_occurrence_name(parent_occ)
    child_name = _get_occurrence_name(child_occ)

    # Extract axis based on joint type
    axis = _extract_joint_axis(joint_motion, fusion_joint_type)

    # Extract limits
    limits = _extract_joint_limits(joint_motion, fusion_joint_type)

    # Get joint origin (position in parent frame)
    origin = _get_joint_origin(joint)

    joint_data = JointData(
        name=sanitize_name(joint.name),
        joint_type=mujoco_type,
        parent_name=parent_name,
        child_name=child_name,
        axis=axis,
        origin=origin,
        limits=limits,
    )

    # Handle composite joints (cylindrical, planar)
    if fusion_joint_type == 3:  # Cylindrical
        # Add a secondary slide joint
        slide_axis = axis  # Same axis as rotation
        slide_limits = _extract_slide_limits(joint_motion)
        joint_data.secondary_joint = JointData(
            name=f"{sanitize_name(joint.name)}_slide",
            joint_type=MuJoCoJointType.SLIDE,
            parent_name=parent_name,
            child_name=child_name,
            axis=slide_axis,
            origin=origin,
            limits=slide_limits,
        )
    elif fusion_joint_type == 5:  # Planar
        # Add a secondary slide joint (Y direction)
        # Get perpendicular axis
        slide_axis_y = _get_perpendicular_axis(axis)
        slide_limits_y = _extract_slide_limits(joint_motion)
        joint_data.secondary_joint = JointData(
            name=f"{sanitize_name(joint.name)}_slide_y",
            joint_type=MuJoCoJointType.SLIDE,
            parent_name=parent_name,
            child_name=child_name,
            axis=slide_axis_y,
            origin=origin,
            limits=slide_limits_y,
        )

    return joint_data


def _get_occurrence_name(occurrence) -> str:
    """Get sanitized name of an occurrence, handling None (root component)."""
    if occurrence is None:
        return "base_link"
    return sanitize_name(occurrence.name)


def _extract_joint_axis(joint_motion, fusion_joint_type: int) -> Vector3:
    """Extract the joint axis vector based on joint type."""
    import adsk.fusion

    try:
        if fusion_joint_type in [1, 3, 4]:  # Revolute, Cylindrical, PinSlot
            axis_vec = joint_motion.rotationAxisVector
            if axis_vec:
                arr = axis_vec.asArray()
                return Vector3(arr[0], arr[1], arr[2])
        elif fusion_joint_type == 2:  # Slider
            axis_vec = joint_motion.slideDirectionVector
            if axis_vec:
                arr = axis_vec.asArray()
                return Vector3(arr[0], arr[1], arr[2])
        elif fusion_joint_type == 5:  # Planar - primary slide direction
            axis_vec = joint_motion.primarySlideDirectionVector
            if axis_vec:
                arr = axis_vec.asArray()
                return Vector3(arr[0], arr[1], arr[2])
        elif fusion_joint_type == 6:  # Ball - no specific axis
            return Vector3(0, 0, 1)  # Default
    except Exception:
        # Axis vector properties may not exist for all joint motion types,
        # or may fail if the joint is in an invalid state
        pass

    # Default axis (Z-up)
    return Vector3(0, 0, 1)


def _extract_joint_limits(joint_motion, fusion_joint_type: int) -> Optional[JointLimits]:
    """Extract joint limits for rotation or translation."""
    import adsk.fusion

    try:
        if fusion_joint_type in [1, 3, 4]:  # Rotational joints
            limits = joint_motion.rotationLimits
            if limits:
                max_enabled = limits.isMaximumValueEnabled
                min_enabled = limits.isMinimumValueEnabled

                if max_enabled and min_enabled:
                    return JointLimits(
                        lower=limits.minimumValue,  # Already in radians
                        upper=limits.maximumValue,
                        limited=True,
                    )
                elif not max_enabled and not min_enabled:
                    # Continuous joint (no limits)
                    return None

        elif fusion_joint_type == 2:  # Slider
            return _extract_slide_limits(joint_motion)

    except Exception:
        # Limit properties may not be available for all joint types,
        # or limits may not be configured - treat as unlimited
        pass

    return None


def _extract_slide_limits(joint_motion) -> Optional[JointLimits]:
    """Extract slide/translation limits."""
    try:
        limits = joint_motion.slideLimits
        if limits:
            max_enabled = limits.isMaximumValueEnabled
            min_enabled = limits.isMinimumValueEnabled

            if max_enabled and min_enabled:
                return JointLimits(
                    lower=limits.minimumValue * CM_TO_M,  # Convert cm to m
                    upper=limits.maximumValue * CM_TO_M,
                    limited=True,
                )
    except Exception:
        # Slide limits may not be available for non-sliding joints
        # or when limits haven't been configured
        pass

    return None


def _get_joint_origin(joint) -> Transform:
    """
    Get the joint origin transform in the parent body's frame.

    This is where the joint is located relative to the parent.
    """
    try:
        # Get the joint geometry
        geometry = joint.geometryOrOriginOne

        if geometry:
            # Try to get origin point
            if hasattr(geometry, "origin"):
                origin = geometry.origin
                pos = Vector3(
                    origin.x * CM_TO_M,
                    origin.y * CM_TO_M,
                    origin.z * CM_TO_M,
                )
                return Transform(pos, Quaternion(1, 0, 0, 0))
    except Exception:
        # Joint geometry access can fail for certain joint types or
        # when geometry hasn't been fully computed yet
        pass

    # Fallback: compute child transform relative to parent (parent^-1 * child)
    try:
        parent_occ = joint.occurrenceTwo
        child_occ = joint.occurrenceOne

        if child_occ:
            child_matrix = child_occ.transform.asArray()

            if parent_occ:
                # Compute relative transform: parent^-1 * child
                parent_matrix = parent_occ.transform.asArray()
                parent_inv = invert_matrix3d(parent_matrix)
                relative_matrix = multiply_matrices(parent_inv, child_matrix)
                return matrix3d_to_transform(relative_matrix)
            else:
                # No parent (connected to root), use child's world transform
                return matrix3d_to_transform(child_matrix)
    except Exception:
        # Transform access can fail if occurrences are suppressed or invalid
        pass

    # Default: identity transform
    return Transform.identity()


def _get_perpendicular_axis(axis: Vector3) -> Vector3:
    """Get a vector perpendicular to the given axis."""
    import math

    # Use Gram-Schmidt to find perpendicular
    if abs(axis.x) < 0.9:
        perp = Vector3(1, 0, 0)
    else:
        perp = Vector3(0, 1, 0)

    # perp = perp - (perp . axis) * axis
    dot = perp.x * axis.x + perp.y * axis.y + perp.z * axis.z
    perp = Vector3(
        perp.x - dot * axis.x,
        perp.y - dot * axis.y,
        perp.z - dot * axis.z,
    )

    # Normalize
    length = math.sqrt(perp.x ** 2 + perp.y ** 2 + perp.z ** 2)
    if length > 0:
        return Vector3(perp.x / length, perp.y / length, perp.z / length)

    return Vector3(0, 1, 0)


def build_kinematic_tree(
    joints: Dict[str, JointData],
    root_name: str = "base_link",
) -> Dict[str, List[str]]:
    """
    Build a kinematic tree from joint parent/child relationships.

    Returns a dictionary mapping parent names to lists of child names.
    """
    tree: Dict[str, List[str]] = {}

    for joint_data in joints.values():
        parent = joint_data.parent_name
        child = joint_data.child_name

        if parent not in tree:
            tree[parent] = []
        if child not in tree[parent]:
            tree[parent].append(child)

    return tree


def get_joint_for_body(
    body_name: str,
    joints: Dict[str, JointData],
) -> Optional[JointData]:
    """Get the joint that connects a body to its parent."""
    for joint_data in joints.values():
        if joint_data.child_name == body_name:
            return joint_data
    return None
