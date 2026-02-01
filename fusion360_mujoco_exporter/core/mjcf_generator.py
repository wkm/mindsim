"""
MJCF (MuJoCo XML) generator.

Generates MuJoCo-compatible XML files from extracted Fusion 360 data.
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional
from xml.dom import minidom

from .joint_extractor import JointData, MuJoCoJointType, get_joint_for_body
from .body_extractor import BodyData
from .mesh_exporter import get_mesh_relative_path


class MJCFGenerator:
    """Generates MJCF XML from extracted robot data."""

    def __init__(
        self,
        model_name: str,
        bodies: Dict[str, BodyData],
        joints: Dict[str, JointData],
        mesh_scale: str = "0.001 0.001 0.001",
    ):
        """
        Initialize the MJCF generator.

        Args:
            model_name: Name for the model
            bodies: Dictionary of body data
            joints: Dictionary of joint data
            mesh_scale: Scale factor for meshes (STL in mm -> m)
        """
        self.model_name = model_name
        self.bodies = bodies
        self.joints = joints
        self.mesh_scale = mesh_scale

    def generate(self) -> ET.Element:
        """Generate the complete MJCF XML tree."""
        # Root element
        mujoco = ET.Element("mujoco", model=self.model_name)

        # Add compiler settings
        self._add_compiler(mujoco)

        # Add simulation options
        self._add_option(mujoco)

        # Add defaults
        self._add_defaults(mujoco)

        # Add assets (meshes)
        self._add_assets(mujoco)

        # Add worldbody with nested bodies
        self._add_worldbody(mujoco)

        # Add actuators for motorized joints
        self._add_actuators(mujoco)

        return mujoco

    def generate_xml_string(self, pretty: bool = True) -> str:
        """Generate the MJCF as a formatted XML string."""
        root = self.generate()

        if pretty:
            # Use minidom for pretty printing
            rough_string = ET.tostring(root, encoding="unicode")
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")
        else:
            return ET.tostring(root, encoding="unicode")

    def write_to_file(self, filepath: str) -> None:
        """Write the MJCF to a file."""
        xml_string = self.generate_xml_string(pretty=True)

        # Remove extra blank lines that minidom adds
        lines = xml_string.split("\n")
        lines = [line for line in lines if line.strip()]
        xml_string = "\n".join(lines)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(xml_string)

    def _add_compiler(self, parent: ET.Element) -> None:
        """Add compiler settings."""
        ET.SubElement(
            parent,
            "compiler",
            angle="radian",
            meshdir="meshes",
            autolimits="true",
        )

    def _add_option(self, parent: ET.Element) -> None:
        """Add simulation options."""
        ET.SubElement(
            parent,
            "option",
            gravity="0 0 -9.81",
            timestep="0.002",
        )

    def _add_defaults(self, parent: ET.Element) -> None:
        """Add default settings for joints and geoms."""
        default = ET.SubElement(parent, "default")

        # Joint defaults
        ET.SubElement(
            default,
            "joint",
            damping="0.5",
            armature="0.01",
        )

        # Geom defaults
        ET.SubElement(
            default,
            "geom",
            friction="1 0.005 0.0001",
            condim="3",
        )

        # Motor defaults
        ET.SubElement(
            default,
            "motor",
            ctrllimited="true",
            ctrlrange="-1 1",
        )

    def _add_assets(self, parent: ET.Element) -> None:
        """Add mesh assets."""
        asset = ET.SubElement(parent, "asset")

        # Add a default material
        ET.SubElement(
            asset,
            "material",
            name="default_material",
            rgba="0.7 0.7 0.7 1",
        )

        # Add mesh for each body
        for body_name, body_data in self.bodies.items():
            if body_data.mesh_name:
                mesh_file = get_mesh_relative_path(body_data.mesh_name)
                ET.SubElement(
                    asset,
                    "mesh",
                    name=body_name,
                    file=mesh_file,
                    scale=self.mesh_scale,
                )

    def _add_worldbody(self, parent: ET.Element) -> None:
        """Add worldbody with nested body hierarchy."""
        worldbody = ET.SubElement(parent, "worldbody")

        # Add ground plane
        ET.SubElement(
            worldbody,
            "geom",
            name="ground",
            type="plane",
            size="5 5 0.1",
            rgba="0.9 0.9 0.9 1",
            conaffinity="1",
            condim="3",
        )

        # Add light
        ET.SubElement(
            worldbody,
            "light",
            name="spotlight",
            mode="targetbodycom",
            target="base_link",
            diffuse="0.8 0.8 0.8",
            specular="0.2 0.2 0.2",
            pos="0 -2 3",
        )

        # Add bodies starting from base_link
        self._add_body_recursive(worldbody, "base_link")

    def _add_body_recursive(
        self,
        parent_element: ET.Element,
        body_name: str,
        depth: int = 0,
    ) -> None:
        """Recursively add body and its children."""
        if body_name not in self.bodies:
            return

        body_data = self.bodies[body_name]

        # Create body element
        body_attribs = {"name": body_name}

        # Add position if not identity
        pos = body_data.transform.position
        if not (abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6):
            body_attribs["pos"] = pos.to_mjcf_string()

        # Add orientation if not identity
        rot = body_data.transform.rotation
        if not rot.is_identity():
            body_attribs["quat"] = rot.to_mjcf_string()

        body_elem = ET.SubElement(parent_element, "body", **body_attribs)

        # Add joint if this body is connected to parent via a joint
        joint = get_joint_for_body(body_name, self.joints)
        if joint and joint.joint_type != MuJoCoJointType.FIXED:
            self._add_joint(body_elem, joint)

            # Add secondary joint for composite joints
            if joint.secondary_joint:
                self._add_joint(body_elem, joint.secondary_joint)

        # Add inertial if available
        if body_data.inertial:
            self._add_inertial(body_elem, body_data)

        # Add geom for visualization and collision
        if body_data.mesh_name:
            self._add_geom(body_elem, body_data)

        # Recursively add children
        for child_name in body_data.children:
            self._add_body_recursive(body_elem, child_name, depth + 1)

    def _add_joint(self, parent: ET.Element, joint: JointData) -> None:
        """Add a joint element."""
        joint_attribs = {
            "name": joint.name,
            "type": joint.joint_type.value,
        }

        # Add axis
        axis = joint.axis
        if not (abs(axis.x) < 1e-6 and abs(axis.y) < 1e-6 and abs(axis.z - 1.0) < 1e-6):
            # Only add axis if not default (0 0 1)
            joint_attribs["axis"] = axis.to_mjcf_string()

        # Add limits
        if joint.limits and joint.limits.limited:
            joint_attribs["range"] = f"{joint.limits.lower:.4g} {joint.limits.upper:.4g}"
            joint_attribs["limited"] = "true"

        # Add damping if non-default
        if joint.damping != 0.5:  # Different from default
            joint_attribs["damping"] = f"{joint.damping:.4g}"

        ET.SubElement(parent, "joint", **joint_attribs)

    def _add_inertial(self, parent: ET.Element, body: BodyData) -> None:
        """Add inertial properties."""
        inertial = body.inertial

        inertial_attribs = {
            "mass": f"{inertial.mass:.6g}",
            "diaginertia": inertial.to_mjcf_diaginertia(),
        }

        # Add center of mass position if not at origin
        com = inertial.center_of_mass
        if not (abs(com.x) < 1e-6 and abs(com.y) < 1e-6 and abs(com.z) < 1e-6):
            inertial_attribs["pos"] = com.to_mjcf_string()

        ET.SubElement(parent, "inertial", **inertial_attribs)

    def _add_geom(self, parent: ET.Element, body: BodyData) -> None:
        """Add geom for visualization and collision."""
        ET.SubElement(
            parent,
            "geom",
            type="mesh",
            mesh=body.mesh_name,
            material="default_material",
        )

    def _add_actuators(self, parent: ET.Element) -> None:
        """Add actuators for motorized joints."""
        actuator = ET.SubElement(parent, "actuator")

        for joint_name, joint_data in self.joints.items():
            # Skip fixed joints
            if joint_data.joint_type == MuJoCoJointType.FIXED:
                continue

            # Add motor for each movable joint
            ET.SubElement(
                actuator,
                "motor",
                name=f"motor_{joint_name}",
                joint=joint_name,
                gear="100",  # Default gear ratio
            )

            # Add motor for secondary joint if exists
            if joint_data.secondary_joint:
                sec = joint_data.secondary_joint
                ET.SubElement(
                    actuator,
                    "motor",
                    name=f"motor_{sec.name}",
                    joint=sec.name,
                    gear="100",
                )


def generate_mjcf(
    model_name: str,
    bodies: Dict[str, BodyData],
    joints: Dict[str, JointData],
    output_path: str,
) -> str:
    """
    Generate MJCF file from extracted data.

    Args:
        model_name: Name for the model
        bodies: Dictionary of body data
        joints: Dictionary of joint data
        output_path: Path to write the XML file

    Returns:
        Path to the generated file
    """
    generator = MJCFGenerator(model_name, bodies, joints)
    generator.write_to_file(output_path)
    return output_path
