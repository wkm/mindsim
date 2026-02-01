"""
Mesh export functionality for Fusion 360 to MuJoCo.

Exports component meshes as STL files for use in MuJoCo simulations.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

from .transforms import sanitize_name

if TYPE_CHECKING:
    import adsk.fusion
    import adsk.core


@dataclass
class MeshExportResult:
    """Result of exporting a single mesh."""
    body_name: str
    file_path: str
    success: bool
    error: Optional[str] = None


class MeshRefinement:
    """Mesh refinement levels for STL export."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


def export_meshes(
    design,
    bodies: Dict[str, "BodyData"],
    output_dir: str,
    refinement: str = MeshRefinement.MEDIUM,
) -> Dict[str, MeshExportResult]:
    """
    Export STL meshes for all bodies.

    Args:
        design: adsk.fusion.Design - The active design
        bodies: Dictionary of body data
        output_dir: Directory to save mesh files
        refinement: Mesh refinement level

    Returns:
        Dictionary mapping body names to export results
    """
    import adsk.fusion

    results: Dict[str, MeshExportResult] = {}

    # Create meshes subdirectory
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # Get export manager
    export_mgr = design.exportManager

    # Map refinement level
    refinement_setting = _get_refinement_setting(refinement)

    # Get root component
    root = design.rootComponent

    # Export root component mesh (base_link)
    if "base_link" in bodies:
        result = _export_component_mesh(
            export_mgr,
            root,
            "base_link",
            meshes_dir,
            refinement_setting,
        )
        results["base_link"] = result

    # Export each occurrence's mesh
    for occ in root.allOccurrences:
        body_name = sanitize_name(occ.name)

        if body_name not in bodies:
            continue

        result = _export_occurrence_mesh(
            export_mgr,
            occ,
            body_name,
            meshes_dir,
            refinement_setting,
        )
        results[body_name] = result

    return results


def _get_refinement_setting(refinement: str):
    """Map refinement string to Fusion 360 enum."""
    import adsk.fusion

    mapping = {
        MeshRefinement.LOW: adsk.fusion.MeshRefinementSettings.MeshRefinementLow,
        MeshRefinement.MEDIUM: adsk.fusion.MeshRefinementSettings.MeshRefinementMedium,
        MeshRefinement.HIGH: adsk.fusion.MeshRefinementSettings.MeshRefinementHigh,
    }
    return mapping.get(refinement, adsk.fusion.MeshRefinementSettings.MeshRefinementMedium)


def _export_component_mesh(
    export_mgr,
    component,
    name: str,
    output_dir: str,
    refinement_setting,
) -> MeshExportResult:
    """Export mesh for a component (used for root component)."""
    file_path = os.path.join(output_dir, f"{name}.stl")

    try:
        # Create STL export options
        stl_options = export_mgr.createSTLExportOptions(component)
        stl_options.filename = file_path
        stl_options.meshRefinement = refinement_setting

        # Execute export
        export_mgr.execute(stl_options)

        return MeshExportResult(
            body_name=name,
            file_path=file_path,
            success=True,
        )
    except Exception as e:
        return MeshExportResult(
            body_name=name,
            file_path=file_path,
            success=False,
            error=str(e),
        )


def _export_occurrence_mesh(
    export_mgr,
    occurrence,
    name: str,
    output_dir: str,
    refinement_setting,
) -> MeshExportResult:
    """Export mesh for an occurrence."""
    file_path = os.path.join(output_dir, f"{name}.stl")

    try:
        # Create STL export options for the occurrence
        stl_options = export_mgr.createSTLExportOptions(occurrence)
        stl_options.filename = file_path
        stl_options.meshRefinement = refinement_setting

        # Execute export
        export_mgr.execute(stl_options)

        return MeshExportResult(
            body_name=name,
            file_path=file_path,
            success=True,
        )
    except Exception as e:
        return MeshExportResult(
            body_name=name,
            file_path=file_path,
            success=False,
            error=str(e),
        )


def export_mesh_for_body(
    export_mgr,
    body_entity,
    name: str,
    output_dir: str,
    refinement: str = MeshRefinement.MEDIUM,
) -> MeshExportResult:
    """
    Export a single mesh for a body entity.

    Args:
        export_mgr: Design's export manager
        body_entity: Body or occurrence to export
        name: Name for the output file
        output_dir: Directory to save the file
        refinement: Mesh refinement level

    Returns:
        MeshExportResult with status
    """
    refinement_setting = _get_refinement_setting(refinement)
    file_path = os.path.join(output_dir, f"{name}.stl")

    try:
        stl_options = export_mgr.createSTLExportOptions(body_entity)
        stl_options.filename = file_path
        stl_options.meshRefinement = refinement_setting

        export_mgr.execute(stl_options)

        return MeshExportResult(
            body_name=name,
            file_path=file_path,
            success=True,
        )
    except Exception as e:
        return MeshExportResult(
            body_name=name,
            file_path=file_path,
            success=False,
            error=str(e),
        )


def get_mesh_relative_path(mesh_name: str) -> str:
    """Get the relative path to a mesh file for use in MJCF."""
    return f"meshes/{mesh_name}.stl"
