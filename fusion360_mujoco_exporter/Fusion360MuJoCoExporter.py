"""
Fusion 360 to MuJoCo Exporter Add-in

Exports Fusion 360 assemblies with joints to MuJoCo-compatible MJCF XML files.

Installation:
1. Copy this folder to:
   - Windows: C:\\Users\\<user>\\AppData\\Roaming\\Autodesk\\Autodesk Fusion 360\\API\\AddIns\\
   - Mac: ~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns/
2. Restart Fusion 360
3. Go to Tools > Add-Ins and enable "Fusion360MuJoCoExporter"
4. Find "Export to MuJoCo" in the Tools menu

Usage:
1. Open a Fusion 360 design with components and joints
2. Ensure you have a component named "base_link" as the root
3. Run "Export to MuJoCo" from the Tools menu
4. Select output folder
5. The exporter generates:
   - model.xml (MJCF file)
   - meshes/*.stl (mesh files for each component)
"""

import os
import traceback

import adsk.core
import adsk.fusion

# Global variables for add-in lifecycle
_app: adsk.core.Application = None
_ui: adsk.core.UserInterface = None
_handlers = []

# Command identifiers
CMD_ID = "Fusion360MuJoCoExporter"
CMD_NAME = "Export to MuJoCo"
CMD_DESCRIPTION = "Export assembly with joints to MuJoCo MJCF format"


class ExportCommandCreatedHandler(adsk.core.CommandCreatedEventHandler):
    """Handler for command creation - sets up the command dialog."""

    def __init__(self):
        super().__init__()

    def notify(self, args: adsk.core.CommandCreatedEventArgs):
        try:
            cmd = args.command
            inputs = cmd.commandInputs

            # Add model name input
            inputs.addStringValueInput(
                "modelName",
                "Model Name",
                _get_default_model_name(),
            )

            # Add output folder browser
            inputs.addBoolValueInput(
                "selectFolder",
                "Select Output Folder",
                False,
            )

            # Add mesh quality dropdown
            mesh_quality = inputs.addDropDownCommandInput(
                "meshQuality",
                "Mesh Quality",
                adsk.core.DropDownStyles.TextListDropDownStyle,
            )
            mesh_quality.listItems.add("Low", False)
            mesh_quality.listItems.add("Medium", True)  # Default
            mesh_quality.listItems.add("High", False)

            # Add info text
            inputs.addTextBoxCommandInput(
                "info",
                "",
                "<b>Requirements:</b><br>"
                "- Assembly must have a 'base_link' component<br>"
                "- Joints must connect components (not internal bodies)<br>"
                "- Supported joints: Revolute, Slider, Ball, Rigid",
                4,
                True,
            )

            # Connect execute handler
            on_execute = ExportCommandExecuteHandler()
            cmd.execute.add(on_execute)
            _handlers.append(on_execute)

            # Connect destroy handler
            on_destroy = ExportCommandDestroyHandler()
            cmd.destroy.add(on_destroy)
            _handlers.append(on_destroy)

        except Exception:
            _ui.messageBox(f"Command creation failed:\n{traceback.format_exc()}")


class ExportCommandExecuteHandler(adsk.core.CommandEventHandler):
    """Handler for command execution - performs the export."""

    def __init__(self):
        super().__init__()

    def notify(self, args: adsk.core.CommandEventArgs):
        try:
            # Get command inputs
            inputs = args.command.commandInputs
            model_name = inputs.itemById("modelName").value
            mesh_quality = inputs.itemById("meshQuality").selectedItem.name

            # Get active design
            design = adsk.fusion.Design.cast(_app.activeProduct)
            if not design:
                _ui.messageBox("No active Fusion 360 design found.")
                return

            # Show folder dialog
            folder_dialog = _ui.createFolderDialog()
            folder_dialog.title = "Select Output Folder for MuJoCo Export"

            result = folder_dialog.showDialog()
            if result != adsk.core.DialogResults.DialogOK:
                return

            output_folder = folder_dialog.folder

            # Perform export
            _perform_export(design, model_name, output_folder, mesh_quality)

        except Exception:
            _ui.messageBox(f"Export failed:\n{traceback.format_exc()}")


class ExportCommandDestroyHandler(adsk.core.CommandEventHandler):
    """Handler for command destruction."""

    def __init__(self):
        super().__init__()

    def notify(self, args: adsk.core.CommandEventArgs):
        # Cleanup if needed
        pass


def _get_default_model_name() -> str:
    """Get default model name from the active design."""
    try:
        design = adsk.fusion.Design.cast(_app.activeProduct)
        if design and design.rootComponent:
            name = design.rootComponent.name
            # Clean up the name
            name = name.split()[0]  # Remove version suffix
            return name
    except Exception:
        pass
    return "robot"


def _perform_export(
    design: adsk.fusion.Design,
    model_name: str,
    output_folder: str,
    mesh_quality: str,
) -> None:
    """
    Perform the complete export process.

    Args:
        design: Active Fusion 360 design
        model_name: Name for the model
        output_folder: Directory to write output files
        mesh_quality: Mesh refinement level ("Low", "Medium", "High")
    """
    # Import core modules
    from .core.joint_extractor import extract_joints
    from .core.body_extractor import extract_bodies, has_base_link_component
    from .core.mesh_exporter import export_meshes, MeshRefinement
    from .core.mjcf_generator import generate_mjcf

    root = design.rootComponent

    # Validate base_link exists in the actual design structure
    # (before extraction, which always creates a synthetic base_link entry)
    if not has_base_link_component(root):
        _ui.messageBox(
            "Error: No 'base_link' component found.\n\n"
            "Please ensure your assembly has:\n"
            "- A component named 'base_link', OR\n"
            "- Joints connected to the root component\n\n"
            "The base_link serves as the root of your kinematic tree."
        )
        return

    # Progress dialog
    progress = _ui.createProgressDialog()
    progress.isCancelButtonShown = False
    progress.show("Exporting to MuJoCo", "Extracting joints...", 0, 4, 0)

    try:
        # Step 1: Extract joints
        progress.message = "Extracting joints..."
        joints = extract_joints(root)
        progress.progressValue = 1

        # Step 2: Extract bodies
        progress.message = "Extracting bodies..."
        bodies = extract_bodies(root, joints)
        progress.progressValue = 2

        # Step 3: Export meshes
        progress.message = "Exporting meshes..."
        refinement = getattr(
            MeshRefinement, mesh_quality.upper(), MeshRefinement.MEDIUM
        )
        mesh_results = export_meshes(design, bodies, output_folder, refinement)
        progress.progressValue = 3

        # Check for mesh export failures
        failed_meshes = [r for r in mesh_results.values() if not r.success]
        if failed_meshes:
            print(f"Warning: Failed to export {len(failed_meshes)} meshes")

        # Step 4: Generate MJCF
        progress.message = "Generating MJCF..."
        mjcf_path = os.path.join(output_folder, f"{model_name}.xml")
        generate_mjcf(model_name, bodies, joints, mjcf_path)
        progress.progressValue = 4

        progress.hide()

        # Show success message
        num_bodies = len(bodies)
        num_joints = len([j for j in joints.values() if j.joint_type.value != "fixed"])

        _ui.messageBox(
            f"Export successful!\n\n"
            f"Model: {model_name}\n"
            f"Bodies: {num_bodies}\n"
            f"Joints: {num_joints}\n\n"
            f"Output folder:\n{output_folder}"
        )

    except Exception:
        progress.hide()
        _ui.messageBox(f"Export failed:\n{traceback.format_exc()}")


def run(context):
    """
    Entry point when the add-in is started.

    Creates the command button in the Tools menu.
    """
    global _app, _ui

    try:
        _app = adsk.core.Application.get()
        _ui = _app.userInterface

        # Get the "Tools" tab
        tools_tab = _ui.allToolbarTabs.itemById("ToolsTab")
        if not tools_tab:
            _ui.messageBox("Could not find Tools tab")
            return

        # Create a new panel or use existing
        panel_id = "SolidScriptsAddinsPanel"
        panel = tools_tab.toolbarPanels.itemById(panel_id)

        if not panel:
            # Try to find any panel in the Tools tab
            if tools_tab.toolbarPanels.count > 0:
                panel = tools_tab.toolbarPanels.item(0)
            else:
                _ui.messageBox("Could not find a panel in Tools tab")
                return

        # Create command definition
        cmd_def = _ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def:
            cmd_def.deleteMe()

        cmd_def = _ui.commandDefinitions.addButtonDefinition(
            CMD_ID,
            CMD_NAME,
            CMD_DESCRIPTION,
        )

        # Connect command created handler
        on_command_created = ExportCommandCreatedHandler()
        cmd_def.commandCreated.add(on_command_created)
        _handlers.append(on_command_created)

        # Add command to panel
        control = panel.controls.itemById(CMD_ID)
        if control:
            control.deleteMe()

        panel.controls.addCommand(cmd_def)

    except Exception:
        if _ui:
            _ui.messageBox(f"Failed to start add-in:\n{traceback.format_exc()}")


def stop(context):
    """
    Entry point when the add-in is stopped.

    Cleans up UI elements.
    """
    global _handlers

    try:
        # Remove command definition
        cmd_def = _ui.commandDefinitions.itemById(CMD_ID)
        if cmd_def:
            cmd_def.deleteMe()

        # Remove from panel
        tools_tab = _ui.allToolbarTabs.itemById("ToolsTab")
        if tools_tab:
            for panel in tools_tab.toolbarPanels:
                control = panel.controls.itemById(CMD_ID)
                if control:
                    control.deleteMe()

        _handlers = []

    except Exception:
        if _ui:
            _ui.messageBox(f"Failed to stop add-in:\n{traceback.format_exc()}")
