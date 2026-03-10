# New Component Workflow (For AI Agents)

When asked to add or refine a physical component (like a servo, camera, or compute board) in MindSim, follow this rigorous reverse-engineering pipeline.

**Core Philosophy:** *Dimensional Accuracy over Volume Matching.*
Do not manipulate arbitrary dimensions to achieve a specific volume percentage. Compensating errors (e.g., making a part too narrow but too tall) ruins physical fitment. A 3D-printed bracket cares about the exact coordinates of faces and holes, not the aggregate mass.

Follow standard CAD reverse-engineering practices: **Macro to Micro**.

## Phase 1: Ground Truth Acquisition
1.  **Search for Official Specs:** Find the manufacturer's datasheet and dimensions online.
2.  **Acquire Reference CAD:** Find the official manufacturer STEP file (often on GrabCAD, Printables, or GitHub repos).
3.  **Store Reference:** Save the reference STEP file in `references/components/<name>.step`. This folder is ignored by git but used for local `compare_cad` sessions.
4.  **Link in Code:** In the component's Python file (e.g., `botcad/components/servo.py`), add a docstring comment linking to the reference source (URL and local path).

## Phase 2: Interrogation (Measure Twice, Cut Once)
Before writing any parametric geometry, write a temporary Python script (e.g., `inspect_step.py`) using `build123d` to interrogate the reference STEP file in `references/components/`.
Extract precise measurements:
1.  **The Envelope:** Get the exact bounding box (`ref_solid.bounding_box()`).
2.  **Primary Planes:** Find the major structural planes. For example, group faces by their normals (X, Y, Z) and sort by area to find the exact Z-heights where the top cap, middle body, and bottom cap meet.
3.  **Functional Features:** Find the exact centers and radii of cylinders (shafts, mounting holes, screw recesses).

## Phase 3: Parametric Implementation (Macro to Micro)
In `botcad/components/`, build the parametric solid (`servo_solid()`, etc.) iteratively:
1.  **Level 1 - The Envelope (Macro):** Start with basic blocks (`Box`, `Cylinder`) representing the major sub-volumes (top, middle, bottom). Ensure their boundaries perfectly match the Z-planes measured in Phase 2.
2.  **Level 2 - Interfaces:** Add the functional features critical for fitment: mounting flanges, ear holes, output shafts, and connector cutouts. Use exact coordinates from Phase 2.
3.  **Level 3 - Clearances & Contours (Micro):** Add the final details: corner radii (`fillet`, `chamfer`), structural support ribs, and screw head recesses.
*Note: ALWAYS wrap boolean and fillet results in `_as_solid()` to prevent OpenCASCADE `ShapeList` exceptions.*

## Phase 4: Physical Diff Validation
1.  **Run Diff:** Execute the comparison script:
    ```bash
    uv run python -m scripts.compare_cad --component <NAME> --ref references/components/<name>.step
    ```
    *Ensure the script aligns the solids appropriately (e.g., by bounding box center or specific faces) without artificially offsetting them.*
2.  **Visual Inspection (The True Test):** Open the generated `diff_results/<NAME>/<NAME>_diff_overview.png`.
    - **Yellow (Missing):** The parametric model is missing volume that exists in the reference.
    - **Red (Extra):** The parametric model has extra volume that does not exist in the reference.
    - **Goal:** Eliminate all structural Red/Yellow artifacts. Minor discrepancies on complex molded curves or internal hollows are acceptable; dimensional deviations on mounting faces or outer envelopes are not.
3.  **Iterate:** If structural deviations exist, return to Phase 2 to remeasure, then update Phase 3.

## Phase 5: Cleanup & Integration
1.  **Validate Project:** Run `make validate` to ensure your new geometry doesn't break bracket envelopes, collision checks, or tests.
2.  **Specific Component Validation:** If you modified an existing component, run:
    ```bash
    make renders-components
    make renders-rom
    ```
    Review the updated renders in `botcad/components/test_*.png` for visual regressions.
3.  **Clean up:** Remove any temporary inspection scripts and the `diff_results` directory. Keep the reference STEP in `references/components/` for future developers.
