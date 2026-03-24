#!/usr/bin/env python3
"""Run FEA on an STS3215 bracket component."""

from botcad.bracket import BracketSpec, bracket_solid
from botcad.components.servo import STS3215
from botcad.fea.component import analyze_component
from botcad.shapescript.backend_occt import OcctBackend


def main():
    print("Generating STS3215 bracket IR...")
    servo = STS3215()
    spec = BracketSpec()
    program = bracket_solid(servo, spec)

    print("Executing CAD backend...")
    backend = OcctBackend()
    result = backend.execute(program)

    # The output ref is the final fused solid
    solid = result.shapes[program.output_ref.id]

    print("Starting Component FEA...")
    # Torque for STS3215 is 2.94 N-m
    solve_result = analyze_component(solid, result, torque_nm=2.94, res=(15, 15, 15))

    if solve_result:
        u_field, stress_array, vd = solve_result
        stress_np = stress_array.numpy()
        print("\n--- FEA Report: STS3215 Bracket ---")
        print(f"Max Stress: {stress_np.max() / 1e6:.2f} MPa")
        print(f"Safety Factor: {40e6 / stress_np.max():.2f}")

        # --- Visualization Export ---
        from pathlib import Path

        from botcad.fea.export import export_stress_mesh, export_voxel_mesh

        output_dir = Path("bots/so101_arm")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting visualization assets to {output_dir}...")

        # Export Bolt/Push masks
        export_voxel_mesh(
            vd,
            vd.tag_masks["fastener_hole"],
            [0, 255, 0, 255],
            str(output_dir / "fixed_voxels.ply"),
        )

        # Find a load mask
        load_tag = "pocket" if "pocket" in vd.tag_masks else "front_plate"
        if load_tag in vd.tag_masks:
            export_voxel_mesh(
                vd,
                vd.tag_masks[load_tag],
                [255, 0, 0, 255],
                str(output_dir / "loaded_voxels.ply"),
            )

        # Export structure voxels (semi-transparent)
        export_voxel_mesh(
            vd,
            vd.inside_mask,
            [200, 200, 200, 100],
            str(output_dir / "structure_voxels.ply"),
        )

        # Export high-res stress heatmap
        export_stress_mesh(
            solid,
            u_field.space,
            u_field,
            stress_array,
            str(output_dir / "stress_heatmap.ply"),
        )
        print(
            "\nDone. Load .ply files into a 3D viewer (e.g. MeshLab or CloudCompare) to see results."
        )


if __name__ == "__main__":
    main()
