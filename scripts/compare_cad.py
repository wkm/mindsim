#!/usr/bin/env python3
"""
CAD Comparison Tool (MindSim Physical Diff)

Compares a parametrically generated component against a reference STEP file.
Outputs numerical metrics and generates 'diff' visuals for LLM analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import build123d as b3d


def _as_solid(shape):
    from build123d import Compound, ShapeList, Solid

    """Extract a single Solid from a boolean result or shape list."""
    if isinstance(shape, ShapeList):
        if len(shape) == 1:
            return _as_solid(shape[0])
        if all(isinstance(s, Solid) for s in shape):
            return _as_solid(Compound(list(shape)))
        return shape
    if isinstance(shape, Solid):
        return shape
    if isinstance(shape, Compound):
        solids = shape.solids()
        if len(solids) == 1:
            return solids[0]
        return shape
    return shape


@dataclass
class DiffMetrics:
    gen_volume: float
    ref_volume: float
    vol_diff_pct: float
    extra_vol: float
    missing_vol: float
    gen_bbox: tuple
    ref_bbox: tuple
    com_shift: float


def render_diff_sheet(output_dir: Path, name: str, artifacts: dict[str, Path]):
    """Generates a composite PNG showing the diff from multiple angles."""
    from PIL import Image, ImageDraw

    from botcad.colors import (
        COLOR_COMPARE_EXTRA,
        COLOR_COMPARE_GEN,
        COLOR_COMPARE_MISSING,
    )
    from botcad.emit.composite import FONT_LABEL, FONT_TITLE, save_png
    from botcad.emit.render3d import (
        VIEWS_4,
        Renderer3D,
        SceneBuilder,
    )

    # Diff colors from centralized palette
    COLOR_GEN = COLOR_COMPARE_GEN
    COLOR_EXTRA = COLOR_COMPARE_EXTRA
    COLOR_MISSING = COLOR_COMPARE_MISSING

    # 1. Build Scene
    scene = SceneBuilder(width=1000, height=1000)

    # Add generated (solid) and reference (wireframe/transparent)
    if artifacts.get("generated"):
        scene.add_mesh("gen", str(artifacts["generated"]), COLOR_GEN)
    if artifacts.get("missing"):
        scene.add_mesh("missing", str(artifacts["missing"]), COLOR_MISSING)
    if artifacts.get("extra"):
        scene.add_mesh("extra", str(artifacts["extra"]), COLOR_EXTRA)

    xml = scene.to_xml()

    with Renderer3D(xml, width=1000, height=1000) as r:
        views = r.render_views(VIEWS_4)

    # 2. Composite
    margin = 10
    label_h = 30
    title_h = 50
    cols, rows = 2, 2
    vw, vh = 1000, 1000
    cw = vw * cols + margin * (cols + 1)
    ch = title_h + (vh + label_h) * rows + margin * (rows + 1)

    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (margin, margin), f"Physical Diff: {name}", fill=(0, 0, 0), font=FONT_TITLE
    )

    # Legend
    legend_x = cw - 400
    draw.text((legend_x, margin), "Legend:", fill=(0, 0, 0), font=FONT_LABEL)
    draw.text(
        (legend_x + 80, margin), "Generated", fill=COLOR_GEN.rgb_int, font=FONT_LABEL
    )
    draw.text(
        (legend_x + 180, margin), "Extra", fill=COLOR_EXTRA.rgb_int, font=FONT_LABEL
    )
    draw.text(
        (legend_x + 250, margin), "Missing", fill=COLOR_MISSING.rgb_int, font=FONT_LABEL
    )

    for idx, (img, label) in enumerate(views):
        col = idx % cols
        row = idx // cols
        x = margin + col * (vw + margin)
        y = title_h + margin + row * (vh + label_h + margin)
        draw.text((x, y), label, fill=(100, 100, 100), font=FONT_LABEL)
        canvas.paste(img, (x, y + label_h))

    out_path = output_dir / f"{name}_diff_overview.png"
    save_png(canvas, out_path)
    return out_path


def compare_solids(
    gen_solid: b3d.Shape, ref_solid: b3d.Shape, name: str, output_dir: Path
):
    """Aligns and diffs two solids."""
    from build123d import Location, export_stl

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Basic properties
    gen_vol = gen_solid.volume
    ref_vol = ref_solid.volume

    # 2. Alignment
    # Align X and Y by bounding box center
    gen_bbox = gen_solid.bounding_box()
    ref_bbox = ref_solid.bounding_box()

    gen_aligned = gen_solid.locate(
        Location((-gen_bbox.center().X, -gen_bbox.center().Y, 0))
    )
    ref_aligned = ref_solid.locate(
        Location((-ref_bbox.center().X, -ref_bbox.center().Y, 0))
    )

    # Align Z by finding the top-most large horizontal face (the plastic cap)
    # This ignores bosses like the shaft or step which might be asymmetric.
    def find_top_face_z(shape, name):
        # Filter for faces with +Z normal and area large enough to be the cap
        faces = [
            f
            for f in shape.faces()
            if f.normal_at(f.center()).Z > 0.99 and f.area > 1e-4
        ]
        if not faces:
            print(f"  {name}: No large horizontal faces found.")
            return shape.bounding_box().max.Z

        # Sort by Z
        faces.sort(key=lambda f: f.center().Z, reverse=True)
        for i, f in enumerate(faces[:3]):
            print(f"  {name} face {i}: Z={f.center().Z:.5f}, area={f.area:.2e}")

        # We want the CAP face, which is the largest horizontal face usually.
        # But for servos, the STEP is higher but smaller area.
        # Let's pick the largest area face among the top few.
        best_face = max(faces[:2], key=lambda f: f.area)
        return best_face.center().Z

    gen_top_z = find_top_face_z(gen_aligned, "Gen")
    ref_top_z = find_top_face_z(ref_aligned, "Ref")

    print(f"Aligning DATUM top faces: Gen={gen_top_z:.5f}, Ref={ref_top_z:.5f}")

    # Shift so top faces match exactly at Z=0
    gen_aligned = gen_aligned.locate(Location((0, 0, -gen_top_z)))
    ref_aligned = ref_aligned.locate(Location((0, 0, -ref_top_z)))

    # 3. Boolean Diff
    # Extra = Material in generated but NOT in reference
    try:
        extra = _as_solid(gen_aligned - ref_aligned)
        extra_vol = extra.volume if hasattr(extra, "volume") else 0.0
    except Exception:
        extra_vol = 0.0
        extra = None

    # Missing = Material in reference but NOT in generated
    try:
        missing = _as_solid(ref_aligned - gen_aligned)
        missing_vol = missing.volume if hasattr(missing, "volume") else 0.0
    except Exception:
        missing_vol = 0.0
        missing = None

    # 4. Metrics
    gen_bbox_aligned = gen_aligned.bounding_box()
    ref_bbox_aligned = ref_aligned.bounding_box()
    metrics = DiffMetrics(
        gen_volume=gen_vol,
        ref_volume=ref_vol,
        vol_diff_pct=((gen_vol - ref_vol) / ref_vol) * 100 if ref_vol > 0 else 0,
        extra_vol=extra_vol,
        missing_vol=missing_vol,
        gen_bbox=(tuple(gen_bbox_aligned.min), tuple(gen_bbox_aligned.max)),
        ref_bbox=(tuple(ref_bbox_aligned.min), tuple(ref_bbox_aligned.max)),
        com_shift=(gen_aligned.center() - ref_aligned.center()).length,
    )

    # 5. Export STLs
    artifacts = {}
    gen_path = output_dir / f"{name}_generated.stl"
    export_stl(gen_aligned, str(gen_path))
    artifacts["generated"] = gen_path

    ref_path = output_dir / f"{name}_reference.stl"
    export_stl(ref_aligned, str(ref_path))
    artifacts["reference"] = ref_path

    if extra and extra_vol > 1e-12:
        extra_path = output_dir / f"{name}_extra.stl"
        export_stl(extra, str(extra_path))
        artifacts["extra"] = extra_path

    if missing and missing_vol > 1e-12:
        missing_path = output_dir / f"{name}_missing.stl"
        export_stl(missing, str(missing_path))
        artifacts["missing"] = missing_path

    # 6. Render PNG
    print("Generating visual diff PNG...")
    render_diff_sheet(output_dir, name, artifacts)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compare parametric component to STEP reference"
    )
    parser.add_argument("--component", type=str, help="Component name (e.g. STS3215)")
    parser.add_argument(
        "--ref", type=str, required=True, help="Path to reference STEP file"
    )
    parser.add_argument(
        "--out", type=str, default="diff_results", help="Output directory"
    )
    parser.add_argument(
        "--roi", type=str, help="ROI box as 'xmin,xmax,ymin,ymax,zmin,zmax' in mm"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Exclusion box as 'xmin,xmax,ymin,ymax,zmin,zmax' in mm",
    )
    parser.add_argument(
        "--exclude-sphere", type=str, help="Exclusion sphere as 'x,y,z,radius' in mm"
    )
    args = parser.parse_args()

    import build123d as b3d
    from build123d import Compound, import_step

    # Load component
    if args.component == "STS3215":
        from botcad.bracket import servo_solid
        from botcad.components.servo import STS3215

        gen_solid = servo_solid(STS3215())
    else:
        print(f"Unknown component: {args.component}")
        sys.exit(1)

    # Load reference
    print(f"Loading reference: {args.ref}")
    ref_shape = import_step(args.ref)

    if isinstance(ref_shape, Compound):
        solids = ref_shape.solids()
        if len(solids) > 1:
            print(f"Reference has {len(solids)} solids. Fusing for comparison...")
            ref_solid = _as_solid(solids[0].fuse(*solids[1:]))
        else:
            ref_solid = solids[0]
    else:
        ref_solid = ref_shape

    # Unit Detection
    if ref_solid.volume > gen_solid.volume * 1000:
        print(
            "Reference volume is extremely large. Assuming units are mm, scaling to m..."
        )
        ref_solid = ref_solid.scale(0.001)

    # ROI Clipping
    if args.roi:
        try:
            coords = [float(x) / 1000.0 for x in args.roi.split(",")]
            xmin, xmax, ymin, ymax, zmin, zmax = coords
            roi_box = b3d.Box(xmax - xmin, ymax - ymin, zmax - zmin).locate(
                b3d.Location(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
            )
            print(f"Clipping reference to ROI: {args.roi} mm")
            ref_solid = _as_solid(ref_solid.intersect(roi_box))
        except Exception as e:
            print(f"ROI clipping failed: {e}")

    # Exclusion Masking (Box)
    if args.exclude:
        try:
            coords = [float(x) / 1000.0 for x in args.exclude.split(",")]
            xmin, xmax, ymin, ymax, zmin, zmax = coords
            mask_box = b3d.Box(xmax - xmin, ymax - ymin, zmax - zmin).locate(
                b3d.Location(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
            )
            print(f"Masking out geometry in exclusion box: {args.exclude} mm")
            ref_solid = _as_solid(ref_solid - mask_box)
        except Exception as e:
            print(f"Exclusion masking failed: {e}")

    # Exclusion Masking (Sphere)
    if args.exclude_sphere:
        try:
            coords = [float(x) / 1000.0 for x in args.exclude_sphere.split(",")]
            x, y, z, r = coords
            mask_sphere = b3d.Sphere(r).locate(
                b3d.Location((x / 1000.0, y / 1000.0, z / 1000.0))
            )
            print(f"Masking out geometry in exclusion sphere: {args.exclude_sphere} mm")
            ref_solid = _as_solid(ref_solid - mask_sphere)
        except Exception as e:
            print(f"Spherical exclusion masking failed: {e}")

    output_dir = Path(args.out) / args.component
    metrics = compare_solids(gen_solid, ref_solid, args.component, output_dir)

    # Output LLM-friendly report
    report = {
        "component": args.component,
        "metrics": {
            "volume_gen_mm3": metrics.gen_volume * 1e9,
            "volume_ref_mm3": metrics.ref_volume * 1e9,
            "volume_diff_pct": metrics.vol_diff_pct,
            "extra_material_mm3": metrics.extra_vol * 1e9,
            "missing_material_mm3": metrics.missing_vol * 1e9,
            "com_shift_mm": metrics.com_shift * 1000,
        },
        "analysis": {
            "summary": f"Generated model is {abs(metrics.vol_diff_pct):.1f}% {'larger' if metrics.vol_diff_pct > 0 else 'smaller'} than reference.",
            "status": "PASS" if abs(metrics.vol_diff_pct) < 1 else "WARNING",
        },
        "artifacts": {
            "png": str(output_dir / f"{args.component}_diff_overview.png"),
            "stls": [str(p) for p in output_dir.glob("*.stl")],
        },
    }

    with open(output_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Physical Diff Report ---")
    print(json.dumps(report, indent=2))
    print(f"\nVisuals exported to: {output_dir}")


if __name__ == "__main__":
    main()
