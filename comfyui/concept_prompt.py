"""Generate image prompts from scene_gen concepts for the ComfyUI pipeline.

Usage:
    # List all concepts and their variations
    python concept_prompt.py --list

    # Generate a prompt for a specific concept variation
    python concept_prompt.py table "dining table"

    # Generate prompt for default params
    python concept_prompt.py chair

    # Generate workflow JSON with prompt baked in
    python concept_prompt.py table "coffee table" --workflow > my_workflow.json

    # Batch: generate prompts for ALL variations of ALL concepts
    python concept_prompt.py --batch
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Add project root to path so we can import scene_gen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Color name reverse-lookup
# ---------------------------------------------------------------------------

_COLOR_NAMES: dict[tuple[float, float, float, float], str] = {}


def _build_color_names():
    """Build RGBA → human-readable name mapping from primitives module."""
    from scene_gen import primitives

    for name in dir(primitives):
        val = getattr(primitives, name)
        if (
            isinstance(val, tuple)
            and len(val) == 4
            and all(isinstance(v, float) for v in val)
            and name.isupper()
        ):
            human = name.lower().replace("_", " ")
            _COLOR_NAMES[val] = human


def color_name(rgba: tuple[float, float, float, float]) -> str:
    """Best-effort human name for an RGBA color."""
    if not _COLOR_NAMES:
        _build_color_names()
    if rgba in _COLOR_NAMES:
        return _COLOR_NAMES[rgba]
    r, g, b, _a = rgba
    # Fallback: describe the dominant channel
    if r > 0.7 and g > 0.7 and b > 0.7:
        return "white"
    if r < 0.2 and g < 0.2 and b < 0.2:
        return "black"
    if r > g and r > b:
        return "reddish brown" if g > 0.3 else "red"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    return "gray"


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

STYLE_SUFFIX = (
    "Clean product shot on plain white background. Studio lighting, soft shadows. "
    "Slightly chunky, toylike proportions (Sims 1 style). "
    "Isometric 3/4 view from above. Single isolated object, no context."
)


def concept_prompt(concept_name: str, params) -> str:
    """Generate a FLUX-optimized image prompt from a concept name + params.

    Args:
        concept_name: e.g. "table", "chair", "plant"
        params: The frozen Params dataclass instance
    """
    fields = dataclasses.fields(params)
    field_dict = {f.name: getattr(params, f.name) for f in fields}

    # Extract dimensions
    dim_keys = [
        "width",
        "depth",
        "height",
        "seat_width",
        "seat_depth",
        "seat_height",
        "back_height",
        "radius",
        "pot_radius",
        "pot_height",
    ]
    dims = {
        k: v
        for k, v in field_dict.items()
        if k in dim_keys and isinstance(v, (int, float))
    }

    # Extract colors
    color_keys = [
        "color",
        "seat_color",
        "leg_color",
        "cushion_color",
        "body_color",
        "door_color",
        "accent_color",
        "handle_color",
        "shell_color",
        "pot_color",
        "foliage_color",
        "shade_color",
        "frame_color",
        "top_color",
        "drawer_color",
        "panel_color",
    ]
    colors = {}
    for k in color_keys:
        if (
            k in field_dict
            and isinstance(field_dict[k], tuple)
            and len(field_dict[k]) == 4
        ):
            colors[k.replace("_", " ")] = color_name(field_dict[k])

    # Extract boolean features
    bool_features = []
    for k, v in field_dict.items():
        if isinstance(v, bool) and v and k.startswith("has_"):
            bool_features.append(k.replace("has_", "").replace("_", " "))

    # Extract string type fields
    type_fields = {
        k: v
        for k, v in field_dict.items()
        if isinstance(v, str) and k in ("plant_type", "lamp_type", "style")
    }

    # Build the prompt
    parts = []

    # Object description
    obj_desc = concept_name.replace("_", " ")
    if type_fields:
        type_str = ", ".join(f"{v}" for v in type_fields.values())
        parts.append(f"A 3D rendering of a {type_str} {obj_desc}")
    else:
        parts.append(f"A 3D rendering of a {obj_desc}")

    # Dimensions
    if dims:
        dim_str = ", ".join(f"{k.replace('_', ' ')} {v:.2f}m" for k, v in dims.items())
        parts.append(f"Dimensions: {dim_str}")

    # Materials / colors
    if colors:
        mat_str = ", ".join(f"{k}: {v}" for k, v in colors.items())
        parts.append(f"Materials: {mat_str}")

    # Features
    if bool_features:
        parts.append(f"Features: {', '.join(bool_features)}")

    # Special structural notes
    if field_dict.get("pedestal"):
        parts.append("Single thick pedestal base instead of legs")
    if field_dict.get("n_doors"):
        parts.append(f"{field_dict['n_doors']} door(s)")
    if field_dict.get("n_shelves"):
        parts.append(f"{field_dict['n_shelves']} shelves")
    if field_dict.get("n_drawers"):
        parts.append(f"{field_dict['n_drawers']} drawers")

    # Style
    parts.append(STYLE_SUFFIX)

    return ". ".join(parts)


# ---------------------------------------------------------------------------
# Workflow generation
# ---------------------------------------------------------------------------


def make_workflow(prompt: str, seed: int = 42) -> dict:
    """Load the template workflow and inject the prompt text."""
    workflow_path = Path(__file__).parent / "workflow_concept_to_3d.json"
    with open(workflow_path) as f:
        wf = json.load(f)

    # Inject prompt into CLIPTextEncode node (node "3")
    wf["3"]["inputs"]["text"] = prompt

    # Set seeds
    wf["5"]["inputs"]["noise_seed"] = seed
    wf["25"]["inputs"]["seed"] = seed
    wf["26"]["inputs"]["seed"] = seed

    # Remove comment field
    wf.pop("_comment", None)

    return wf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate image prompts from scene_gen concepts"
    )
    parser.add_argument(
        "concept", nargs="?", help="Concept name (e.g. table, chair, plant)"
    )
    parser.add_argument(
        "variation", nargs="?", help="Variation name (e.g. 'dining table')"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all concepts and variations"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Generate prompts for all variations"
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="Output full workflow JSON instead of just prompt",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for generation"
    )
    args = parser.parse_args()

    from scene_gen.concepts import all_concepts

    concepts = all_concepts()

    if args.list:
        for name in sorted(concepts):
            mod = concepts[name]
            variations = getattr(mod, "VARIATIONS", {})
            var_str = f" ({len(variations)} variations)" if variations else ""
            print(f"  {name}{var_str}")
            for vname in sorted(variations):
                print(f"    - {vname}")
        return

    if args.batch:
        results = []
        for name in sorted(concepts):
            mod = concepts[name]
            variations = getattr(mod, "VARIATIONS", {})
            if variations:
                for vname, vparams in sorted(variations.items()):
                    prompt = concept_prompt(name, vparams)
                    results.append(
                        {"concept": name, "variation": vname, "prompt": prompt}
                    )
            else:
                prompt = concept_prompt(name, mod.Params())
                results.append(
                    {"concept": name, "variation": "default", "prompt": prompt}
                )
        json.dump(results, sys.stdout, indent=2)
        print()
        return

    if not args.concept:
        parser.print_help()
        return

    if args.concept not in concepts:
        print(f"Unknown concept: {args.concept}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(concepts))}", file=sys.stderr)
        sys.exit(1)

    mod = concepts[args.concept]
    variations = getattr(mod, "VARIATIONS", {})

    if args.variation:
        if args.variation not in variations:
            print(f"Unknown variation: {args.variation}", file=sys.stderr)
            print(
                f"Available for {args.concept}: {', '.join(sorted(variations))}",
                file=sys.stderr,
            )
            sys.exit(1)
        params = variations[args.variation]
    else:
        params = mod.Params()

    prompt = concept_prompt(args.concept, params)

    if args.workflow:
        wf = make_workflow(prompt, seed=args.seed)
        json.dump(wf, sys.stdout, indent=2)
        print()
    else:
        print(prompt)


if __name__ == "__main__":
    main()
