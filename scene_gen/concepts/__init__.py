"""Concept registry — auto-discovers concept modules in this package.

=== HOW TO ADD A NEW CONCEPT ===

Each concept is a single Python file in this directory. It must define:

1. A frozen dataclass called `Params` with sensible defaults
2. A function `generate(params: Params) -> tuple[Prim, ...]`
   decorated with @lru_cache

That's it. Drop the file here and it's auto-discovered.

Example (scene_gen/concepts/crate.py):

    from dataclasses import dataclass
    from functools import lru_cache

    from scene_gen.primitives import WOOD_DARK, GeomType, Prim

    @dataclass(frozen=True)
    class Params:
        width: float = 0.4     # half-extent X (meters)
        depth: float = 0.3     # half-extent Y
        height: float = 0.3    # half-extent Z
        color: tuple[float, float, float, float] = WOOD_DARK

    @lru_cache(maxsize=128)
    def generate(params: Params = Params()) -> tuple[Prim, ...]:
        w, d, h = params.width, params.depth, params.height
        return (
            Prim(GeomType.BOX, (w, d, h), (0, 0, h), params.color),
        )

=== CONVENTIONS ===

Coordinate system:
    - Z-up, origin at center of footprint on the floor
    - So a 0.75m-tall table has its top at z=0.75

Sizes (match MuJoCo geom_size semantics):
    - BOX: (half_x, half_y, half_z)
    - CYLINDER: (radius, half_height, 0)  — Z-aligned
    - SPHERE: (radius, 0, 0)

Colors:
    - Import from scene_gen.primitives: WOOD_LIGHT, WOOD_MEDIUM,
      WOOD_DARK, METAL_GRAY, FABRIC_BLUE, etc.

Keep it simple:
    - Params should have sensible real-world defaults (meters)
    - generate() should return a tuple (for hashability / lru_cache)
    - Typical furniture uses 3-8 primitives
"""

from __future__ import annotations

import importlib
import pkgutil

_registry: dict[str, object] = {}


def _discover():
    """Auto-discover concept modules that define Params + generate."""
    for info in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f".{info.name}", __package__)
        if hasattr(mod, "generate") and hasattr(mod, "Params"):
            _registry[info.name] = mod


_discover()


def get(name: str):
    """Get a concept module by name. Raises KeyError if not found."""
    return _registry[name]


def list_concepts() -> list[str]:
    """List all available concept names."""
    return sorted(_registry.keys())


def all_concepts() -> dict[str, object]:
    """Return the full registry {name: module}."""
    return dict(_registry)
