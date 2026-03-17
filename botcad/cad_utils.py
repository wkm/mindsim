"""Shared CAD helpers for build123d shape manipulation."""

from __future__ import annotations


def as_solid(shape):
    """Extract a single Solid from a boolean result or shape list.

    build123d boolean ops (cut, union) and fillets can return Compound
    or ShapeList even when the result is a single solid. This extracts
    the Solid so further operations (like filleting or more booleans)
    work correctly.
    """
    from build123d import Compound, ShapeList, Solid

    # 1. Handle ShapeList (returned by fillet and some boolean ops)
    if isinstance(shape, ShapeList):
        if len(shape) == 1:
            return as_solid(shape[0])
        # If it's a list of multiple solids, try to compound them
        if all(isinstance(s, Solid) for s in shape):
            return as_solid(Compound(list(shape)))
        return shape

    # 2. Handle Solid
    if isinstance(shape, Solid):
        return shape

    # 3. Handle Compound
    if isinstance(shape, Compound):
        solids = shape.solids()
        if len(solids) == 1:
            return solids[0]
        return shape

    return shape
