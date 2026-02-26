"""Furniture groupings — semantic clusters of related objects.

A grouping places multiple objects as a unit with relative offsets,
then applies small jitter for realism. This is the key difference
between "random scatter" and "looks like a real room."

Each grouping defines:
    - An anchor concept (placed first, determines group position)
    - Satellite concepts with offsets relative to the anchor
    - Per-satellite jitter ranges

Usage:
    group = GROUPINGS["desk_setup"]
    placed = group.resolve(rng)
    # Returns list of (concept_name, params, dx, dy, d_rotation) tuples
    # that the composer places relative to the anchor's world position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Satellite:
    """An object placed relative to a group's anchor.

    Offsets are in the anchor's local frame (before world rotation):
        dx: offset along anchor's X axis (left/right)
        dy: offset along anchor's Y axis (front/back)
        d_rotation: rotation offset relative to anchor (radians)
        jitter_pos: max random position offset (meters)
        jitter_rot: max random rotation offset (radians)
    """

    concept: str
    dx: float = 0.0
    dy: float = 0.0
    d_rotation: float = 0.0
    jitter_pos: float = 0.05
    jitter_rot: float = np.radians(5.0)
    optional: bool = False  # if True, may be skipped randomly
    probability: float = 0.7  # probability of inclusion when optional


@dataclass(frozen=True)
class Grouping:
    """A cluster of furniture placed together as a unit.

    The anchor is placed by the normal placement system. Satellites
    are positioned relative to the anchor with small jitter.
    """

    name: str
    anchor: str  # concept name for the anchor object
    satellites: tuple[Satellite, ...] = ()

    def resolve(
        self, rng: np.random.Generator
    ) -> list[tuple[str, float, float, float]]:
        """Resolve which satellites to include and apply jitter.

        Returns list of (concept_name, dx, dy, d_rotation) for each
        satellite that should be placed. Offsets are in the anchor's
        local coordinate frame.
        """
        result: list[tuple[str, float, float, float]] = []
        for sat in self.satellites:
            if sat.optional and rng.random() > sat.probability:
                continue
            dx = sat.dx + rng.uniform(-sat.jitter_pos, sat.jitter_pos)
            dy = sat.dy + rng.uniform(-sat.jitter_pos, sat.jitter_pos)
            drot = sat.d_rotation + rng.uniform(-sat.jitter_rot, sat.jitter_rot)
            result.append((sat.concept, dx, dy, drot))
        return result


# ---------------------------------------------------------------------------
# Predefined groupings
# ---------------------------------------------------------------------------

GROUPINGS: dict[str, Grouping] = {
    "desk_setup": Grouping(
        name="Desk + Chair",
        anchor="desk",
        satellites=(
            # Chair pulled out in front of desk (facing desk, i.e. rotated 180°)
            # Desk depth ~0.35 half + chair ~0.21 half + gap = ~0.80
            Satellite("chair", dx=0.0, dy=-0.80, d_rotation=np.pi, jitter_pos=0.08),
        ),
    ),
    "bed_setup": Grouping(
        name="Bed + Nightstands",
        anchor="bed",
        satellites=(
            # Nightstand on left side of bed
            # Bed half-width ~0.50-0.75 + nightstand ~0.23 half + gap
            Satellite(
                "nightstand",
                dx=-1.05,
                dy=0.0,
                d_rotation=0.0,
                jitter_pos=0.03,
                optional=True,
                probability=0.8,
            ),
            # Nightstand on right side of bed
            Satellite(
                "nightstand",
                dx=1.05,
                dy=0.0,
                d_rotation=0.0,
                jitter_pos=0.03,
                optional=True,
                probability=0.8,
            ),
        ),
    ),
    "dining_set": Grouping(
        name="Dining Table + Chairs",
        anchor="table",
        satellites=(
            # Chair at front (-Y side)
            # Table half-depth ~0.30-0.35 + chair ~0.21 + gap = ~0.75
            Satellite("chair", dx=0.0, dy=-0.75, d_rotation=0.0),
            # Chair at back (+Y side)
            Satellite("chair", dx=0.0, dy=0.75, d_rotation=np.pi),
            # Chair on left (-X side)
            # Table half-width ~0.50-0.60 + chair ~0.22 + gap = ~0.90
            Satellite(
                "chair",
                dx=-0.90,
                dy=0.0,
                d_rotation=-np.pi / 2,
                optional=True,
                probability=0.6,
            ),
            # Chair on right (+X side)
            Satellite(
                "chair",
                dx=0.90,
                dy=0.0,
                d_rotation=np.pi / 2,
                optional=True,
                probability=0.6,
            ),
        ),
    ),
    "living_set": Grouping(
        name="Couch + Coffee Table",
        anchor="couch",
        satellites=(
            # Coffee table in front of couch
            # Couch half-depth ~0.42 + table ~0.30 + gap = ~1.0
            Satellite("table", dx=0.0, dy=-1.05, d_rotation=0.0, jitter_pos=0.10),
        ),
    ),
    "tv_setup": Grouping(
        name="TV Stand + Couch",
        anchor="tv_stand",
        satellites=(
            # Couch facing the TV, about 2.5m away
            Satellite(
                "couch",
                dx=0.0,
                dy=-2.5,
                d_rotation=np.pi,
                jitter_pos=0.15,
            ),
        ),
    ),
}


def list_groupings() -> list[str]:
    """List available grouping names."""
    return sorted(GROUPINGS.keys())


def get(name: str) -> Grouping:
    """Get a grouping by name. Raises KeyError if not found."""
    return GROUPINGS[name]
