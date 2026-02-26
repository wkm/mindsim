"""Room archetypes — templates that define realistic room compositions.

Each archetype specifies which furniture concepts appear and how many,
replacing pure-random concept selection with structured randomness.

An archetype defines:
    - **required**: concepts that always appear (with count)
    - **optional**: concepts that may appear (with weight + max count)
    - **fill_range**: (min, max) total objects to aim for

Usage:
    archetype = ARCHETYPES["bedroom"]
    picks = archetype.sample(rng, max_objects=16)
    # picks is a list of concept names to place
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConceptSlot:
    """A concept that can appear in a room archetype.

    Attributes:
        concept: Module name in scene_gen/concepts/
        count: Exact count (for required) or max count (for optional)
        weight: Selection probability weight (only used for optional)
    """

    concept: str
    count: int = 1
    weight: float = 1.0


@dataclass(frozen=True)
class Archetype:
    """A room template defining its furniture composition.

    Attributes:
        name: Human-readable archetype name
        required: Concepts that always appear
        optional: Concepts randomly drawn to fill the room
        fill_range: (min_total, max_total) target object count
    """

    name: str
    required: tuple[ConceptSlot, ...] = ()
    optional: tuple[ConceptSlot, ...] = ()
    fill_range: tuple[int, int] = (4, 8)

    def sample(
        self,
        rng: np.random.Generator,
        max_objects: int = 16,
    ) -> list[str]:
        """Sample a list of concept names from this archetype.

        Returns concept names (may contain duplicates). Required concepts
        are always included; optional concepts fill up to the target count.
        """
        picks: list[str] = []

        # Always include required concepts
        for slot in self.required:
            picks.extend([slot.concept] * slot.count)

        # Determine how many optional objects to add
        lo, hi = self.fill_range
        target = int(rng.integers(lo, hi + 1))
        target = min(target, max_objects)
        n_optional = max(0, target - len(picks))

        if n_optional > 0 and self.optional:
            # Build weighted pool, respecting max counts
            pool_names: list[str] = []
            pool_weights: list[float] = []
            for slot in self.optional:
                pool_names.append(slot.concept)
                pool_weights.append(slot.weight)

            weights = np.array(pool_weights, dtype=float)
            weights /= weights.sum()

            # Track how many of each optional we've picked
            counts: dict[str, int] = {}
            max_counts = {s.concept: s.count for s in self.optional}

            for _ in range(n_optional):
                # Mask out concepts that hit their max
                mask = np.array(
                    [
                        1.0 if counts.get(n, 0) < max_counts.get(n, 1) else 0.0
                        for n in pool_names
                    ]
                )
                masked = weights * mask
                if masked.sum() == 0:
                    break  # all maxed out
                masked /= masked.sum()

                idx = int(rng.choice(len(pool_names), p=masked))
                name = pool_names[idx]
                picks.append(name)
                counts[name] = counts.get(name, 0) + 1

        return picks


# ---------------------------------------------------------------------------
# Room definitions
# ---------------------------------------------------------------------------

ARCHETYPES: dict[str, Archetype] = {
    "bedroom": Archetype(
        name="Bedroom",
        required=(ConceptSlot("bed", count=1),),
        optional=(
            ConceptSlot("nightstand", count=2, weight=3.0),
            ConceptSlot("dresser", count=1, weight=2.5),
            ConceptSlot("wardrobe", count=1, weight=2.0),
            ConceptSlot("lamp", count=2, weight=2.0),
            ConceptSlot("rug", count=1, weight=1.5),
            ConceptSlot("mirror", count=1, weight=1.5),
            ConceptSlot("chair", count=1, weight=1.0),
            ConceptSlot("plant", count=1, weight=1.0),
            ConceptSlot("painting", count=1, weight=1.0),
            ConceptSlot("bookstack", count=1, weight=0.5),
            ConceptSlot("crate", count=1, weight=0.5),
        ),
        fill_range=(4, 8),
    ),
    "living_room": Archetype(
        name="Living Room",
        required=(ConceptSlot("couch", count=1),),
        optional=(
            ConceptSlot("tv_stand", count=1, weight=3.0),
            ConceptSlot("table", count=1, weight=2.5),  # coffee table
            ConceptSlot("bookcase", count=1, weight=2.0),
            ConceptSlot("chair", count=2, weight=2.0),
            ConceptSlot("lamp", count=2, weight=2.0),
            ConceptSlot("rug", count=1, weight=2.0),
            ConceptSlot("shelf", count=1, weight=1.5),
            ConceptSlot("plant", count=2, weight=1.5),
            ConceptSlot("painting", count=2, weight=1.5),
            ConceptSlot("mirror", count=1, weight=1.0),
            ConceptSlot("bookstack", count=1, weight=0.8),
            ConceptSlot("crate", count=1, weight=0.5),
        ),
        fill_range=(5, 10),
    ),
    "office": Archetype(
        name="Office",
        required=(
            ConceptSlot("desk", count=1),
            ConceptSlot("chair", count=1),
        ),
        optional=(
            ConceptSlot("bookcase", count=1, weight=3.0),
            ConceptSlot("filing_cabinet", count=2, weight=3.0),
            ConceptSlot("shelf", count=2, weight=2.5),
            ConceptSlot("lamp", count=1, weight=2.0),
            ConceptSlot("trash_can", count=1, weight=2.0),
            ConceptSlot("crate", count=2, weight=1.5),
            ConceptSlot("plant", count=1, weight=1.5),
            ConceptSlot("bookstack", count=2, weight=1.5),
            ConceptSlot("painting", count=1, weight=1.0),
            ConceptSlot("rug", count=1, weight=1.0),
        ),
        fill_range=(4, 8),
    ),
    "dining_room": Archetype(
        name="Dining Room",
        required=(
            ConceptSlot("table", count=1),
            ConceptSlot("chair", count=2),
        ),
        optional=(
            ConceptSlot("chair", count=4, weight=3.0),  # more chairs around table
            ConceptSlot("bookcase", count=1, weight=2.0),  # china cabinet
            ConceptSlot("shelf", count=1, weight=1.5),
            ConceptSlot("lamp", count=1, weight=1.5),
            ConceptSlot("rug", count=1, weight=1.5),
            ConceptSlot("painting", count=2, weight=1.5),
            ConceptSlot("plant", count=1, weight=1.0),
            ConceptSlot("dresser", count=1, weight=1.0),  # sideboard / buffet
        ),
        fill_range=(5, 10),
    ),
    "kitchen": Archetype(
        name="Kitchen",
        required=(
            ConceptSlot("kitchen_counter", count=1),
            ConceptSlot("fridge", count=1),
        ),
        optional=(
            ConceptSlot("stove", count=1, weight=3.0),
            ConceptSlot("kitchen_counter", count=2, weight=2.5),  # extra counters
            ConceptSlot("table", count=1, weight=2.0),  # small kitchen table
            ConceptSlot("chair", count=2, weight=2.0),
            ConceptSlot("trash_can", count=1, weight=2.0),
            ConceptSlot("shelf", count=1, weight=1.5),
            ConceptSlot("plant", count=1, weight=1.0),
        ),
        fill_range=(4, 8),
    ),
    "bathroom": Archetype(
        name="Bathroom",
        required=(
            ConceptSlot("toilet", count=1),
            ConceptSlot("sink_vanity", count=1),
        ),
        optional=(
            ConceptSlot("bathtub", count=1, weight=3.0),
            ConceptSlot("mirror", count=1, weight=2.5),
            ConceptSlot("trash_can", count=1, weight=2.0),
            ConceptSlot("shelf", count=1, weight=1.5),
            ConceptSlot("rug", count=1, weight=1.5),
            ConceptSlot("plant", count=1, weight=1.0),
        ),
        fill_range=(3, 6),
    ),
    "industrial": Archetype(
        name="Industrial",
        required=(
            ConceptSlot("safety_cone", count=2),
            ConceptSlot("crate", count=2),
        ),
        optional=(
            ConceptSlot("safety_cone", count=4, weight=3.0),
            ConceptSlot("crate", count=4, weight=3.0),
            ConceptSlot("bench", count=2, weight=2.0),
            ConceptSlot("trash_can", count=1, weight=1.5),
            ConceptSlot("shelf", count=2, weight=1.5),
            ConceptSlot("lamp", count=1, weight=0.5),
        ),
        fill_range=(6, 12),
    ),
}


def list_archetypes() -> list[str]:
    """List available archetype names."""
    return sorted(ARCHETYPES.keys())


def get(name: str) -> Archetype:
    """Get an archetype by name. Raises KeyError if not found."""
    return ARCHETYPES[name]
