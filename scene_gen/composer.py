"""Scene composer — maps procedural concepts to MuJoCo model geom slots.

Two-phase workflow:
  1. **Build time** — call SceneComposer.prepare_spec(spec) to inject obstacle
     body+geom slots into an MjSpec before compilation.
  2. **Runtime** — create a SceneComposer(model, data) and call apply() to
     write concept primitives into the slots. No recompilation needed.

Slot naming convention:
    Bodies:  obstacle_0, obstacle_1, ..., obstacle_{N-1}
    Geoms:   obs_0_g0, obs_0_g1, ..., obs_{N-1}_g{M-1}

Usage:
    # Phase 1: inject slots into spec
    spec = mujoco.MjSpec.from_file("worlds/room.xml")
    SceneComposer.prepare_spec(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Phase 2: write scenes at runtime
    composer = SceneComposer(model, data)

    # Option A: random scene
    scene = composer.random_scene(n_objects=3)
    composer.apply(scene)

    # Option B: explicit placement
    from scene_gen.concepts import table
    scene = [
        PlacedObject("table", table.Params(width=1.2), pos=(1, 0), rotation=0.3),
    ]
    composer.apply(scene)

    mujoco.mj_forward(model, data)  # caller must run forward kinematics
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import mujoco
import numpy as np

from scene_gen import concepts
from scene_gen.primitives import Prim, euler_to_quat

# ---------------------------------------------------------------------------
# Defaults for obstacle slot allocation
# ---------------------------------------------------------------------------

DEFAULT_MAX_OBJECTS = 8
DEFAULT_GEOMS_PER_OBJECT = 8

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PlacedObject:
    """An object to place in the scene.

    Attributes:
        concept: Concept module name (e.g., "table", "chair", "shelf")
        params: Concept-specific Params dataclass instance
        pos: (x, y) world position — z is always 0 (floor)
        rotation: Rotation around Z axis in radians
    """

    concept: str
    params: object  # concept's Params dataclass
    pos: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0


@dataclass
class _ObjectSlot:
    """Internal: a pre-allocated body + its child geoms in the model."""

    body_id: int
    geom_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


class SceneComposer:
    """Writes procedural scene geometry into pre-allocated MuJoCo model slots."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._slots: list[_ObjectSlot] = []
        self._discover_slots()

    @staticmethod
    def prepare_spec(
        spec,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        geoms_per_object: int = DEFAULT_GEOMS_PER_OBJECT,
    ):
        """Add obstacle body+geom slots to an MjSpec before compilation.

        Call this once before spec.compile(). The compiled model will contain
        the same named bodies/geoms that SceneComposer discovers at runtime.
        """
        for i in range(max_objects):
            body = spec.worldbody.add_body()
            body.name = f"obstacle_{i}"
            body.pos = [0, 0, 0]
            for j in range(geoms_per_object):
                geom = body.add_geom()
                geom.name = f"obs_{i}_g{j}"
                geom.type = mujoco.mjtGeom.mjGEOM_BOX
                geom.size = [0.001, 0.001, 0.001]
                geom.rgba = [0, 0, 0, 0]
                geom.contype = 0
                geom.conaffinity = 0

    @property
    def max_objects(self) -> int:
        return len(self._slots)

    @property
    def max_geoms_per_object(self) -> int:
        if not self._slots:
            return 0
        return len(self._slots[0].geom_ids)

    def _discover_slots(self):
        """Find obstacle_N bodies and their obs_N_gM child geoms."""
        i = 0
        while True:
            body_name = f"obstacle_{i}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                break

            geom_ids = []
            j = 0
            while True:
                geom_name = f"obs_{i}_g{j}"
                geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                )
                if geom_id < 0:
                    break
                geom_ids.append(geom_id)
                j += 1

            self._slots.append(_ObjectSlot(body_id, geom_ids))
            i += 1

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def apply(self, placed_objects: list[PlacedObject]):
        """Write placed objects into model slots. Hide unused slots.

        Call mj_forward() after this to update kinematics.
        """
        if len(placed_objects) > self.max_objects:
            raise ValueError(
                f"Too many objects ({len(placed_objects)}) for {self.max_objects} slots"
            )

        for i, slot in enumerate(self._slots):
            if i < len(placed_objects):
                obj = placed_objects[i]
                concept_mod = concepts.get(obj.concept)
                prims = concept_mod.generate(obj.params)
                self._write_object(slot, obj.pos, obj.rotation, prims)
            else:
                self._hide_object(slot)

    def clear(self):
        """Hide all obstacle slots."""
        for slot in self._slots:
            self._hide_object(slot)

    def random_scene(
        self,
        n_objects: int | None = None,
        min_objects: int = 1,
        max_objects: int | None = None,
        arena_size: float = 3.5,
        min_spacing: float = 0.6,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[PlacedObject]:
        """Sample a random scene layout.

        Returns a list of PlacedObject that can be passed to apply().
        Objects are placed with rejection sampling to avoid overlaps.

        Args:
            n_objects: Exact count (overrides min/max). None = random.
            min_objects: Minimum object count when n_objects is None.
            max_objects: Maximum object count (defaults to self.max_objects).
            arena_size: Half-extent of placement area (meters from origin).
            min_spacing: Minimum distance between object centers.
            seed: Integer seed for reproducibility. Overrides rng if both given.
            rng: Numpy random generator (for reproducibility).
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        elif rng is None:
            rng = np.random.default_rng()

        if max_objects is None:
            max_objects = self.max_objects
        max_objects = min(max_objects, self.max_objects)

        if n_objects is None:
            n_objects = rng.integers(min_objects, max_objects + 1)
        n_objects = min(n_objects, max_objects)

        available = concepts.list_concepts()
        if not available:
            return []

        placed: list[PlacedObject] = []
        positions: list[np.ndarray] = []

        for _ in range(n_objects):
            # Pick a random concept
            concept_name = available[rng.integers(len(available))]
            concept_mod = concepts.get(concept_name)

            # Use default params (with random color variation later)
            params = concept_mod.Params()

            # Rejection-sample a position that doesn't overlap
            pos = None
            for _attempt in range(50):
                candidate = rng.uniform(-arena_size, arena_size, size=2)
                if all(np.linalg.norm(candidate - p) >= min_spacing for p in positions):
                    pos = candidate
                    break

            if pos is None:
                break  # couldn't place — stop adding objects

            rotation = rng.uniform(0, 2 * np.pi)
            positions.append(pos)
            placed.append(
                PlacedObject(
                    concept=concept_name,
                    params=params,
                    pos=(float(pos[0]), float(pos[1])),
                    rotation=rotation,
                )
            )

        return placed

    # -------------------------------------------------------------------
    # Internal: write / hide geom slots
    # -------------------------------------------------------------------

    def _write_object(
        self,
        slot: _ObjectSlot,
        pos: tuple[float, float],
        rotation: float,
        prims: tuple[Prim, ...],
    ):
        """Write concept primitives into a slot's geoms."""
        if len(prims) > len(slot.geom_ids):
            raise ValueError(
                f"Concept has {len(prims)} prims but slot only has "
                f"{len(slot.geom_ids)} geom slots"
            )

        # Position the body at (x, y, 0)
        self.model.body_pos[slot.body_id] = [pos[0], pos[1], 0.0]

        # Rotation around Z — set body quaternion
        if rotation != 0.0:
            self.model.body_quat[slot.body_id] = euler_to_quat(0, 0, rotation)
        else:
            self.model.body_quat[slot.body_id] = [1, 0, 0, 0]

        # Write each primitive to a geom slot
        for j, geom_id in enumerate(slot.geom_ids):
            if j < len(prims):
                prim = prims[j]
                self.model.geom_type[geom_id] = int(prim.geom_type)
                self.model.geom_size[geom_id] = prim.size
                self.model.geom_pos[geom_id] = prim.pos
                self.model.geom_rgba[geom_id] = prim.rgba
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 1

                # Geom-local rotation
                if prim.euler != (0.0, 0.0, 0.0):
                    self.model.geom_quat[geom_id] = euler_to_quat(*prim.euler)
                else:
                    self.model.geom_quat[geom_id] = [1, 0, 0, 0]
            else:
                # Hide unused geom slots
                self._hide_geom(geom_id)

    def _hide_object(self, slot: _ObjectSlot):
        """Hide all geoms in an object slot."""
        self.model.body_pos[slot.body_id] = [0, 0, 0]
        self.model.body_quat[slot.body_id] = [1, 0, 0, 0]
        for geom_id in slot.geom_ids:
            self._hide_geom(geom_id)

    def _hide_geom(self, geom_id: int):
        """Make a geom invisible and non-colliding."""
        self.model.geom_size[geom_id] = [0.001, 0.001, 0.001]
        self.model.geom_rgba[geom_id] = [0, 0, 0, 0]
        self.model.geom_contype[geom_id] = 0
        self.model.geom_conaffinity[geom_id] = 0


# ---------------------------------------------------------------------------
# Scene description & identity
# ---------------------------------------------------------------------------

# Param fields that represent key spatial dimensions (for describe_scene)
_DIMENSION_FIELDS = {
    "width",
    "depth",
    "height",
    "seat_width",
    "seat_depth",
    "seat_height",
    "n_shelves",
}


def scene_id(seed: int) -> str:
    """Short hex identifier for a scene seed (6 chars)."""
    return f"{seed & 0xFFFFFF:06x}"


def describe_object(obj: PlacedObject) -> str:
    """One-line description of a placed object."""
    x, y = obj.pos
    rot_deg = np.degrees(obj.rotation)

    # Extract key dimensions from params
    dims = []
    for f in dataclasses.fields(obj.params):
        if f.name in _DIMENSION_FIELDS:
            val = getattr(obj.params, f.name)
            if isinstance(val, float):
                dims.append(f"{f.name}={val:.2f}")
            else:
                dims.append(f"{f.name}={val}")

    dim_str = f"  ({', '.join(dims)})" if dims else ""
    return f"{obj.concept} at ({x:+.2f}, {y:+.2f}) rot {rot_deg:.0f}\u00b0{dim_str}"


def describe_scene(
    placed_objects: list[PlacedObject],
    seed: int | None = None,
) -> str:
    """Multi-line textual description of a full scene.

    Example output:
        Scene #a3f2b1 (seed=2847391)  3 objects
          [0] table at (+2.10, -1.30) rot 45°  (width=1.00, depth=0.60, height=0.75)
          [1] chair at (-0.80, +1.50) rot 180°  (seat_width=0.45, ...)
          [2] shelf at (+1.20, +0.40) rot 271°  (width=0.80, depth=0.30, height=1.20)
    """
    lines = []

    # Header
    if seed is not None:
        sid = scene_id(seed)
        lines.append(f"Scene #{sid} (seed={seed})  {len(placed_objects)} objects")
    else:
        lines.append(f"Scene  {len(placed_objects)} objects")

    # Each object
    for i, obj in enumerate(placed_objects):
        lines.append(f"  [{i}] {describe_object(obj)}")

    return "\n".join(lines)
