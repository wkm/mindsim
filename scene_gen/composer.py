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
from scene_gen.primitives import Placement, Prim, euler_to_quat, footprint, obb_overlaps

# ---------------------------------------------------------------------------
# Placement sampling helpers
# ---------------------------------------------------------------------------

# Wall offset: how far from the arena edge the object center sits
_WALL_OFFSET = 0.15
# Inner zone fraction for CENTER placement (±65% of arena)
_CENTER_FRAC = 0.65
# Corner offset from arena edge
_CORNER_OFFSET = 0.4
# Rotation jitter: ±3° in radians
_ROT_JITTER = np.radians(3.0)

# The 4 cardinal rotations (facing into room from N/E/S/W walls)
# Wall at +Y → face -Y (π), wall at -Y → face +Y (0),
# wall at +X → face -X (π/2), wall at -X → face +X (-π/2)
_WALL_ROTS = {
    "N": np.pi,        # back against +Y wall, face into room
    "S": 0.0,          # back against -Y wall
    "E": np.pi / 2,    # back against +X wall
    "W": -np.pi / 2,   # back against -X wall
}


def _snap_rotation(rng: np.random.Generator) -> float:
    """Random axis-aligned rotation (0/90/180/270°) + small jitter."""
    base = rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2])
    return float(base + rng.uniform(-_ROT_JITTER, _ROT_JITTER))


def _sample_position(
    rng: np.random.Generator,
    placement: Placement,
    arena_size: float,
    hx: float,
    hy: float,
) -> tuple[float, float, float]:
    """Sample (cx, cy, rotation) according to placement type.

    Args:
        rng: Random generator
        placement: WALL, CENTER, or CORNER
        arena_size: Half-extent of the arena
        hx, hy: Object half-extents (used to keep objects inside arena)

    Returns:
        (cx, cy, rotation) tuple
    """
    if placement == Placement.WALL:
        return _sample_wall_pos(rng, arena_size, hx, hy)
    elif placement == Placement.CORNER:
        return _sample_corner_pos(rng, arena_size, hx, hy)
    else:  # CENTER
        return _sample_center_pos(rng, arena_size)


def _sample_wall_pos(
    rng: np.random.Generator,
    arena_size: float,
    hx: float,
    hy: float,
) -> tuple[float, float, float]:
    """Place against a random wall, back facing wall, slide along wall."""
    wall = rng.choice(["N", "S", "E", "W"])
    rotation = float(_WALL_ROTS[wall] + rng.uniform(-_ROT_JITTER, _ROT_JITTER))

    # The "depth" extent that faces the wall depends on rotation:
    # for N/S walls the object's local Y faces the wall → use hy
    # for E/W walls the object's local Y faces the wall → use hy
    # (rotation already handles orientation, hy is the back-facing extent)
    depth = max(hx, hy)
    slide_extent = arena_size - depth - 0.1  # keep off adjacent walls

    if wall == "N":
        cy = float(arena_size - _WALL_OFFSET - depth)
        cx = float(rng.uniform(-slide_extent, slide_extent))
    elif wall == "S":
        cy = float(-arena_size + _WALL_OFFSET + depth)
        cx = float(rng.uniform(-slide_extent, slide_extent))
    elif wall == "E":
        cx = float(arena_size - _WALL_OFFSET - depth)
        cy = float(rng.uniform(-slide_extent, slide_extent))
    else:  # W
        cx = float(-arena_size + _WALL_OFFSET + depth)
        cy = float(rng.uniform(-slide_extent, slide_extent))

    return (cx, cy, rotation)


def _sample_center_pos(
    rng: np.random.Generator,
    arena_size: float,
) -> tuple[float, float, float]:
    """Place in the inner zone with axis-aligned rotation."""
    inner = arena_size * _CENTER_FRAC
    cx = float(rng.uniform(-inner, inner))
    cy = float(rng.uniform(-inner, inner))
    rotation = _snap_rotation(rng)
    return (cx, cy, rotation)


def _sample_corner_pos(
    rng: np.random.Generator,
    arena_size: float,
    hx: float,
    hy: float,
) -> tuple[float, float, float]:
    """Place near one of 4 corners with axis-aligned rotation."""
    corner_base = arena_size - _CORNER_OFFSET
    # Pick a random corner
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    sx, sy = signs[rng.integers(4)]
    # Small random offset from the corner
    jitter = 0.3
    cx = float(sx * corner_base + rng.uniform(-jitter, jitter))
    cy = float(sy * corner_base + rng.uniform(-jitter, jitter))
    # Clamp to stay inside arena
    limit = arena_size - max(hx, hy) - 0.05
    cx = float(np.clip(cx, -limit, limit))
    cy = float(np.clip(cy, -limit, limit))
    rotation = _snap_rotation(rng)
    return (cx, cy, rotation)


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
        margin: float = 0.1,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[PlacedObject]:
        """Sample a random scene layout with placement-aware positioning.

        Returns a list of PlacedObject that can be passed to apply().

        Objects are placed via rejection sampling.  Overlap is checked with
        rotated bounding boxes (SAT), not just center distance.  Objects on
        different *layers* are allowed to overlap — e.g. a table can sit on
        a rug.  Layer is determined by the concept module: modules with
        ``GROUND_COVER = True`` are on the ground-cover layer; everything
        else is furniture.

        Position sampling respects each concept's ``PLACEMENT`` attribute:
        - WALL: placed near a wall edge, facing into the room
        - CENTER: placed in the inner zone with axis-aligned rotation
        - CORNER: placed near a room corner with axis-aligned rotation

        Objects are sorted by placement priority (ground_cover first, then
        WALL, CORNER, CENTER) so the most constrained objects get first pick.

        Args:
            n_objects: Exact count (overrides min/max). None = random.
            min_objects: Minimum object count when n_objects is None.
            max_objects: Maximum object count (defaults to self.max_objects).
            arena_size: Half-extent of placement area (meters from origin).
            margin: Extra padding (meters) added to each bounding box edge.
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

        # Pick concepts first, then sort by placement priority
        picks: list[tuple[str, object, bool, Placement]] = []
        for _ in range(n_objects):
            concept_name = available[rng.integers(len(available))]
            concept_mod = concepts.get(concept_name)
            params = concept_mod.Params()
            is_ground = getattr(concept_mod, "GROUND_COVER", False)
            placement = getattr(concept_mod, "PLACEMENT", Placement.CENTER)
            picks.append((concept_name, params, is_ground, placement))

        # Sort: ground_cover first, then WALL (most constrained) → CORNER → CENTER
        _priority = {Placement.WALL: 0, Placement.CORNER: 1, Placement.CENTER: 2}
        picks.sort(key=lambda p: (not p[2], _priority.get(p[3], 2)))

        placed: list[PlacedObject] = []
        # Parallel tracking arrays for fast collision checks
        _cx: list[float] = []  # center x
        _cy: list[float] = []  # center y
        _hx: list[float] = []  # half-extent x
        _hy: list[float] = []  # half-extent y
        _rot: list[float] = []  # rotation
        _ground: list[bool] = []  # is ground-cover layer?

        for concept_name, params, is_ground, placement in picks:
            concept_mod = concepts.get(concept_name)

            # Compute footprint from generated prims
            prims = concept_mod.generate(params)
            hx, hy = footprint(prims)

            # Rejection-sample a position with OBB overlap check
            pos = None
            for _attempt in range(50):
                cx, cy, rotation = _sample_position(
                    rng, placement, arena_size, hx, hy,
                )

                # Only check against objects on the same layer
                ok = True
                for j in range(len(placed)):
                    if _ground[j] != is_ground:
                        continue  # different layers — skip
                    if obb_overlaps(
                        cx, cy, hx, hy, rotation,
                        _cx[j], _cy[j], _hx[j], _hy[j], _rot[j],
                        margin=margin,
                    ):
                        ok = False
                        break

                if ok:
                    pos = (cx, cy)
                    break

            if pos is None:
                continue  # couldn't place — skip, try next object

            _cx.append(pos[0])
            _cy.append(pos[1])
            _hx.append(hx)
            _hy.append(hy)
            _rot.append(rotation)
            _ground.append(is_ground)
            placed.append(
                PlacedObject(
                    concept=concept_name,
                    params=params,
                    pos=pos,
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
