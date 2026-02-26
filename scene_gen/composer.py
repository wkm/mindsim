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
    composer.apply(scene)  # writes slots + mj_forward + geom sync

    # Option B: explicit placement
    from scene_gen.concepts import table
    scene = [
        PlacedObject("table", table.Params(width=1.2), pos=(1, 0), rotation=0.3),
    ]
    composer.apply(scene)  # ready for rendering after this call
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import mujoco
import numpy as np

from scene_gen import concepts
from scene_gen.archetypes import Archetype
from scene_gen.groupings import GROUPINGS
from scene_gen.primitives import Placement, Prim, euler_to_quat, footprint, obb_overlaps
from scene_gen.room import (
    WALL_THICKNESS,
    Room,
    RoomConfig,
    prepare_room,
    random_room_config,
)

# ---------------------------------------------------------------------------
# Placement sampling helpers
# ---------------------------------------------------------------------------

# Wall offset: how far from the arena edge the object center sits
_WALL_OFFSET = 0.15
# Inner zone fraction for CENTER placement (±65% of arena)
_CENTER_FRAC = 0.65
# Corner offset from arena edge
_CORNER_OFFSET = 0.4
# Grid snap size for major object placement
_GRID = 0.25

# The 4 cardinal rotations (facing into room from N/E/S/W walls)
# Wall at +Y → face -Y (π), wall at -Y → face +Y (0),
# wall at +X → face -X (π/2), wall at -X → face +X (-π/2)
_WALL_ROTS = {
    "N": np.pi,  # back against +Y wall, face into room
    "S": 0.0,  # back against -Y wall
    "E": np.pi / 2,  # back against +X wall
    "W": -np.pi / 2,  # back against -X wall
}


# Margin to keep furniture from touching walls
_WALL_MARGIN = 0.05


def _snap_to_grid(value: float, grid: float) -> float:
    """Snap a scalar value to the nearest grid step."""
    if grid <= 0:
        return value
    return float(round(value / grid) * grid)


def _build_wall_obbs(
    room_config: RoomConfig,
) -> list[tuple[float, float, float, float, float]]:
    """Build a list of wall segment OBBs from a room config.

    Each OBB is (cx, cy, hx, hy, rotation). These are used for
    collision checking against placed objects via obb_overlaps().

    Interior walls are split into two segments around the doorway gap.
    Only the solid wall segments are returned (not the doorway).
    """
    obbs: list[tuple[float, float, float, float, float]] = []
    he = room_config.half_extent

    for iwall in room_config.interior_walls:
        door_half = iwall.door_width / 2

        if iwall.axis == "x":
            # Wall runs along X axis at y = offset
            # Segment 1: from -he to (door_pos - door_half)
            seg1_start = -he
            seg1_end = iwall.door_pos - door_half
            if seg1_end > seg1_start + 0.05:
                seg1_hx = (seg1_end - seg1_start) / 2
                seg1_cx = (seg1_start + seg1_end) / 2
                obbs.append((seg1_cx, iwall.offset, seg1_hx, WALL_THICKNESS, 0.0))

            # Segment 2: from (door_pos + door_half) to +he
            seg2_start = iwall.door_pos + door_half
            seg2_end = he
            if seg2_end > seg2_start + 0.05:
                seg2_hx = (seg2_end - seg2_start) / 2
                seg2_cx = (seg2_start + seg2_end) / 2
                obbs.append((seg2_cx, iwall.offset, seg2_hx, WALL_THICKNESS, 0.0))

        else:  # axis == "y"
            # Wall runs along Y axis at x = offset
            seg1_start = -he
            seg1_end = iwall.door_pos - door_half
            if seg1_end > seg1_start + 0.05:
                seg1_hy = (seg1_end - seg1_start) / 2
                seg1_cy = (seg1_start + seg1_end) / 2
                obbs.append((iwall.offset, seg1_cy, WALL_THICKNESS, seg1_hy, 0.0))

            seg2_start = iwall.door_pos + door_half
            seg2_end = he
            if seg2_end > seg2_start + 0.05:
                seg2_hy = (seg2_end - seg2_start) / 2
                seg2_cy = (seg2_start + seg2_end) / 2
                obbs.append((iwall.offset, seg2_cy, WALL_THICKNESS, seg2_hy, 0.0))

    return obbs


def _inside_exterior_walls(
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    rot: float,
    half_extent: float,
) -> bool:
    """Check that a rotated bounding box is fully inside the exterior walls.

    The max reach of a rotated box along each axis:
        reach_x = hx * |cos(rot)| + hy * |sin(rot)|
        reach_y = hx * |sin(rot)| + hy * |cos(rot)|

    The object must stay inside half_extent - WALL_THICKNESS - margin.
    """
    limit = half_extent - WALL_THICKNESS - _WALL_MARGIN
    cos_r = abs(np.cos(rot))
    sin_r = abs(np.sin(rot))
    reach_x = hx * cos_r + hy * sin_r
    reach_y = hx * sin_r + hy * cos_r
    return (abs(cx) + reach_x) < limit and (abs(cy) + reach_y) < limit


def _snap_rotation(rng: np.random.Generator) -> float:
    """Axis-aligned rotation (0/90/180/270°)."""
    base = rng.choice([0.0, np.pi / 2, np.pi, -np.pi / 2])
    return float(base)


def _sample_position(
    rng: np.random.Generator,
    placement: Placement,
    arena_size: float,
    hx: float,
    hy: float,
    wall_counts: dict[str, int] | None = None,
) -> tuple[float, float, float]:
    """Sample (cx, cy, rotation) according to placement type.

    Args:
        rng: Random generator
        placement: WALL, CENTER, or CORNER
        arena_size: Half-extent of the arena
        hx, hy: Object half-extents (used to keep objects inside arena)
        wall_counts: Track wall usage for even distribution (WALL only)

    Returns:
        (cx, cy, rotation) tuple
    """
    if placement == Placement.WALL:
        return _sample_wall_pos(rng, arena_size, hx, hy, wall_counts)
    elif placement == Placement.CORNER:
        return _sample_corner_pos(rng, arena_size, hx, hy)
    else:  # CENTER
        return _sample_center_pos(rng, arena_size)


def _sample_wall_pos(
    rng: np.random.Generator,
    arena_size: float,
    hx: float,
    hy: float,
    wall_counts: dict[str, int] | None = None,
) -> tuple[float, float, float]:
    """Place against a wall, back facing wall, slide along wall.

    When wall_counts is provided, prefers the least-used wall to
    distribute furniture around the perimeter evenly.
    """
    if wall_counts is not None:
        # Pick the least-used wall(s), break ties randomly
        min_count = min(wall_counts.values())
        candidates = [w for w, c in wall_counts.items() if c == min_count]
        wall = candidates[rng.integers(len(candidates))]
        wall_counts[wall] += 1
    else:
        wall = rng.choice(["N", "S", "E", "W"])

    rotation = float(_WALL_ROTS[wall])

    # The "depth" extent that faces the wall depends on rotation:
    # for N/S walls the object's local Y faces the wall → use hy
    # for E/W walls the object's local Y faces the wall → use hy
    # (rotation already handles orientation, hy is the back-facing extent)
    depth = max(hx, hy)
    slide_extent = arena_size - depth - 0.1  # keep off adjacent walls

    if wall == "N":
        cy = float(arena_size - _WALL_OFFSET - depth)
        cx = float(rng.uniform(-slide_extent, slide_extent))
        cx = _snap_to_grid(cx, _GRID)
        cx = float(np.clip(cx, -slide_extent, slide_extent))
    elif wall == "S":
        cy = float(-arena_size + _WALL_OFFSET + depth)
        cx = float(rng.uniform(-slide_extent, slide_extent))
        cx = _snap_to_grid(cx, _GRID)
        cx = float(np.clip(cx, -slide_extent, slide_extent))
    elif wall == "E":
        cx = float(arena_size - _WALL_OFFSET - depth)
        cy = float(rng.uniform(-slide_extent, slide_extent))
        cy = _snap_to_grid(cy, _GRID)
        cy = float(np.clip(cy, -slide_extent, slide_extent))
    else:  # W
        cx = float(-arena_size + _WALL_OFFSET + depth)
        cy = float(rng.uniform(-slide_extent, slide_extent))
        cy = _snap_to_grid(cy, _GRID)
        cy = float(np.clip(cy, -slide_extent, slide_extent))

    return (cx, cy, rotation)


def _sample_center_pos(
    rng: np.random.Generator,
    arena_size: float,
) -> tuple[float, float, float]:
    """Place in the inner zone with axis-aligned rotation."""
    inner = arena_size * _CENTER_FRAC
    cx = float(rng.uniform(-inner, inner))
    cy = float(rng.uniform(-inner, inner))
    cx = _snap_to_grid(cx, _GRID)
    cy = _snap_to_grid(cy, _GRID)
    cx = float(np.clip(cx, -inner, inner))
    cy = float(np.clip(cy, -inner, inner))
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
    cx = _snap_to_grid(cx, _GRID)
    cy = _snap_to_grid(cy, _GRID)
    cx = float(np.clip(cx, -limit, limit))
    cy = float(np.clip(cy, -limit, limit))
    rotation = _snap_rotation(rng)
    return (cx, cy, rotation)


# ---------------------------------------------------------------------------
# Defaults for obstacle slot allocation
# ---------------------------------------------------------------------------

DEFAULT_MAX_OBJECTS = 16
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
class _GeomSlot:
    """A single body+geom pair. One body per geom eliminates the need for
    manual geom_xpos patching — mj_kinematics reliably propagates
    model.body_pos → data.xpos → data.geom_xpos."""

    body_id: int
    geom_id: int


@dataclass
class _ObjectSlot:
    """A logical object: a group of geom slots that form one piece of furniture."""

    geom_slots: list[_GeomSlot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


class SceneComposer:
    """Writes procedural scene geometry into pre-allocated MuJoCo model slots."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._slots: list[_ObjectSlot] = []
        self._room = Room(model, data)
        self._last_room_config: RoomConfig | None = None
        self._discover_slots()

    @staticmethod
    def prepare_spec(
        spec,
        max_objects: int = DEFAULT_MAX_OBJECTS,
        geoms_per_object: int = DEFAULT_GEOMS_PER_OBJECT,
        walls: bool = True,
    ):
        """Add obstacle slots (and optionally wall slots) to an MjSpec.

        Creates one body per geom (not one body per object). This ensures
        mj_kinematics correctly computes geom world positions from
        model.body_pos, which it reliably reads at runtime. The alternative
        (one body + N child geoms with modified model.geom_pos) breaks in
        the passive viewer because mj_step overwrites data.geom_xpos.

        When walls=True (default), also injects room wall slots for
        variable-size rooms and interior walls.
        """
        if walls:
            prepare_room(spec)

        for i in range(max_objects):
            for j in range(geoms_per_object):
                body = spec.worldbody.add_body()
                body.name = f"obs_{i}_g{j}"
                body.pos = [0, 0, 0]
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
        return len(self._slots[0].geom_slots)

    def _discover_slots(self):
        """Find obs_N_gM body+geom pairs and group them into object slots."""
        i = 0
        while True:
            # Probe for the first geom slot of object i
            probe = f"obs_{i}_g0"
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, probe) < 0:
                break

            geom_slots = []
            j = 0
            while True:
                name = f"obs_{i}_g{j}"
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id < 0:
                    break
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                geom_slots.append(_GeomSlot(body_id, geom_id))
                j += 1

            self._slots.append(_ObjectSlot(geom_slots))
            i += 1

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def apply(
        self,
        placed_objects: list[PlacedObject],
        room_config: RoomConfig | None = None,
    ):
        """Write placed objects into model slots, then run forward kinematics.

        This is the single code path for scene application. After this call,
        the model is ready for rendering (viewer or offscreen). No manual
        geom_xpos patching needed — each geom has its own body, so
        mj_kinematics correctly computes world positions from body_pos.

        Args:
            placed_objects: Objects to place in the scene.
            room_config: Optional room layout (walls + size). When provided,
                walls are positioned before furniture. When None, walls are
                hidden (backward compatible with callers that don't use rooms).
        """
        if len(placed_objects) > self.max_objects:
            raise ValueError(
                f"Too many objects ({len(placed_objects)}) for {self.max_objects} slots"
            )

        # Position room walls: use explicit config, or fall back to what
        # random_scene() stored, so callers don't need to pass it twice.
        effective_room = room_config or self._last_room_config
        if effective_room is not None:
            self._room.apply(effective_room)

        for i, slot in enumerate(self._slots):
            if i < len(placed_objects):
                obj = placed_objects[i]
                concept_mod = concepts.get(obj.concept)
                prims = concept_mod.generate(obj.params)
                self._write_object(slot, obj.pos, obj.rotation, prims)
            else:
                self._hide_object(slot)

        mujoco.mj_forward(self.model, self.data)

    def clear(self):
        """Hide all obstacle slots."""
        for slot in self._slots:
            self._hide_object(slot)

    def random_scene(
        self,
        n_objects: int | None = None,
        min_objects: int = 1,
        max_objects: int | None = None,
        archetype: str | Archetype | None = None,
        arena_size: float | None = None,
        margin: float = 0.1,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        room_config: RoomConfig | None = None,
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
            archetype: Room archetype (name or Archetype instance). When given,
                concepts are drawn from the archetype's pool instead of uniform
                random. A random archetype is chosen if set to "random".
            arena_size: Half-extent of placement area (meters from origin).
                When None, derived from room_config or defaults to 3.5.
            margin: Extra padding (meters) added to each bounding box edge.
            seed: Integer seed for reproducibility. Overrides rng if both given.
            rng: Numpy random generator (for reproducibility).
            room_config: Room layout (walls + size). When provided, arena_size
                is derived from the config. Stored on the returned scene for
                apply() to use.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        elif rng is None:
            rng = np.random.default_rng()

        # Generate room config if not provided (variable room size)
        if room_config is None and self._room.has_slots:
            room_config = random_room_config(rng)

        # Derive arena_size from room config (leave margin inside walls)
        if arena_size is None:
            if room_config is not None:
                arena_size = room_config.half_extent - 0.15
            else:
                arena_size = 3.5

        # Build wall OBBs for collision checking
        _wall_obbs: list[tuple[float, float, float, float, float]] = []
        _room_half_extent: float | None = None
        if room_config is not None:
            _wall_obbs = _build_wall_obbs(room_config)
            _room_half_extent = room_config.half_extent

        if max_objects is None:
            max_objects = self.max_objects
        max_objects = min(max_objects, self.max_objects)

        # Resolve archetype
        arch: Archetype | None = None
        if archetype is not None:
            from scene_gen.archetypes import ARCHETYPES

            if isinstance(archetype, str):
                if archetype == "random":
                    names = list(ARCHETYPES.keys())
                    archetype = names[rng.integers(len(names))]
                arch = ARCHETYPES[archetype]
            else:
                arch = archetype

        # Pick concept names — from archetype or uniform random
        allowed_groupings: set[str] = set()
        if arch is not None:
            allowed_groupings = set(getattr(arch, "groupings", ()))
            ensure_concepts: tuple[str, ...] = ()
            if allowed_groupings:
                ensure_concepts = tuple(
                    GROUPINGS[g].anchor for g in allowed_groupings if g in GROUPINGS
                )
            concept_names = arch.sample(
                rng,
                max_objects=max_objects,
                ensure_concepts=ensure_concepts,
            )
        else:
            available = concepts.list_concepts()
            if not available:
                return []
            if n_objects is None:
                n_objects = int(rng.integers(min_objects, max_objects + 1))
            n_objects = min(n_objects, max_objects)
            concept_names = [
                available[rng.integers(len(available))] for _ in range(n_objects)
            ]

        # Build index: which concepts are anchors for groupings?
        # Only use each grouping once per scene to avoid duplicate clusters.
        _anchor_to_grouping: dict[str, str] = {}
        if arch is not None and allowed_groupings:
            for gname in allowed_groupings:
                grouping = GROUPINGS.get(gname)
                if grouping is None:
                    continue
                _anchor_to_grouping.setdefault(grouping.anchor, gname)
        else:
            for gname, grouping in GROUPINGS.items():
                _anchor_to_grouping.setdefault(grouping.anchor, gname)
        _used_groupings: set[str] = set()

        # Resolve each concept name to (name, params, is_ground, placement).
        # When a concept has VARIATIONS, randomly pick one for visual diversity.
        picks: list[tuple[str, object, bool, Placement]] = []
        for concept_name in concept_names:
            concept_mod = concepts.get(concept_name)
            variations = getattr(concept_mod, "VARIATIONS", None)
            if variations:
                var_list = list(variations.values())
                params = var_list[rng.integers(len(var_list))]
            else:
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
        # Track wall usage for even distribution
        _wall_counts: dict[str, int] = {"N": 0, "S": 0, "E": 0, "W": 0}

        def _try_place(
            c_name: str,
            c_params: object,
            c_ground: bool,
            c_placement: Placement,
            fixed_pos: tuple[float, float] | None = None,
            fixed_rot: float | None = None,
            overlap_margin: float | None = None,
        ) -> bool:
            """Try to place one object. Returns True if placed successfully.

            overlap_margin overrides the default margin for collision checks.
            Satellites use a smaller margin since they're meant to sit close
            to their anchor.
            """
            if len(placed) >= max_objects:
                return False

            m_eff = overlap_margin if overlap_margin is not None else margin

            c_mod = concepts.get(c_name)
            c_prims = c_mod.generate(c_params)
            c_hx, c_hy = footprint(c_prims)

            def _check_candidate(cx: float, cy: float, rot: float) -> bool:
                """Check a candidate position against objects and walls."""
                # Object-vs-object overlap (same layer only)
                for j in range(len(placed)):
                    if _ground[j] != c_ground:
                        continue
                    if obb_overlaps(
                        cx,
                        cy,
                        c_hx,
                        c_hy,
                        rot,
                        _cx[j],
                        _cy[j],
                        _hx[j],
                        _hy[j],
                        _rot[j],
                        margin=m_eff,
                    ):
                        return False

                # Exterior wall check: rotated bbox must be inside walls
                if _room_half_extent is not None:
                    if not _inside_exterior_walls(
                        cx, cy, c_hx, c_hy, rot, _room_half_extent
                    ):
                        return False
                else:
                    # Fallback: simple arena bounds check
                    limit = arena_size - max(c_hx, c_hy) - 0.05
                    if abs(cx) > limit or abs(cy) > limit:
                        return False

                # Interior wall segment checks
                for wcx, wcy, whx, why, wrot in _wall_obbs:
                    if obb_overlaps(
                        cx,
                        cy,
                        c_hx,
                        c_hy,
                        rot,
                        wcx,
                        wcy,
                        whx,
                        why,
                        wrot,
                        margin=_WALL_MARGIN,
                    ):
                        return False

                return True

            if fixed_pos is not None:
                # Place at a specific position (for satellites)
                cx, cy = fixed_pos
                rot = fixed_rot if fixed_rot is not None else 0.0
                if not _check_candidate(cx, cy, rot):
                    return False
                pos = (cx, cy)
                rotation = rot
            else:
                # Rejection-sample a position
                pos = None
                for _attempt in range(50):
                    cx, cy, rotation = _sample_position(
                        rng,
                        c_placement,
                        arena_size,
                        c_hx,
                        c_hy,
                        wall_counts=_wall_counts,
                    )
                    if _check_candidate(cx, cy, rotation):
                        pos = (cx, cy)
                        break
                if pos is None:
                    return False

            _cx.append(pos[0])
            _cy.append(pos[1])
            _hx.append(c_hx)
            _hy.append(c_hy)
            _rot.append(rotation)
            _ground.append(c_ground)
            placed.append(
                PlacedObject(
                    concept=c_name,
                    params=c_params,
                    pos=pos,
                    rotation=rotation,
                )
            )
            return True

        for concept_name, params, is_ground, placement in picks:
            if not _try_place(concept_name, params, is_ground, placement):
                continue

            # If this concept is a grouping anchor, try to place satellites
            gname = _anchor_to_grouping.get(concept_name)
            if gname and gname not in _used_groupings and arch is not None:
                _used_groupings.add(gname)
                grouping = GROUPINGS[gname]
                anchor = placed[-1]
                anchor_cx, anchor_cy = anchor.pos
                anchor_rot = anchor.rotation
                cos_r, sin_r = np.cos(anchor_rot), np.sin(anchor_rot)

                for sat_name, dx, dy, drot in grouping.resolve(rng):
                    # Transform local offset to world coordinates
                    wx = anchor_cx + dx * cos_r - dy * sin_r
                    wy = anchor_cy + dx * sin_r + dy * cos_r
                    wrot = anchor_rot + drot

                    sat_mod = concepts.get(sat_name)
                    sat_variations = getattr(sat_mod, "VARIATIONS", None)
                    if sat_variations:
                        var_list = list(sat_variations.values())
                        sat_params = var_list[rng.integers(len(var_list))]
                    else:
                        sat_params = sat_mod.Params()
                    sat_ground = getattr(sat_mod, "GROUND_COVER", False)
                    sat_placement = getattr(sat_mod, "PLACEMENT", Placement.CENTER)

                    _try_place(
                        sat_name,
                        sat_params,
                        sat_ground,
                        sat_placement,
                        fixed_pos=(wx, wy),
                        fixed_rot=wrot,
                        overlap_margin=0.02,
                    )

        # Store room config so apply() can use it without explicit passing
        self._last_room_config = room_config

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
        """Write concept primitives into a slot's geom bodies."""
        if len(prims) > len(slot.geom_slots):
            raise ValueError(
                f"Concept has {len(prims)} prims but slot only has "
                f"{len(slot.geom_slots)} geom slots"
            )

        # Precompute rotation matrix for the object
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        for j, gs in enumerate(slot.geom_slots):
            if j < len(prims):
                prim = prims[j]

                # Compute world position: object origin + rotated prim offset
                px, py, pz = prim.pos
                wx = pos[0] + cos_r * px - sin_r * py
                wy = pos[1] + sin_r * px + cos_r * py
                wz = pz

                self.model.body_pos[gs.body_id] = [wx, wy, wz]

                # Body quat = object rotation (Z) composed with prim rotation
                if prim.euler != (0.0, 0.0, 0.0):
                    obj_quat = euler_to_quat(0, 0, rotation)
                    prim_quat = euler_to_quat(*prim.euler)
                    combined = np.zeros(4)
                    mujoco.mju_mulQuat(combined, obj_quat, prim_quat)
                    self.model.body_quat[gs.body_id] = combined
                elif rotation != 0.0:
                    self.model.body_quat[gs.body_id] = euler_to_quat(0, 0, rotation)
                else:
                    self.model.body_quat[gs.body_id] = [1, 0, 0, 0]

                # Geom stays at body origin — body_pos IS the world position
                self.model.geom_type[gs.geom_id] = int(prim.geom_type)
                self.model.geom_size[gs.geom_id] = prim.size
                self.model.geom_pos[gs.geom_id] = [0, 0, 0]
                self.model.geom_quat[gs.geom_id] = [1, 0, 0, 0]
                self.model.geom_rgba[gs.geom_id] = prim.rgba
                self.model.geom_contype[gs.geom_id] = 1
                self.model.geom_conaffinity[gs.geom_id] = 1
            else:
                self._hide_geom_slot(gs)

    def _hide_object(self, slot: _ObjectSlot):
        """Hide all geom slots in an object slot."""
        for gs in slot.geom_slots:
            self._hide_geom_slot(gs)

    def _hide_geom_slot(self, gs: _GeomSlot):
        """Make a geom slot invisible and non-colliding."""
        self.model.body_pos[gs.body_id] = [0, 0, 0]
        self.model.body_quat[gs.body_id] = [1, 0, 0, 0]
        self.model.geom_size[gs.geom_id] = [0.001, 0.001, 0.001]
        self.model.geom_rgba[gs.geom_id] = [0, 0, 0, 0]
        self.model.geom_contype[gs.geom_id] = 0
        self.model.geom_conaffinity[gs.geom_id] = 0


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
        header = f"Scene #{sid} (seed={seed})  {len(placed_objects)} objects"
    else:
        header = f"Scene  {len(placed_objects)} objects"
    lines.append(header)

    # Each object
    for i, obj in enumerate(placed_objects):
        lines.append(f"  [{i}] {describe_object(obj)}")

    return "\n".join(lines)
