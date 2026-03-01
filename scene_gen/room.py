"""Room layout — variable room size and interior walls.

Walls are injected into the MjSpec at build time (like obstacle slots) and
positioned at runtime. This means the same compiled model can represent
different room sizes and wall configurations by writing to model.body_pos.

Exterior walls replace the tiny curbs in room.xml with proper 2.5m walls.
Interior walls are optional dividers with doorway gaps.

Usage:
    config = RoomConfig(half_extent=3.0, interior_walls=[
        InteriorWall(axis="x", offset=0.5, door_pos=-1.0, door_width=1.0),
    ])

    # At spec time:
    prepare_room(spec, config)

    # At runtime (after compile):
    room = Room(model, data)
    room.apply(config)   # positions walls + runs mj_forward
    room.arena_size       # usable half-extent for furniture placement
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Wall geometry constants
WALL_HEIGHT = 2.5  # meters
WALL_HALF_HEIGHT = WALL_HEIGHT / 2
WALL_THICKNESS = 0.08  # half-thickness (total = 16cm)
WALL_COLOR = (0.82, 0.80, 0.76, 1.0)  # off-white plaster

# How many interior wall segment slots to pre-allocate.
# Each interior wall uses 2 segments (one on each side of the doorway).
MAX_INTERIOR_WALLS = 2
SEGMENTS_PER_WALL = 2
MAX_WALL_SEGMENTS = MAX_INTERIOR_WALLS * SEGMENTS_PER_WALL

# Exterior wall slot names: room_wall_N, room_wall_S, room_wall_E, room_wall_W
# Interior wall segment names: room_iwall_0_seg0, room_iwall_0_seg1, ...


@dataclass(frozen=True)
class InteriorWall:
    """An interior wall dividing the room.

    Attributes:
        axis: "x" or "y" — the axis the wall runs along.
            "x" means the wall runs parallel to the X axis (divides Y).
            "y" means the wall runs parallel to the Y axis (divides X).
        offset: Position along the perpendicular axis (meters from center).
            For axis="x", offset is the Y position of the wall.
        door_pos: Position of the doorway center along the wall's axis
            (meters from center). 0 = center of wall.
        door_width: Width of the doorway opening (meters). Default 1.0m.
    """

    axis: str = "x"
    offset: float = 0.0
    door_pos: float = 0.0
    door_width: float = 1.0


@dataclass(frozen=True)
class RoomConfig:
    """Room size and wall layout.

    Attributes:
        half_extent: Half-size of the room (meters from center to wall).
            The room is 2 * half_extent on each side.
        interior_walls: Optional interior wall dividers.
        wall_height: Height of walls (meters).
    """

    half_extent: float = 3.5
    interior_walls: tuple[InteriorWall, ...] = ()
    wall_height: float = WALL_HEIGHT


def random_room_config(
    rng: np.random.Generator,
    min_extent: float = 2.5,
    max_extent: float = 4.0,
    interior_wall_prob: float = 0.3,
) -> RoomConfig:
    """Sample a random room configuration.

    Args:
        rng: Numpy random generator.
        min_extent: Minimum half-extent (meters).
        max_extent: Maximum half-extent (meters).
        interior_wall_prob: Probability of adding an interior wall.
    """
    half_extent = float(rng.uniform(min_extent, max_extent))

    walls: list[InteriorWall] = []
    if rng.random() < interior_wall_prob:
        axis = rng.choice(["x", "y"])
        # Offset: somewhere in the inner 60% of the room
        offset = float(rng.uniform(-half_extent * 0.3, half_extent * 0.3))
        # Door position: somewhere along the wall
        wall_length = half_extent * 2
        door_pos = float(rng.uniform(-wall_length * 0.3, wall_length * 0.3))
        door_width = float(rng.uniform(0.8, 1.4))
        walls.append(
            InteriorWall(
                axis=axis,
                offset=offset,
                door_pos=door_pos,
                door_width=door_width,
            )
        )

    return RoomConfig(half_extent=half_extent, interior_walls=tuple(walls))


def prepare_room(spec) -> None:
    """Inject wall body+geom slots into an MjSpec before compilation.

    Creates:
      - 4 exterior wall slots (room_wall_N/S/E/W)
      - MAX_WALL_SEGMENTS interior wall segment slots

    All start hidden (tiny, transparent). Position them at runtime via Room.apply().
    """
    # Exterior walls
    for name in ["room_wall_N", "room_wall_S", "room_wall_E", "room_wall_W"]:
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = [0, 0, 0]
        geom = body.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.001, 0.001, 0.001]
        geom.rgba = [0, 0, 0, 0]
        geom.contype = 1
        geom.conaffinity = 1

    # Interior wall segments
    for i in range(MAX_INTERIOR_WALLS):
        for j in range(SEGMENTS_PER_WALL):
            name = f"room_iwall_{i}_seg{j}"
            body = spec.worldbody.add_body()
            body.name = name
            body.pos = [0, 0, 0]
            geom = body.add_geom()
            geom.name = name
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = [0.001, 0.001, 0.001]
            geom.rgba = [0, 0, 0, 0]
            geom.contype = 1
            geom.conaffinity = 1


@dataclass
class _WallSlot:
    body_id: int
    geom_id: int


class Room:
    """Manages room walls at runtime by writing to pre-allocated MjSpec slots."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._exterior: dict[str, _WallSlot] = {}
        self._interior: list[_WallSlot] = []
        self._discover_slots()

    def _discover_slots(self):
        """Find wall slots in the compiled model."""
        for name in ["room_wall_N", "room_wall_S", "room_wall_E", "room_wall_W"]:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if bid >= 0 and gid >= 0:
                self._exterior[name] = _WallSlot(bid, gid)

        for i in range(MAX_INTERIOR_WALLS):
            for j in range(SEGMENTS_PER_WALL):
                name = f"room_iwall_{i}_seg{j}"
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if bid >= 0 and gid >= 0:
                    self._interior.append(_WallSlot(bid, gid))

    @property
    def has_slots(self) -> bool:
        """True if wall slots were found in the model."""
        return len(self._exterior) > 0

    def apply(self, config: RoomConfig) -> None:
        """Position walls according to the room config.

        Does NOT call mj_forward — the caller (SceneComposer.apply) handles that.
        """
        if not self.has_slots:
            return

        he = config.half_extent
        hh = config.wall_height / 2

        # Exterior walls: each is a box along one edge of the room
        # North wall: runs along X at y = +he
        self._set_wall(
            self._exterior["room_wall_N"],
            pos=(0, he, hh),
            size=(he + WALL_THICKNESS, WALL_THICKNESS, hh),
        )
        # South wall
        self._set_wall(
            self._exterior["room_wall_S"],
            pos=(0, -he, hh),
            size=(he + WALL_THICKNESS, WALL_THICKNESS, hh),
        )
        # East wall
        self._set_wall(
            self._exterior["room_wall_E"],
            pos=(he, 0, hh),
            size=(WALL_THICKNESS, he + WALL_THICKNESS, hh),
        )
        # West wall
        self._set_wall(
            self._exterior["room_wall_W"],
            pos=(-he, 0, hh),
            size=(WALL_THICKNESS, he + WALL_THICKNESS, hh),
        )

        # Interior walls
        seg_idx = 0
        for i, iwall in enumerate(config.interior_walls):
            if i >= MAX_INTERIOR_WALLS:
                break
            seg_idx = i * SEGMENTS_PER_WALL
            self._apply_interior_wall(config, iwall, seg_idx)

        # Hide unused interior wall segments
        used = len(config.interior_walls) * SEGMENTS_PER_WALL
        for slot in self._interior[used:]:
            self._hide_wall(slot)

    def _apply_interior_wall(
        self, config: RoomConfig, iwall: InteriorWall, seg_start: int
    ) -> None:
        """Place an interior wall with a doorway gap as 2 segments."""
        he = config.half_extent
        hh = config.wall_height / 2
        door_half = iwall.door_width / 2

        if iwall.axis == "x":
            # Wall runs along X axis at y = offset
            # Segment 1: from -he to (door_pos - door_half)
            seg1_start = -he
            seg1_end = iwall.door_pos - door_half
            # Segment 2: from (door_pos + door_half) to +he
            seg2_start = iwall.door_pos + door_half
            seg2_end = he

            if seg1_end > seg1_start + 0.05:
                seg1_half_len = (seg1_end - seg1_start) / 2
                seg1_center_x = (seg1_start + seg1_end) / 2
                self._set_wall(
                    self._interior[seg_start],
                    pos=(seg1_center_x, iwall.offset, hh),
                    size=(seg1_half_len, WALL_THICKNESS, hh),
                )
            else:
                self._hide_wall(self._interior[seg_start])

            if seg2_end > seg2_start + 0.05:
                seg2_half_len = (seg2_end - seg2_start) / 2
                seg2_center_x = (seg2_start + seg2_end) / 2
                self._set_wall(
                    self._interior[seg_start + 1],
                    pos=(seg2_center_x, iwall.offset, hh),
                    size=(seg2_half_len, WALL_THICKNESS, hh),
                )
            else:
                self._hide_wall(self._interior[seg_start + 1])

        else:  # axis == "y"
            # Wall runs along Y axis at x = offset
            seg1_start = -he
            seg1_end = iwall.door_pos - door_half
            seg2_start = iwall.door_pos + door_half
            seg2_end = he

            if seg1_end > seg1_start + 0.05:
                seg1_half_len = (seg1_end - seg1_start) / 2
                seg1_center_y = (seg1_start + seg1_end) / 2
                self._set_wall(
                    self._interior[seg_start],
                    pos=(iwall.offset, seg1_center_y, hh),
                    size=(WALL_THICKNESS, seg1_half_len, hh),
                )
            else:
                self._hide_wall(self._interior[seg_start])

            if seg2_end > seg2_start + 0.05:
                seg2_half_len = (seg2_end - seg2_start) / 2
                seg2_center_y = (seg2_start + seg2_end) / 2
                self._set_wall(
                    self._interior[seg_start + 1],
                    pos=(iwall.offset, seg2_center_y, hh),
                    size=(WALL_THICKNESS, seg2_half_len, hh),
                )
            else:
                self._hide_wall(self._interior[seg_start + 1])

    def _set_wall(
        self,
        slot: _WallSlot,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> None:
        """Make a wall segment visible at the given position and size."""
        self.model.body_pos[slot.body_id] = pos
        self.model.body_quat[slot.body_id] = [1, 0, 0, 0]
        self.model.geom_size[slot.geom_id] = size
        self.model.geom_pos[slot.geom_id] = [0, 0, 0]
        self.model.geom_quat[slot.geom_id] = [1, 0, 0, 0]
        self.model.geom_rgba[slot.geom_id] = WALL_COLOR
        self.model.geom_contype[slot.geom_id] = 1
        self.model.geom_conaffinity[slot.geom_id] = 1

    def _hide_wall(self, slot: _WallSlot) -> None:
        """Hide a wall segment."""
        self.model.body_pos[slot.body_id] = [0, 0, 0]
        self.model.geom_size[slot.geom_id] = [0.001, 0.001, 0.001]
        self.model.geom_rgba[slot.geom_id] = [0, 0, 0, 0]
        self.model.geom_contype[slot.geom_id] = 0
        self.model.geom_conaffinity[slot.geom_id] = 0
