"""ShapeScript emitters for bracket geometry.

bracket_envelope_script / cradle_envelope_script emit native Box + locate
ops (no PrebuiltOp).

bracket_solid_script translates bracket_solid() line-by-line into
ShapeScript ops: outer box, servo pocket, horn clearance, shaft boss,
fastener holes, and connector/cable passage. Both STS3215 and SCS0009
use native ops. Connector ports that are hard to express purely in
ShapeScript use native Box + locate ops.

cradle_solid_script translates cradle_solid() into native ShapeScript ops:
outer box, servo pocket cut, fastener holes, and connector/cable passage.

coupler_solid_script builds the C-shaped coupler: front plate, rear plate,
side wall, horn holes, boss clearance. D-clip plate geometry uses native
cylinder-cut-in-half ops (no PrebuiltOp).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.ops import ShapeRef
from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.bracket import BracketSpec
    from botcad.component import ServoSpec


def _emit_connector_port(
    prog: ShapeScript,
    servo: ServoSpec,
    spec: BracketSpec,
    wall_center: tuple[float, float, float],
    exit_axis: tuple[float, float, float],
) -> ShapeRef | None:
    """Emit native ShapeScript ops for a connector passage through a bracket wall.

    Computes a bounding box over all wire port connector envelopes + clearance,
    then emits a single Box + locate positioned at wall_center. Returns a ShapeRef
    for the cut solid, or None if no connectors are present.

    This is the ShapeScript equivalent of bracket._connector_port().
    """
    from botcad.connectors import connector_spec

    # Collect all wire ports with connectors
    ports = []
    for wp in servo.wire_ports:
        if wp.connector_type:
            try:
                cspec = connector_spec(wp.connector_type)
                ports.append((wp, cspec))
            except KeyError:
                pass

    if not ports:
        return None

    tol = spec.tolerance
    wall = spec.wall
    ax, ay, az = exit_axis

    clearance = tol + 0.001  # per side

    if abs(ax) > 0.5:
        # Passage through X wall — footprint in Y-Z plane
        y_coords: list[float] = []
        z_coords: list[float] = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            y_coords.extend([wp.pos[1] - hw, wp.pos[1] + hw])
            z_coords.extend([wp.pos[2] - hh, wp.pos[2] + hh])
        cut_w = max(y_coords) - min(y_coords)
        cut_h = max(z_coords) - min(z_coords)
        center_y = (max(y_coords) + min(y_coords)) / 2
        center_z = (max(z_coords) + min(z_coords)) / 2
        passage_depth = wall + tol + 0.004
        cut = prog.box(passage_depth, cut_w, cut_h, tag="connector_port")
        cut_pos = (wall_center[0], center_y, center_z)
    elif abs(az) > 0.5:
        # Passage through Z wall — footprint in X-Y plane
        x_coords: list[float] = []
        y_coords2: list[float] = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            x_coords.extend([wp.pos[0] - hw, wp.pos[0] + hw])
            y_coords2.extend([wp.pos[1] - hh, wp.pos[1] + hh])
        cut_w = max(x_coords) - min(x_coords)
        cut_h = max(y_coords2) - min(y_coords2)
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords2) + min(y_coords2)) / 2
        passage_depth = wall + tol + 0.004
        cut = prog.box(cut_w, cut_h, passage_depth, tag="connector_port")
        cut_pos = (center_x, center_y, wall_center[2])
    else:
        # Passage through Y wall — footprint in X-Z plane
        x_coords2: list[float] = []
        z_coords2: list[float] = []
        for wp, cspec in ports:
            bx, by, bz = cspec.body_dimensions
            hw = max(bx, by) / 2 + clearance
            hh = max(min(bx, by), bz) / 2 + clearance
            x_coords2.extend([wp.pos[0] - hw, wp.pos[0] + hw])
            z_coords2.extend([wp.pos[2] - hh, wp.pos[2] + hh])
        cut_w = max(x_coords2) - min(x_coords2)
        cut_h = max(z_coords2) - min(z_coords2)
        center_x = (max(x_coords2) + min(x_coords2)) / 2
        center_z = (max(z_coords2) + min(z_coords2)) / 2
        passage_depth = wall + tol + 0.004
        cut = prog.box(cut_w, passage_depth, cut_h, tag="connector_port")
        cut_pos = (center_x, wall_center[1], center_z)

    cut = prog.locate(cut, pos=cut_pos)
    return cut


def bracket_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's bracket_envelope_ir()."""
    from botcad.bracket import bracket_envelope_ir

    return bracket_envelope_ir(servo, spec)


def bracket_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's bracket_solid_ir()."""
    from botcad.bracket import bracket_solid_ir

    return bracket_solid_ir(servo, spec)


def cradle_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's cradle_envelope_ir()."""
    from botcad.bracket import cradle_envelope_ir

    return cradle_envelope_ir(servo, spec)


def cradle_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's cradle_solid_ir()."""
    from botcad.bracket import cradle_solid_ir

    return cradle_solid_ir(servo, spec)


def coupler_solid_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Delegate to bracket.py's coupler_solid_ir()."""
    from botcad.bracket import coupler_solid_ir

    return coupler_solid_ir(servo, spec)
