"""PrebuiltOp wrappers for bracket and cradle envelopes.

These functions build ShapeScript programs that wrap pre-computed
bracket/cradle envelope solids via the PrebuiltOp mechanism. The
resulting programs can be used as sub-programs in emit_body_ir
(via CallOp) or executed standalone for testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from botcad.shapescript.program import ShapeScript

if TYPE_CHECKING:
    from botcad.bracket import BracketSpec
    from botcad.component import ServoSpec


def bracket_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript wrapping bracket_envelope() as a PrebuiltOp."""
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import bracket_envelope

    if spec is None:
        spec = BS()
    prog = ShapeScript()
    ref = prog.prebuilt(bracket_envelope(servo, spec), tag="bracket_envelope")
    prog.output_ref = ref
    return prog


def cradle_envelope_script(
    servo: ServoSpec, spec: BracketSpec | None = None
) -> ShapeScript:
    """Build a ShapeScript wrapping cradle_envelope() as a PrebuiltOp."""
    from botcad.bracket import BracketSpec as BS
    from botcad.bracket import cradle_envelope

    if spec is None:
        spec = BS()
    prog = ShapeScript()
    ref = prog.prebuilt(cradle_envelope(servo, spec), tag="cradle_envelope")
    prog.output_ref = ref
    return prog
