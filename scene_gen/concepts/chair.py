"""Chair — a seat with four legs and a backrest.

Parameters:
    seat_width:     Seat width (X) in meters
    seat_depth:     Seat depth (Y) in meters
    seat_height:    Seat height from floor in meters
    seat_thickness: Seat slab thickness in meters
    back_height:    Backrest height above seat in meters
    back_thickness: Backrest thickness in meters
    leg_width:      Leg cross-section width in meters
    seat_color:     RGBA for seat and back
    leg_color:      RGBA for legs
"""

from dataclasses import dataclass
from functools import lru_cache

from scene_gen.primitives import FABRIC_BLUE, WOOD_DARK, GeomType, Prim


@dataclass(frozen=True)
class Params:
    seat_width: float = 0.45
    seat_depth: float = 0.42
    seat_height: float = 0.46
    seat_thickness: float = 0.04
    back_height: float = 0.40
    back_thickness: float = 0.03
    leg_width: float = 0.03
    seat_color: tuple[float, float, float, float] = FABRIC_BLUE
    leg_color: tuple[float, float, float, float] = WOOD_DARK


@lru_cache(maxsize=128)
def generate(params: Params = Params()) -> tuple[Prim, ...]:
    """Generate a chair (1 seat + 1 backrest + 4 legs = 6 prims)."""
    hw = params.seat_width / 2
    hd = params.seat_depth / 2
    ht = params.seat_thickness / 2
    hlw = params.leg_width / 2

    sc = params.seat_color
    lc = params.leg_color

    # Seat
    seat = Prim(GeomType.BOX, (hw, hd, ht), (0, 0, params.seat_height - ht), sc)

    # Backrest: at rear edge of seat (+Y), rising above seat
    back_half_h = params.back_height / 2
    back_z = params.seat_height + back_half_h
    back_y = hd - params.back_thickness / 2
    backrest = Prim(
        GeomType.BOX,
        (hw, params.back_thickness / 2, back_half_h),
        (0, back_y, back_z),
        sc,
    )

    # Legs: floor to underside of seat
    leg_full = params.seat_height - params.seat_thickness
    leg_half = leg_full / 2
    leg_z = leg_half
    lx = hw - hlw - 0.01
    ly = hd - hlw - 0.01

    legs = (
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, -ly, leg_z), lc),
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, -ly, leg_z), lc),
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (-lx, ly, leg_z), lc),
        Prim(GeomType.BOX, (hlw, hlw, leg_half), (lx, ly, leg_z), lc),
    )

    return (seat, backrest, *legs)
