"""Centralized color palette for bot visualization.

All bot colors derive from the Blueprint.js palette (blueprintjs.com/docs/#core/colors).
Semantic grouping makes component identity readable at a glance:

    Greens  = electronics (PCBs, compute, camera, controller)
    Grays   = structural  (brackets, printed parts, servo body, wheels)
    Blues    = power       (battery) + mounting-point annotations
    Gold    = hardware     (fasteners, brass fittings)
    Silver  = metal        (bearings, steel)

Render-annotation colors (saturated overlays for mounting holes, wire ports,
shafts, horns) also come from Blueprint so everything is one family.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Color dataclass ──


@dataclass(frozen=True)
class Color:
    """Unified color: one definition, multiple representations.

    Construct with MuJoCo-style floats [0..1]. Derive PIL RGB and
    MuJoCo XML string automatically.
    """

    r: float
    g: float
    b: float
    a: float = 1.0
    label: str = ""

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        """Plain (r, g, b, a) tuple for component.color fields."""
        return (self.r, self.g, self.b, self.a)

    def with_alpha(self, a: float) -> Color:
        """Return a copy with a different alpha."""
        return Color(self.r, self.g, self.b, a, self.label)

    @property
    def rgba_str(self) -> str:
        """MuJoCo XML rgba attribute value (4 decimal places)."""
        return f"{self.r:.4f} {self.g:.4f} {self.b:.4f} {self.a:.4f}"

    @property
    def rgb(self) -> tuple[float, float, float]:
        """(r, g, b) floats [0..1], e.g. for build123d solid.color."""
        return (self.r, self.g, self.b)

    @property
    def rgb_int(self) -> tuple[int, int, int]:
        """PIL-compatible (R, G, B) in 0..255."""
        return (int(self.r * 255), int(self.g * 255), int(self.b * 255))

    @property
    def legend(self) -> tuple[str, tuple[int, int, int]]:
        """(label, rgb_int) pair for composite legends."""
        return (self.label, self.rgb_int)


# ── Blueprint.js palette (hex → 0-1 float) ──


def _hex(h: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) floats in [0..1], rounded to 4 dp."""
    h = h.lstrip("#")
    return (
        round(int(h[0:2], 16) / 255, 4),
        round(int(h[2:4], 16) / 255, 4),
        round(int(h[4:6], 16) / 255, 4),
    )


# Grays
BP_DARK_GRAY1 = _hex("#182026")
BP_DARK_GRAY3 = _hex("#293742")
BP_DARK_GRAY5 = _hex("#394B59")
BP_GRAY1 = _hex("#5C7080")
BP_GRAY3 = _hex("#8A9BA8")
BP_GRAY4 = _hex("#A7B6C2")
BP_GRAY5 = _hex("#BFCCD6")
BP_LIGHT_GRAY1 = _hex("#CED9E0")
BP_LIGHT_GRAY3 = _hex("#E1E8ED")
BP_LIGHT_GRAY5 = _hex("#F5F8FA")

# Blues
BP_BLUE1 = _hex("#0E5A8A")
BP_BLUE3 = _hex("#137CBD")
BP_BLUE4 = _hex("#2B95D6")
BP_BLUE5 = _hex("#48AFF0")

# Greens
BP_GREEN1 = _hex("#0A6640")
BP_GREEN3 = _hex("#0F9960")
BP_GREEN4 = _hex("#15B371")
BP_GREEN5 = _hex("#3DCC91")

# Reds
BP_RED3 = _hex("#DB3737")
BP_RED4 = _hex("#F55656")

# Orange
BP_ORANGE3 = _hex("#D9822B")
BP_ORANGE5 = _hex("#FFB366")

# Gold
BP_GOLD3 = _hex("#D99E0B")
BP_GOLD5 = _hex("#FFC940")

# Turquoise
BP_TURQUOISE3 = _hex("#00B3A4")
BP_TURQUOISE5 = _hex("#2EE6D6")

# Forest
BP_FOREST1 = _hex("#1D7324")
BP_FOREST3 = _hex("#238C2C")

# Violet
BP_VIOLET3 = _hex("#7157D9")

# Vermilion
BP_VERMILION3 = _hex("#D13913")


# ═══════════════════════════════════════════════════════════════════════════
# Semantic bot colors — the "visual language" for all bot renders
# ═══════════════════════════════════════════════════════════════════════════

# ── Structural (grays) ──

COLOR_STRUCTURE_BODY = Color(
    *BP_LIGHT_GRAY1, 1.0, "bracket (light gray)"
)  # brackets, printed parts
COLOR_STRUCTURE_DARK = Color(
    *BP_DARK_GRAY1, 1.0, "servo (dark)"
)  # servo body, dark housings
COLOR_STRUCTURE_RUBBER = Color(
    *BP_DARK_GRAY3, 1.0, "wheel (dark)"
)  # wheels, rubber parts
COLOR_STRUCTURE_DEFAULT = Color(
    *BP_GRAY3, 1.0, "part (gray)"
)  # generic component fallback
COLOR_STRUCTURE_WIREFRAME = Color(
    *BP_GRAY4, 1.0, "context (silver)"
)  # wireframe / context ghost
COLOR_STRUCTURE_HORN_DISC = Color(
    *BP_LIGHT_GRAY3, 1.0, "horn disc (pale)"
)  # CAD horn disc

# ── Electronics (greens) ──

COLOR_ELECTRONICS_PCB = Color(
    *BP_GREEN3, 1.0, "PCB (green)"
)  # generic PCB (compute, camera)
COLOR_ELECTRONICS_DARK = Color(
    *BP_GREEN1, 1.0, "PCB dark (green)"
)  # darker PCB variant (camera)
COLOR_ELECTRONICS_CONTROLLER = Color(
    *BP_FOREST3, 1.0, "controller (forest)"
)  # bus controller boards

# ── Power (blues) ──

COLOR_POWER_BATTERY = Color(*BP_BLUE3, 1.0, "battery (blue)")

# ── Hardware / metal ──

COLOR_METAL_STEEL = Color(*BP_GRAY4, 1.0, "bearing (steel)")  # bearings
COLOR_METAL_BRASS = Color(
    *BP_GOLD3, 1.0, "fastener (brass)"
)  # fasteners, brass fittings

# ── Render annotations (saturated overlays) ──

COLOR_MOUNTING = Color(*BP_BLUE4, 1.0, "mounting (blue)")
COLOR_WIRE_PORT = Color(*BP_ORANGE3, 1.0, "wire port (orange)")
COLOR_SHAFT = Color(*BP_RED3, 1.0, "shaft (red)")
COLOR_HORN = Color(*BP_GOLD5, 0.7, "horn (yellow)")
COLOR_HORN_HOLE = Color(*BP_GREEN4, 1.0, "horn holes (green)")
COLOR_REAR_HOLE = Color(*BP_TURQUOISE3, 1.0, "rear holes (cyan)")

# ── Assembly part roles ──

COLOR_BRACKET = COLOR_STRUCTURE_BODY
COLOR_SERVO_BODY = COLOR_STRUCTURE_DARK
COLOR_CRADLE = COLOR_STRUCTURE_BODY
COLOR_COUPLER = Color(*BP_RED4, 1.0, "coupler/moving (red)")

# ── Assembly feedback ──

COLOR_ENVELOPE = Color(*BP_RED3, 0.25, "envelope (red)")
COLOR_COLLISION = Color(*BP_RED3, 1.0, "collision (red)")
COLOR_CLEAR = Color(*BP_GREEN3, 1.0, "clear (green)")
COLOR_ARROW = Color(*BP_ORANGE3, 0.9, "arrow (orange)")

# ── Debug drawing palette (0-255 RGB for PIL/SVG) ──

DEBUG_PALETTE: list[tuple[int, int, int]] = [
    COLOR_COLLISION.rgb_int,  # red
    Color(*BP_BLUE3).rgb_int,  # blue
    Color(*BP_DARK_GRAY5).rgb_int,  # dark gray
    COLOR_CLEAR.rgb_int,  # green
    COLOR_ARROW.rgb_int,  # amber/orange
    Color(*BP_VIOLET3).rgb_int,  # violet
]

DIM_COLOR = "#5C7080"  # BP_GRAY1 — dimension line/text

# ── Compare-CAD colors ──

COLOR_COMPARE_GEN = COLOR_BRACKET
COLOR_COMPARE_EXTRA = Color(*BP_RED3, 1.0, "extra (red)")
COLOR_COMPARE_MISSING = Color(*BP_GOLD5, 1.0, "missing (yellow)")

# ── Wire routing (sim visualization) ──

COLOR_WIRE_UART = Color(*BP_BLUE4, 0.9, "wire UART (blue)")
COLOR_WIRE_CSI = Color(*BP_GOLD5, 0.9, "wire CSI (yellow)")
COLOR_WIRE_POWER = Color(*BP_RED3, 0.9, "wire power (red)")
COLOR_WIRE_DEFAULT = Color(*BP_GRAY3, 0.9, "wire default (gray)")

# ── Viz / Rerun colors ──

COLOR_VIZ_FLOOR = Color(*BP_GRAY3, 0.39, "floor (gray)")
COLOR_VIZ_ARENA = Color(*BP_RED4, 0.71, "arena boundary (red)")
COLOR_VIZ_TRAJECTORY = Color(*BP_GREEN4, 1.0, "trajectory (green)")

# ── TUI theme (hex strings for Textual CSS) ──
# Maps Textual design tokens to Blueprint.js dark palette.

TUI_PRIMARY = "#137CBD"  # BP_BLUE3 — accent, borders, highlights
TUI_SECONDARY = "#2B95D6"  # BP_BLUE4 — secondary accent
TUI_SURFACE = "#293742"  # BP_DARK_GRAY3 — panel backgrounds
TUI_BACKGROUND = "#182026"  # BP_DARK_GRAY1 — screen background
TUI_PRIMARY_BG = "#30404D"  # BP_DARK_GRAY4 — header/bar backgrounds
TUI_TEXT = "#F5F8FA"  # BP_LIGHT_GRAY5 — primary text
TUI_TEXT_MUTED = "#8A9BA8"  # BP_GRAY3 — dimmed/secondary text
TUI_SUCCESS = "#0F9960"  # BP_GREEN3 — healthy / good
TUI_WARNING = "#D99E0B"  # BP_GOLD3 — warning / moderate
TUI_ERROR = "#DB3737"  # BP_RED3 — error / bad
