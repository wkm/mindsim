"""Dimension types to prevent unit confusion.

Zero-cost at runtime (NewType), caught by type checkers.
All values are SI: meters for lengths, radians for angles.
"""

from typing import NewType

Meters = NewType("Meters", float)
Radians = NewType("Radians", float)
