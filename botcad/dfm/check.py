"""DFM check framework.

Each check is a DFMCheck subclass. The runner discovers all subclasses
at import time — no central enum to maintain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from botcad.assembly.refs import ComponentRef, FastenerRef, WireRef
from botcad.assembly.sequence import AssemblySequence
from botcad.ids import BodyId, JointId
from botcad.skeleton import Bot


class DFMSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class DFMFinding:
    """A single DFM issue found by a check."""

    check_name: str
    severity: DFMSeverity
    body: BodyId
    target: ComponentRef | FastenerRef | WireRef | JointId
    assembly_step: int
    title: str
    description: str
    pos: tuple[float, float, float]
    direction: tuple[float, float, float] | None
    measured: float | None
    threshold: float | None
    has_overlay: bool

    @property
    def id(self) -> str:
        """Deterministic, stable across runs."""
        if isinstance(self.target, FastenerRef):
            target_str = f"fastener:{self.target.body}:{self.target.index}"
        elif isinstance(self.target, ComponentRef):
            target_str = f"component:{self.target.body}:{self.target.mount_label}"
        elif isinstance(self.target, WireRef):
            target_str = f"wire:{self.target.label}"
        else:
            target_str = f"joint:{self.target}"
        return f"{self.check_name}:{self.body}:{target_str}"


class DFMCheck(ABC):
    """Base class for all DFM checks."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(
        self,
        bot: Bot,
        sequence: AssemblySequence,
    ) -> list[DFMFinding]: ...
