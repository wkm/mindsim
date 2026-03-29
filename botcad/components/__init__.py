"""Component catalog — real-world parts with accurate specs."""

from botcad.components.battery import LiPo2S, LiPo3S
from botcad.components.bec import BEC5V
from botcad.components.camera import OV5647, PiCamera2, PiCamera3, PiCamera3Wide
from botcad.components.compute import RaspberryPiZero2W
from botcad.components.controller import WaveshareSerialBus
from botcad.components.esc import SimonK30A
from botcad.components.flight_controller import MatekF405Wing
from botcad.components.motor import MT2213
from botcad.components.propeller import Propeller9x45
from botcad.components.servo import SCS0009, STS3215, STS3250
from botcad.components.test_fastener import TestFastenerPrism
from botcad.components.wheel import PololuWheel90mm

__all__ = [
    "BEC5V",
    "MT2213",
    "OV5647",
    "SCS0009",
    "STS3215",
    "STS3250",
    "LiPo2S",
    "LiPo3S",
    "MatekF405Wing",
    "PiCamera2",
    "PiCamera3",
    "PiCamera3Wide",
    "PololuWheel90mm",
    "Propeller9x45",
    "RaspberryPiZero2W",
    "SimonK30A",
    "TestFastenerPrism",
    "WaveshareSerialBus",
]
