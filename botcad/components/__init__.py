"""Component catalog — real-world parts with accurate specs."""

from botcad.components.battery import LiPo2S
from botcad.components.camera import OV5647, PiCamera2
from botcad.components.compute import RaspberryPiZero2W
from botcad.components.controller import WaveshareSerialBus
from botcad.components.servo import SCS0009, STS3215, STS3250
from botcad.components.test_fastener import TestFastenerPrism
from botcad.components.wheel import PololuWheel90mm

__all__ = [
    "SCS0009",
    "STS3215",
    "STS3250",
    "RaspberryPiZero2W",
    "LiPo2S",
    "OV5647",
    "PiCamera2",
    "PololuWheel90mm",
    "WaveshareSerialBus",
    "TestFastenerPrism",
]
