"""Component catalog — real-world parts with accurate specs."""

from botcad.components.battery import LiPo2S
from botcad.components.camera import OV5647, PiCamera2
from botcad.components.compute import RaspberryPiZero2W
from botcad.components.controller import WaveshareSerialBus
from botcad.components.servo import STS3215
from botcad.components.wheel import PololuWheel90mm

__all__ = [
    "STS3215",
    "RaspberryPiZero2W",
    "LiPo2S",
    "OV5647",
    "PiCamera2",
    "PololuWheel90mm",
    "WaveshareSerialBus",
]
