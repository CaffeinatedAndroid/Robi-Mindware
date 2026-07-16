from dataclasses import dataclass
from datetime import datetime
import numpy as np

# TODO CONVERT TO TOPIC ENUM CLASS
@dataclass
class TargetMessage:
    x: int
    y: int
    timestamp: datetime

    def __repr__(self):

        ts = self.timestamp.strftime("%H:%M:%S")

        return (
            f"TargetMessage("
            f"x={self.x}, "
            f"y={self.y}, "
            f"time={ts})"
        )

@dataclass
class ServoCommand:
    servo_ID: int
    servo_Angle: int
    timestamp: datetime

    def __repr__(self):

        ts = self.timestamp.strftime("%H:%M:%S")

        return (
            f"ServoCommand("
            f"command={self.servo_ID}:{self.servo_Angle}, "
            f"time={ts})"
        )



@dataclass
class VisionFrameMessage:

    frame: np.ndarray

    timestamp: datetime

    def __repr__(self):

        ts = self.timestamp.strftime("%H:%M:%S")

        return (
            f"FrameMessage("
            f"{self.frame},"
            f"time={ts})"
        )


