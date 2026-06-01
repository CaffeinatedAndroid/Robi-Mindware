from dataclasses import dataclass
from datetime import datetime


#TODO ADD SOURCE NODE IE: FROM VISION NODE
@dataclass
class Envelope:

    topic: str

    timestamp: datetime

    payload: object

