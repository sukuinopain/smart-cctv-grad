from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class EventType(str, Enum):
    ARRIVAL = "Arrival"
    DEPARTURE = "Departure"
    FIGHT = "Fight"
    POSSIBLE_THEFT = "Possible_Theft"


class EventState(str, Enum):
    OPEN = "open"
    ONGOING = "ongoing"
    CLOSED = "closed"


@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    timestamp: float
    frame_id: int

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)

    @property
    def area(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0]) * max(0, self.bbox[3] - self.bbox[1])


@dataclass
class BestCropRecord:
    track_id: int
    score: float
    crop_bgr: np.ndarray
    is_grayscale_ir: bool
    timestamp: float


@dataclass
class SecurityEvent:
    event_id: str
    event_type: EventType
    start_ts: float
    last_seen_ts: float
    zone_name: str
    involved_track_ids: List[int] = field(default_factory=list)
    state: EventState = EventState.OPEN
    requires_video_clip: bool = True
    description: Optional[str] = None
    clip_path: Optional[str] = None
    crop_paths: List[str] = field(default_factory=list)