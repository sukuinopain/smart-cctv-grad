from collections import deque
from dataclasses import dataclass
import threading
from typing import List
import cv2
import numpy as np


@dataclass
class CompressedFrame:
    frame_id: int
    timestamp: float
    jpeg_bytes: np.ndarray


class RingBuffer:
    def __init__(self, buffer_seconds: int = 30, buffer_fps: int = 16, jpeg_quality: int = 85):
        self.max_frames = buffer_seconds * buffer_fps
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        self._buffer: deque[CompressedFrame] = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def push(self, frame_id: int, timestamp: float, frame_bgr: np.ndarray) -> None:
        success, encoded = cv2.imencode(".jpg", frame_bgr, self.encode_params)
        if not success:
            return
        with self._lock:
            self._buffer.append(
                CompressedFrame(frame_id=frame_id, timestamp=timestamp, jpeg_bytes=encoded)
            )

    def extract_window(self, start_ts: float, end_ts: float) -> List[np.ndarray]:
        with self._lock:
            matching = [
                item.jpeg_bytes
                for item in self._buffer
                if start_ts <= item.timestamp <= end_ts
            ]

        frames = []
        for buf in matching:
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is not None:
                frames.append(decoded)
        return frames

    def latest_timestamp(self) -> float:
        with self._lock:
            return self._buffer[-1].timestamp if self._buffer else 0.0
