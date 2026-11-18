import time
from typing import Generator, Tuple, Union
import cv2
import numpy as np


class StreamReader:
    def __init__(self, source: Union[str, int], buffer_fps: int = 16, analysis_fps: int = 8):
        self.source = source
        self.buffer_fps = buffer_fps
        self.analysis_stride = max(1, round(buffer_fps / analysis_fps))
        self.frame_interval_sec = 1.0 / float(buffer_fps)
        self.is_live = isinstance(source, int) or str(source).startswith(("rtsp://", "http://", "https://"))

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = raw_fps if raw_fps and raw_fps > 0 else 25.0

    def frames(self) -> Generator[Tuple[int, float, np.ndarray, bool], None, None]:
        sampled_id = 0
        raw_index = 0
        last_emit_time = -self.frame_interval_sec
        start_wall_time = time.monotonic()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.is_live:
                current_ts = time.monotonic() - start_wall_time
            else:
                pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                current_ts = (pos_msec / 1000.0) if pos_msec > 0 else (raw_index / self.source_fps)

            raw_index += 1

            if (current_ts - last_emit_time) >= (self.frame_interval_sec - 1e-4):
                last_emit_time = current_ts
                is_analysis_frame = (sampled_id % self.analysis_stride) == 0
                yield sampled_id, round(current_ts, 3), frame, is_analysis_frame
                sampled_id += 1

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
