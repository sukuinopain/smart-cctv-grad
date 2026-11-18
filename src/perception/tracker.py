from typing import List
import numpy as np
from ultralytics import YOLO
from src.schemas import Detection


class PersonTracker:
    def __init__(
        self,
        weights_path: str = "weights/yolov8n.pt",
        tracker_cfg: str = "bytetrack.yaml",
        conf_threshold: float = 0.45,
        person_class_id: int = 0,
    ):
        self.model = YOLO(weights_path)
        self.tracker_cfg = tracker_cfg
        self.conf_threshold = conf_threshold
        self.person_class_id = person_class_id

    def update(self, frame_bgr: np.ndarray, frame_id: int, timestamp: float) -> List[Detection]:
        results = self.model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf_threshold,
            classes=[self.person_class_id],
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        for box, tid, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            detections.append(
                Detection(
                    track_id=tid,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    timestamp=timestamp,
                    frame_id=frame_id,
                )
            )
        return detections
