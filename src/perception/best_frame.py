from typing import Dict, List, Optional
import cv2
import numpy as np
from src.schemas import BestCropRecord, Detection


class BestFrameSelector:
    def __init__(self, padding_ratio: float = 0.1):
        self.padding_ratio = padding_ratio
        self.records: Dict[int, BestCropRecord] = {}

    @staticmethod
    def _is_grayscale_ir(crop_bgr: np.ndarray, threshold: float = 5.0) -> bool:
        b, g, r = cv2.split(crop_bgr)
        diff_rg = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
        diff_gb = np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
        return (diff_rg + diff_gb) < threshold

    @staticmethod
    def _iou(box_a: tuple, box_b: tuple) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / float(area_a + area_b - inter + 1e-6)

    def update(self, frame_bgr: np.ndarray, detections: List[Detection]) -> None:
        h, w = frame_bgr.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            touches_edge = x1 <= 2 or y1 <= 2 or x2 >= (w - 2) or y2 >= (h - 2)
            edge_factor = 0.5 if touches_edge else 1.0

            max_overlap = max(
                (self._iou(det.bbox, other.bbox) for other in detections if other.track_id != det.track_id),
                default=0.0,
            )
            if max_overlap > 0.4:
                continue

            bw, bh = x2 - x1, y2 - y1
            pad_x = int(bw * self.padding_ratio)
            pad_y = int(bh * self.padding_ratio)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            score = sharpness * (det.area ** 0.5) * edge_factor

            prev = self.records.get(det.track_id)
            if prev is None or score > prev.score:
                self.records[det.track_id] = BestCropRecord(
                    track_id=det.track_id,
                    score=score,
                    crop_bgr=crop.copy(),
                    is_grayscale_ir=self._is_grayscale_ir(crop),
                    timestamp=det.timestamp,
                )

    def get_best(self, track_id: int) -> Optional[BestCropRecord]:
        return self.records.get(track_id)
