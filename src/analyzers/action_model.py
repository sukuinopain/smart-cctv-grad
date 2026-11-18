from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18
from src.schemas import Detection


class FightActionDetector:
    """3D-CNN (R3D-18) temporal action classifier over a 16-frame (2.0s @ 8 FPS) rolling window."""

    def __init__(
        self,
        weights_path: str = "weights/fight_r3d18.pth",
        clip_len: int = 16,
        eval_stride: int = 4,
        conf_threshold: float = 0.75,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.clip_len = clip_len
        self.eval_stride = eval_stride
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.frame_buffer: deque[np.ndarray] = deque(maxlen=clip_len)
        self.step_counter = 0
        self.is_trained_weights = False
        self.model: Optional[nn.Module] = None

        # Kinetics-400 normalization constants expected by torchvision r3d_18
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

        if self.enabled:
            self._load_model(weights_path)

    def _load_model(self, weights_path: str) -> None:
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Class 0: NonFight, Class 1: Fight

        w_file = Path(weights_path)
        if w_file.exists():
            state = torch.load(w_file, map_location=self.device)
            model.load_state_dict(state)
            self.is_trained_weights = True
            print(f"[ACTION] Loaded fine-tuned Fight weights from {weights_path} on {self.device}.")
        else:
            print(
                f"[ACTION WARNING] {weights_path} not found! Model initialized with un-tuned head. "
                "Run scripts/train_action.py on RWF-2000 before relying on Fight alerts."
            )

        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def _letterbox_112(frame_bgr: np.ndarray) -> np.ndarray:
        """Resizes frame to 112x112 with black padding to preserve human posture proportions."""
        h, w = frame_bgr.shape[:2]
        scale = 112.0 / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((112, 112, 3), dtype=np.uint8)
        top = (112 - nh) // 2
        left = (112 - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def update(self, frame_bgr: np.ndarray, detections: List[Detection]) -> Tuple[bool, float, List[int]]:
        """
        Pushes current analysis frame and runs 3D-CNN inference when >= 2 people are visible.
        Returns: (is_fight, fight_probability, involved_track_ids)
        """
        if not self.enabled or self.model is None:
            return False, 0.0, []

        self.frame_buffer.append(self._letterbox_112(frame_bgr))
        self.step_counter += 1

        # Gate: Fight requires at least 2 detected people and a full 16-frame (2s) window
        if len(detections) < 2 or len(self.frame_buffer) < self.clip_len:
            return False, 0.0, []

        # Evaluate every `eval_stride` analysis frames (e.g. every 0.5s at 8 FPS)
        if (self.step_counter % self.eval_stride) != 0:
            return False, 0.0, []

        clip_np = np.stack(list(self.frame_buffer), axis=0)  # (T, H, W, C)
        tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float().to(self.device) / 255.0
        tensor = (tensor - self.mean) / self.std

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            fight_prob = float(probs[1].item())

        if self.is_trained_weights and fight_prob >= self.conf_threshold:
            # Sort participants by largest bounding box area (closest/main subjects)
            sorted_dets = sorted(detections, key=lambda d: d.area, reverse=True)
            involved_ids = [d.track_id for d in sorted_dets[:3]]
            return True, fight_prob, involved_ids

        return False, fight_prob, []