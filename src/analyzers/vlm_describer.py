from typing import Optional
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.schemas import BestCropRecord, SecurityEvent


class VLMDescriber:
    """Extracts concise person attributes from BestCropRecord using Moondream2."""

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        cache_dir: str = "weights/hf_cache",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not self.enabled:
            return

        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            print(f"[VLM] Loading {model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, cache_dir=cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                cache_dir=cache_dir,
            ).to(self.device)
            self.model.eval()
            print("[VLM] Moondream2 loaded successfully.")
        except Exception as e:
            print(f"[VLM WARNING] Failed to load VLM ({e}). Falling back to metadata-only alerts.")
            self.enabled = False

    def _ask(self, pil_img: Image.Image, prompt: str) -> str:
        with torch.no_grad():
            if hasattr(self.model, "query"):
                res = self.model.query(pil_img, prompt)
                return str(res.get("answer", "")).strip()
            enc_image = self.model.encode_image(pil_img)
            return str(self.model.answer_question(enc_image, prompt, self.tokenizer)).strip()

    def describe_person(self, best_crop: Optional[BestCropRecord], track_id: int) -> str:
        if best_crop is None or best_crop.crop_bgr is None or best_crop.crop_bgr.size == 0:
            return f"Person ID {track_id}"

        if not self.enabled or self.model is None:
            mode = "IR Night Mode - colors unavailable" if best_crop.is_grayscale_ir else "Daylight"
            return f"Person ID {track_id} ({mode})"

        try:
            rgb = cv2.cvtColor(best_crop.crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            if best_crop.is_grayscale_ir:
                prompt = (
                    "Describe this person's clothing style and any carried bag or item in one short phrase. "
                    "Do not guess colors."
                )
                ans = self._ask(pil_img, prompt)
                return f"{ans} (IR Night Mode - colors unavailable)"
            else:
                prompt = (
                    "Describe this person's appearance, clothing with colors, and any carried bag or item "
                    "in one concise phrase (e.g. 'Man in a blue jacket and dark pants carrying a backpack')."
                )
                ans = self._ask(pil_img, prompt)
                return ans if ans else f"Person ID {track_id}"
        except Exception as e:
            print(f"[VLM ERROR] Inference failed for ID {track_id}: {e}")
            return f"Person ID {track_id}"

    def build_alert_message(self, event: SecurityEvent, best_crop: Optional[BestCropRecord]) -> str:
        tid = event.involved_track_ids[0] if event.involved_track_ids else -1
        subject_desc = self.describe_person(best_crop, tid)
        return f"[{event.event_type.value}] at {event.zone_name} (t={event.start_ts:.1f}s) | Subject: {subject_desc}"