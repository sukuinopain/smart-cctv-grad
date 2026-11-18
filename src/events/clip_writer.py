import json
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import threading
from typing import Dict, List, Optional
import cv2
import numpy as np
from src.analyzers.vlm_describer import VLMDescriber
from src.schemas import BestCropRecord, SecurityEvent


class AsyncClipWriter:
    """Background worker that saves person snapshots, runs VLM description, encodes H.264 clips, and logs JSONL."""

    def __init__(
        self,
        output_dir: str = "data/outputs",
        fps: int = 16,
        vlm_describer: Optional[VLMDescriber] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.vlm_describer = vlm_describer
        self.has_ffmpeg = shutil.which("ffmpeg") is not None
        self._queue: Queue = Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def submit(
        self,
        event: SecurityEvent,
        frames: List[np.ndarray],
        best_crops: Optional[Dict[int, Optional[BestCropRecord]]] = None,
    ) -> None:
        self._queue.put((event, frames, best_crops or {}))

    def _process_queue(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            event, frames, best_crops = item
            try:
                # 1. Save sharpest subject snapshots for UI display
                saved_crops: List[str] = []
                for tid, crop_rec in best_crops.items():
                    if crop_rec is not None and crop_rec.crop_bgr is not None and crop_rec.crop_bgr.size > 0:
                        crop_file = self.output_dir / f"{event.event_id}_p{tid}.jpg"
                        cv2.imwrite(str(crop_file), crop_rec.crop_bgr)
                        saved_crops.append(str(crop_file))
                event.crop_paths = saved_crops

                # 2. Generate natural-language description via VLM
                if self.vlm_describer is not None:
                    event.description = self.vlm_describer.build_alert_message(event, best_crops)

                # 3. Encode 15s H.264 video clip
                if frames and event.requires_video_clip:
                    clip_path = self.output_dir / f"{event.event_id}.mp4"
                    self._write_h264_mp4(clip_path, frames)
                    event.clip_path = str(clip_path)

                self._write_metadata(event)
                print(f"[SAVED] {event.event_id} -> {event.description}")
            finally:
                self._queue.task_done()

    def _write_h264_mp4(self, path: Path, frames: List[np.ndarray]) -> None:
        h, w = frames[0].shape[:2]
        if self.has_ffmpeg:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-movflags", "+faststart",
                str(path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
            proc.wait()
        else:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

    def _write_metadata(self, event: SecurityEvent) -> None:
        meta_path = self.output_dir / "events_log.jsonl"
        record = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "start_ts": event.start_ts,
            "last_seen_ts": event.last_seen_ts,
            "zone": event.zone_name,
            "track_ids": event.involved_track_ids,
            "description": event.description or "",
            "clip_path": event.clip_path,
            "crop_paths": event.crop_paths,
        }
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join()