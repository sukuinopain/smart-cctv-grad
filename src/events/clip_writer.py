import json
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import threading
from typing import List
import cv2
import numpy as np
from src.schemas import SecurityEvent


class AsyncClipWriter:
    def __init__(self, output_dir: str = "data/outputs", fps: int = 16):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.has_ffmpeg = shutil.which("ffmpeg") is not None
        self._queue: Queue = Queue()
        self._worker = threading.Thread(target=self._process_queue, daemon=True)
        self._worker.start()

    def submit(self, event: SecurityEvent, frames: List[np.ndarray]) -> None:
        self._queue.put((event, frames))

    def _process_queue(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            event, frames = item
            try:
                if frames and event.requires_video_clip:
                    clip_path = self.output_dir / f"{event.event_id}.mp4"
                    self._write_h264_mp4(clip_path, frames)
                    event.clip_path = str(clip_path)
                self._write_metadata(event)
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
        }
        with open(meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def shutdown(self) -> None:
        self._queue.put(None)
        self._worker.join()
