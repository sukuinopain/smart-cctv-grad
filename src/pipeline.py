import argparse
from typing import Dict, List
import uuid
import yaml
from src.events.clip_writer import AsyncClipWriter
from src.ingestion.ring_buffer import RingBuffer
from src.ingestion.stream_reader import StreamReader
from src.perception.best_frame import BestFrameSelector
from src.perception.tracker import PersonTracker
from src.schemas import EventType, SecurityEvent


class SmartCCTVPipeline:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        s_cfg = self.cfg["stream"]
        p_cfg = self.cfg["perception"]
        e_cfg = self.cfg["events"]

        self.pre_roll = float(s_cfg["pre_roll_seconds"])
        self.post_roll = float(s_cfg["post_roll_seconds"])
        self.cooldown = float(e_cfg["cooldown_seconds"])

        self.buffer = RingBuffer(
            buffer_seconds=int(s_cfg["buffer_seconds"]),
            buffer_fps=int(s_cfg["buffer_fps"]),
            jpeg_quality=int(s_cfg["jpeg_quality"]),
        )
        self.tracker = PersonTracker(
            weights_path=p_cfg["model_weights"],
            tracker_cfg=p_cfg["tracker_config"],
            conf_threshold=float(p_cfg["confidence_threshold"]),
            person_class_id=int(p_cfg["person_class_id"]),
        )
        self.best_frame_selector = BestFrameSelector()
        self.clip_writer = AsyncClipWriter(
            output_dir=e_cfg["output_dir"],
            fps=int(s_cfg["buffer_fps"]),
        )

        self.track_first_seen: Dict[int, float] = {}
        self.triggered_tracks: set = set()
        self.pending_exports: List[SecurityEvent] = []

    def _process_pending_exports(self, current_ts: float, force_flush: bool = False) -> None:
        remaining: List[SecurityEvent] = []
        for event in self.pending_exports:
            target_end_ts = event.start_ts + self.post_roll
            if current_ts >= target_end_ts or force_flush:
                clip_start = max(0.0, event.start_ts - self.pre_roll)
                clip_end = min(current_ts, target_end_ts)
                frames = self.buffer.extract_window(clip_start, clip_end)
                print(f"[EXPORT] Event {event.event_id} ({event.event_type.value}) -> {len(frames)} frames")
                self.clip_writer.submit(event, frames)
            else:
                remaining.append(event)
        self.pending_exports = remaining

    def run(self, video_source: str) -> None:
        s_cfg = self.cfg["stream"]
        reader = StreamReader(
            source=video_source,
            buffer_fps=int(s_cfg["buffer_fps"]),
            analysis_fps=int(s_cfg["analysis_fps"]),
        )

        last_ts = 0.0
        try:
            for frame_id, ts, frame, run_ai in reader.frames():
                last_ts = ts
                self.buffer.push(frame_id, ts, frame)

                if run_ai:
                    detections = self.tracker.update(frame, frame_id, ts)
                    self.best_frame_selector.update(frame, detections)

                    for det in detections:
                        if det.track_id not in self.track_first_seen:
                            self.track_first_seen[det.track_id] = ts
                        elif (
                            det.track_id not in self.triggered_tracks
                            and (ts - self.track_first_seen[det.track_id]) >= 1.5
                        ):
                            self.triggered_tracks.add(det.track_id)
                            best = self.best_frame_selector.get_best(det.track_id)
                            ir_note = "IR Night Mode" if (best and best.is_grayscale_ir) else "Daylight Color"
                            event = SecurityEvent(
                                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                                event_type=EventType.ARRIVAL,
                                start_ts=self.track_first_seen[det.track_id],
                                last_seen_ts=ts,
                                zone_name="walking_skeleton_test",
                                involved_track_ids=[det.track_id],
                                description=f"Person ID {det.track_id} detected ({ir_note}).",
                            )
                            print(f"[TRIGGER] {event.event_id} at t={event.start_ts}s — waiting post-roll...")
                            self.pending_exports.append(event)

                self._process_pending_exports(ts, force_flush=False)
        finally:
            self._process_pending_exports(last_ts, force_flush=True)
            reader.release()
            self.clip_writer.shutdown()
            print("[DONE] Pipeline finished and all clips exported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to video file or RTSP URL")
    args = parser.parse_args()

    pipeline = SmartCCTVPipeline()
    pipeline.run(args.source)
