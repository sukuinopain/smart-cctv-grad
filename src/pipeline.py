import argparse
from typing import List, Optional, Tuple
import uuid
import yaml
from src.analyzers.spatial_rules import SpatialRuleEngine
from src.analyzers.vlm_describer import VLMDescriber
from src.events.clip_writer import AsyncClipWriter
from src.events.debouncer import EventDebouncer
from src.ingestion.ring_buffer import RingBuffer
from src.ingestion.stream_reader import StreamReader
from src.perception.best_frame import BestFrameSelector
from src.perception.tracker import PersonTracker
from src.schemas import BestCropRecord, SecurityEvent


class SmartCCTVPipeline:
    def __init__(
        self,
        config_path: str = "configs/settings.yaml",
        zones_path: str = "configs/camera_zones.json",
        enable_vlm: Optional[bool] = None,
    ):
        with open(config_path, "r", encoding="utf-8-sig") as f:
            self.cfg = yaml.safe_load(f)

        s_cfg = self.cfg["stream"]
        p_cfg = self.cfg["perception"]
        v_cfg = self.cfg.get("vlm", {})
        e_cfg = self.cfg["events"]

        self.pre_roll = float(s_cfg["pre_roll_seconds"])
        self.post_roll = float(s_cfg["post_roll_seconds"])

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
        self.spatial_engine = SpatialRuleEngine(zones_path=zones_path)
        self.debouncer = EventDebouncer(cooldown_seconds=float(e_cfg["cooldown_seconds"]))

        use_vlm = v_cfg.get("enabled", True) if enable_vlm is None else enable_vlm
        self.vlm_describer = VLMDescriber(
            model_id=v_cfg.get("model_id", "vikhyatk/moondream2"),
            cache_dir=v_cfg.get("cache_dir", "weights/hf_cache"),
            enabled=use_vlm,
        )
        self.clip_writer = AsyncClipWriter(
            output_dir=e_cfg["output_dir"],
            fps=int(s_cfg["buffer_fps"]),
            vlm_describer=self.vlm_describer,
        )

        self.pending_exports: List[Tuple[SecurityEvent, Optional[BestCropRecord]]] = []

    def _process_pending_exports(self, current_ts: float, force_flush: bool = False) -> None:
        remaining: List[Tuple[SecurityEvent, Optional[BestCropRecord]]] = []
        for event, best_crop in self.pending_exports:
            target_end_ts = event.start_ts + self.post_roll
            if current_ts >= target_end_ts or force_flush:
                clip_start = max(0.0, event.start_ts - self.pre_roll)
                clip_end = min(current_ts, target_end_ts)
                frames = self.buffer.extract_window(clip_start, clip_end)
                tid = event.involved_track_ids[0] if event.involved_track_ids else -1
                latest_best = self.best_frame_selector.get_best(tid) or best_crop
                print(f"[EXPORT] {event.event_id} ({event.event_type.value}) -> {len(frames)} frames")
                self.clip_writer.submit(event, frames, latest_best)
            else:
                remaining.append((event, best_crop))
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
                    raw_events = self.spatial_engine.update(detections, frame, ts)

                    for raw in raw_events:
                        ev_type = raw["event_type"]
                        zone = raw["zone_name"]
                        tid = raw["track_id"]

                        if not self.debouncer.should_trigger(zone, ev_type, raw["trigger_ts"]):
                            continue

                        best = self.best_frame_selector.get_best(tid)
                        event = SecurityEvent(
                            event_id=f"evt_{uuid.uuid4().hex[:8]}",
                            event_type=ev_type,
                            start_ts=raw["start_ts"],
                            last_seen_ts=raw["trigger_ts"],
                            zone_name=zone,
                            involved_track_ids=[tid],
                            description=f"{ev_type.value} at {zone} | Person ID {tid}",
                        )
                        print(f"[TRIGGER] {event.event_id}: {ev_type.value} at {zone} (t={event.start_ts}s)")
                        self.pending_exports.append((event, best))

                self._process_pending_exports(ts, force_flush=False)
        finally:
            self._process_pending_exports(last_ts, force_flush=True)
            reader.release()
            self.clip_writer.shutdown()
            print("[DONE] Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to video file or RTSP URL")
    parser.add_argument("--zones", type=str, default="configs/camera_zones.json", help="Path to zones JSON")
    parser.add_argument("--no-vlm", action="store_true", help="Disable VLM for ultra-fast testing")
    args = parser.parse_args()

    pipeline = SmartCCTVPipeline(zones_path=args.zones, enable_vlm=not args.no_vlm)
    pipeline.run(args.source)