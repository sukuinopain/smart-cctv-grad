import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from src.schemas import Detection, EventType


class SpatialRuleEngine:
    """Evaluates ROI filtering, Arrival/Departure, and Zone-based Possible Theft."""

    def __init__(self, zones_path: str = "configs/camera_zones.json"):
        path = Path(zones_path)
        self.cfg = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        self.roi_polygons = self.cfg.get("roi_polygons", [])
        self.door_zones = self.cfg.get("door_zones", [])
        self.asset_zones = self.cfg.get("asset_zones", [])

        self.track_history: Dict[int, dict] = {}
        self.triggered_tracks: Dict[int, set] = {}

        self.asset_baselines: Dict[str, np.ndarray] = {}
        self.asset_dwell: Dict[str, Dict[int, float]] = {z["name"]: {} for z in self.asset_zones}
        self.asset_candidates: Dict[str, Dict[int, float]] = {z["name"]: {} for z in self.asset_zones}

    @staticmethod
    def _get_anchor(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        if y2 >= (h - 5):
            return (cx, (y1 + y2) / 2.0)
        return (cx, float(y2))

    @staticmethod
    def _point_in_poly(pt: Tuple[float, float], norm_points: List[List[float]], w: int, h: int) -> bool:
        poly = np.array([[p[0] * w, p[1] * h] for p in norm_points], dtype=np.float32)
        return cv2.pointPolygonTest(poly, pt, False) >= 0

    @staticmethod
    def _segments_intersect(
        a: Tuple[float, float], b: Tuple[float, float],
        c: Tuple[float, float], d: Tuple[float, float]
    ) -> bool:
        def ccw(p1, p2, p3):
            return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    @staticmethod
    def _classify_direction(
        prev_pt: Tuple[float, float], curr_pt: Tuple[float, float], inside_dir: str
    ) -> EventType:
        dx = curr_pt[0] - prev_pt[0]
        dy = curr_pt[1] - prev_pt[1]
        is_inside = {
            "down": dy > 0,
            "up": dy < 0,
            "right": dx > 0,
            "left": dx < 0,
        }.get(inside_dir.lower(), dy > 0)
        return EventType.ARRIVAL if is_inside else EventType.DEPARTURE

    @staticmethod
    def _extract_zone_patch(frame_bgr: np.ndarray, norm_points: List[List[float]]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        xs = [int(p[0] * w) for p in norm_points]
        ys = [int(p[1] * h) for p in norm_points]
        x1, x2 = max(0, min(xs)), min(w, max(xs))
        y1, y2 = max(0, min(ys)), min(h, max(ys))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((64, 64), dtype=np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        return cv2.resize(blurred, (64, 64)).astype(np.float32)

    def filter_in_roi(self, detections: List[Detection], w: int, h: int) -> List[Detection]:
        if not self.roi_polygons:
            return detections
        filtered = []
        for det in detections:
            anchor = self._get_anchor(det.bbox, w, h)
            if any(self._point_in_poly(anchor, roi["points"], w, h) for roi in self.roi_polygons):
                filtered.append(det)
        return filtered

    def update(
        self, detections: List[Detection], frame_bgr: np.ndarray, current_ts: float
    ) -> List[dict]:
        h, w = frame_bgr.shape[:2]
        events = []
        active_dets = self.filter_in_roi(detections, w, h)

        for det in active_dets:
            tid = det.track_id
            anchor = self._get_anchor(det.bbox, w, h)
            self.triggered_tracks.setdefault(tid, set())

            if tid not in self.track_history:
                door_flags = {}
                for door in self.door_zones:
                    in_zone = self._point_in_poly(anchor, door["zone_points"], w, h)
                    door_flags[door["name"]] = {
                        "spawned_inside": in_zone,
                        "was_outside": not in_zone,
                        "currently_inside": in_zone,
                    }
                self.track_history[tid] = {
                    "first_ts": current_ts,
                    "last_ts": current_ts,
                    "prev_anchor": anchor,
                    "doors": door_flags,
                }
                continue

            state = self.track_history[tid]
            prev_anchor = state["prev_anchor"]

            for door in self.door_zones:
                d_name = door["name"]
                if d_name in self.triggered_tracks[tid]:
                    continue

                d_state = state["doors"][d_name]
                in_zone_now = self._point_in_poly(anchor, door["zone_points"], w, h)
                if not in_zone_now:
                    d_state["was_outside"] = True
                d_state["currently_inside"] = in_zone_now

                tw = door.get("tripwire")
                if tw and len(tw) == 2:
                    c = (tw[0][0] * w, tw[0][1] * h)
                    d = (tw[1][0] * w, tw[1][1] * h)
                    if self._segments_intersect(prev_anchor, anchor, c, d):
                        ev_type = self._classify_direction(prev_anchor, anchor, door.get("inside_direction", "down"))
                        self.triggered_tracks[tid].add(d_name)
                        events.append({
                            "event_type": ev_type,
                            "zone_name": d_name,
                            "track_id": tid,
                            "start_ts": state["first_ts"],
                            "trigger_ts": current_ts,
                        })
                        continue

                if d_state["spawned_inside"] and not in_zone_now:
                    self.triggered_tracks[tid].add(d_name)
                    events.append({
                        "event_type": EventType.ARRIVAL,
                        "zone_name": d_name,
                        "track_id": tid,
                        "start_ts": state["first_ts"],
                        "trigger_ts": current_ts,
                    })

            state["prev_anchor"] = anchor
            state["last_ts"] = current_ts

        events.extend(self._check_departures(current_ts))

        for asset in self.asset_zones:
            z_name = asset["name"]
            dwell_thresh = float(asset.get("dwell_threshold_sec", 4.0))
            pts = asset["points"]

            inside_tids = []
            for det in active_dets:
                if self._point_in_poly(det.center, pts, w, h):
                    inside_tids.append(det.track_id)
                    if det.track_id not in self.asset_dwell[z_name]:
                        self.asset_dwell[z_name][det.track_id] = current_ts
                    elif (current_ts - self.asset_dwell[z_name][det.track_id]) >= dwell_thresh:
                        self.asset_candidates[z_name][det.track_id] = self.asset_dwell[z_name][det.track_id]

            for tid in list(self.asset_dwell[z_name].keys()):
                if tid not in inside_tids:
                    del self.asset_dwell[z_name][tid]

            if not inside_tids:
                curr_patch = self._extract_zone_patch(frame_bgr, pts)
                if z_name not in self.asset_baselines:
                    self.asset_baselines[z_name] = curr_patch
                else:
                    if self.asset_candidates[z_name]:
                        diff = float(np.mean(np.abs(curr_patch - self.asset_baselines[z_name])))
                        if diff > 12.0:
                            last_tid, entry_ts = next(reversed(self.asset_candidates[z_name].items()))
                            events.append({
                                "event_type": EventType.POSSIBLE_THEFT,
                                "zone_name": z_name,
                                "track_id": last_tid,
                                "start_ts": entry_ts,
                                "trigger_ts": current_ts,
                            })
                        self.asset_candidates[z_name].clear()
                    self.asset_baselines[z_name] = 0.95 * self.asset_baselines[z_name] + 0.05 * curr_patch

        return events

    def _check_departures(self, current_ts: float, stale_sec: float = 1.5) -> List[dict]:
        events = []
        to_remove = []
        for tid, state in self.track_history.items():
            if (current_ts - state["last_ts"]) >= stale_sec:
                for d_name, d_state in state["doors"].items():
                    if (
                        d_name not in self.triggered_tracks.get(tid, set())
                        and d_state["was_outside"]
                        and d_state["currently_inside"]
                    ):
                        self.triggered_tracks[tid].add(d_name)
                        events.append({
                            "event_type": EventType.DEPARTURE,
                            "zone_name": d_name,
                            "track_id": tid,
                            "start_ts": max(state["first_ts"], state["last_ts"] - 2.0),
                            "trigger_ts": state["last_ts"],
                        })
                to_remove.append(tid)

        for tid in to_remove:
            del self.track_history[tid]
        return events