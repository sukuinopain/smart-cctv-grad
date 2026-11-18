import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from src.schemas import Detection, EventType


class SpatialRuleEngine:
    """Evaluates ROI filtering and Arrival/Departure using Tripwires + Door Zones."""

    def __init__(self, zones_path: str = "configs/camera_zones.json"):
        path = Path(zones_path)
        self.cfg = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        self.roi_polygons = self.cfg.get("roi_polygons", [])
        self.door_zones = self.cfg.get("door_zones", [])

        self.track_history: Dict[int, dict] = {}
        self.triggered_tracks: Dict[int, set] = {}

    @staticmethod
    def _get_anchor(bbox: Tuple[int, int, int, int], w: int, h: int) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        # Fallback to box center if feet are clipped at the bottom edge of the frame
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

    def filter_in_roi(self, detections: List[Detection], w: int, h: int) -> List[Detection]:
        if not self.roi_polygons:
            return detections
        filtered = []
        for det in detections:
            anchor = self._get_anchor(det.bbox, w, h)
            if any(self._point_in_poly(anchor, roi["points"], w, h) for roi in self.roi_polygons):
                filtered.append(det)
        return filtered

    def update(self, detections: List[Detection], w: int, h: int, current_ts: float) -> List[dict]:
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

                # 1. Check Tripwire Crossing
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

                # 2. Fallback: Spawned inside door zone and stepped into the room -> Arrival
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

        # 3. Check stale tracks that disappeared inside a door zone -> Departure
        events.extend(self._check_departures(current_ts))
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

