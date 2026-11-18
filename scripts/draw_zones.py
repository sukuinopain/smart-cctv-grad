import argparse
import json
from pathlib import Path
import cv2


def save_grid_image(video_path: str, out_path: str = "data/outputs/zone_grid.jpg") -> None:
    """Saves the first frame overlaid with a normalized 0.0-1.0 coordinate grid (Colab friendly)."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame from {video_path}")

    h, w = frame.shape[:2]
    for i in range(1, 10):
        val = i / 10.0
        x = int(w * val)
        y = int(h * val)
        cv2.line(frame, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.line(frame, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(frame, f"{val:.1f}", (x + 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"{val:.1f}", (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"Saved coordinate grid reference to: {out_path}")


def interactive_draw(video_path: str, config_path: str = "configs/camera_zones.json") -> None:
    """Local OpenCV GUI tool: Click 4 points for Door Zone, then 2 points for Tripwire."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame from {video_path}")

    h, w = frame.shape[:2]
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0) if len(points) <= 4 else (0, 0, 255), -1)
            cv2.imshow("Draw Zones (4 Door Pts -> 2 Tripwire Pts | Press S to Save)", frame)

    cv2.imshow("Draw Zones (4 Door Pts -> 2 Tripwire Pts | Press S to Save)", frame)
    cv2.setMouseCallback("Draw Zones (4 Door Pts -> 2 Tripwire Pts | Press S to Save)", on_mouse)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and len(points) == 6:
            norm_pts = [[round(x / w, 3), round(y / h, 3)] for x, y in points]
            cfg = {
                "roi_polygons": [{"name": "full_scene", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}],
                "door_zones": [{
                    "name": "main_door",
                    "zone_points": norm_pts[:4],
                    "tripwire": norm_pts[4:6],
                    "inside_direction": "down"
                }],
                "asset_zones": []
            }
            Path(config_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"Saved normalized zones to {config_path}")
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--grid", action="store_true", help="Export static grid image for headless/Colab use")
    args = parser.parse_args()

    if args.grid:
        save_grid_image(args.source)
    else:
        interactive_draw(args.source)
