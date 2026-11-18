import argparse
import time
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
import torch.nn as nn
from scripts.train_action import RWFVideoDataset
from src.pipeline import SmartCCTVPipeline


def benchmark_pipeline(video_path: str) -> None:
    """Measures end-to-end throughput (FPS) and real-time factor of the 4-stage pipeline."""
    cap = cv2.VideoCapture(video_path)
    total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    video_duration_sec = total_raw_frames / src_fps
    cap.release()

    print(f"\n=== Pipeline Speed Benchmark ===")
    print(f"Video: {video_path} | Duration: {video_duration_sec:.2f}s ({total_raw_frames} frames @ {src_fps:.1f} FPS)")

    pipeline = SmartCCTVPipeline(enable_vlm=False)
    t0 = time.perf_counter()
    pipeline.run(video_path)
    elapsed = time.perf_counter() - t0

    effective_fps = total_raw_frames / max(1e-6, elapsed)
    realtime_speedup = video_duration_sec / max(1e-6, elapsed)

    print(f"Total Processing Time : {elapsed:.2f}s")
    print(f"Effective Throughput  : {effective_fps:.2f} FPS")
    print(f"Real-Time Speed Factor: {realtime_speedup:.2f}x faster than real-time\n")


def evaluate_fight_model(data_dir: str, weights_path: str = "weights/fight_r3d18.pth") -> None:
    """Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix on RWF-2000 val split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds = RWFVideoDataset(data_dir, split="val")
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    tp = tn = fp = fn = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            tp += int(((preds == 1) & (y == 1)).sum().item())
            tn += int(((preds == 0) & (y == 0)).sum().item())
            fp += int(((preds == 1) & (y == 0)).sum().item())
            fn += int(((preds == 0) & (y == 1)).sum().item())

    total = max(1, tp + tn + fp + fn)
    acc = 100.0 * (tp + tn) / total
    precision = 100.0 * tp / max(1, tp + fp)
    recall = 100.0 * tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    print("\n=== R3D-18 Fight Classifier Evaluation (RWF-2000 Val) ===")
    print(f"Accuracy : {acc:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall   : {recall:.2f}%")
    print(f"F1-Score : {f1:.2f}%")
    print(f"Confusion Matrix -> TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Run speed benchmark on a video file")
    parser.add_argument("--rwf-val", type=str, help="Path to RWF-2000 root dir for classification metrics")
    parser.add_argument("--weights", type=str, default="weights/fight_r3d18.pth")
    args = parser.parse_args()

    if args.video:
        benchmark_pipeline(args.video)
    if args.rwf_val:
        evaluate_fight_model(args.rwf_val, args.weights)