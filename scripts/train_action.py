import argparse
from pathlib import Path
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18


def letterbox_112(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = 112.0 / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((112, 112, 3), dtype=np.uint8)
    top = (112 - nh) // 2
    left = (112 - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class RWFVideoDataset(Dataset):
    """Loads RWF-2000 clips sampled at ~8 FPS (stride=4 on 30 FPS) to match live inference."""

    def __init__(self, root_dir: str, split: str = "train", clip_len: int = 16, target_fps: int = 8):
        self.samples = []
        self.is_train = split == "train"
        self.clip_len = clip_len
        self.target_fps = target_fps

        split_path = Path(root_dir) / split
        for label_idx, cls_name in enumerate(["NonFight", "Fight"]):
            cls_dir = split_path / cls_name
            if not cls_dir.exists():
                continue
            for ext in ("*.avi", "*.mp4", "*.mkv"):
                for vid in cls_dir.glob(ext):
                    self.samples.append((str(vid), label_idx))

        if not self.samples:
            raise RuntimeError(f"No videos found under {split_path}/[Fight|NonFight]")

        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_sampled_clip(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, round(src_fps / self.target_fps))

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                frames.append(letterbox_112(frame))
            idx += 1
        cap.release()

        if not frames:
            return np.zeros((self.clip_len, 112, 112, 3), dtype=np.uint8)

        # Pad if shorter than clip_len
        while len(frames) < self.clip_len:
            frames.append(frames[-1])

        # Random 2.0s temporal crop during training; center crop during validation
        max_start = len(frames) - self.clip_len
        start = random.randint(0, max_start) if self.is_train else (max_start // 2)
        return np.stack(frames[start:start + self.clip_len], axis=0)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        clip = self._read_sampled_clip(path)  # (T, H, W, C)

        if self.is_train:
            if random.random() < 0.5:
                clip = clip[:, :, ::-1, :].copy()  # Horizontal flip
            if random.random() < 0.5:
                # Simulate IR Night Vision grayscale 50% of the time
                gray = np.mean(clip, axis=-1, keepdims=True).astype(np.uint8)
                clip = np.repeat(gray, 3, axis=-1)

        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor, torch.tensor(label, dtype=torch.long)


def train(data_dir: str, out_weights: str, epochs: int = 12, batch_size: int = 16, lr: float = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = RWFVideoDataset(data_dir, split="train")
    val_ds = RWFVideoDataset(data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    Path(out_weights).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = 100.0 * train_correct / max(1, total)
        val_acc = 100.0 * val_correct / max(1, val_total)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss/total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_weights)
            print(f"  -> Saved new best weights ({best_acc:.2f}%) to {out_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to RWF-2000 root folder")
    parser.add_argument("--out", type=str, default="weights/fight_r3d18.pth")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.data_dir, args.out, args.epochs, args.batch_size)