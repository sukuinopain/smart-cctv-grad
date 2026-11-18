import argparse
import csv
import json
import os
from pathlib import Path
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R3D_18_Weights, r3d_18


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AblationRWFDataset(Dataset):
    """Deterministic loader supporting controlled RGB vs. Simulated IR evaluation."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 16,
        gray_prob: float = 0.0,
        force_grayscale: bool = False,
    ):
        self.root = Path(root_dir) / split
        self.split = split
        self.num_frames = num_frames
        self.gray_prob = gray_prob
        self.force_grayscale = force_grayscale
        self.samples = []
        for label_idx, cls_name in enumerate(["NonFight", "Fight"]):
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for p in sorted(cls_dir.glob("*")):
                    if p.suffix.lower() in {".avi", ".mp4", ".mkv"}:
                        self.samples.append((str(p), label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(0, total - 1), self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((112, 112, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.stack(frames, axis=0)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        clip = self._load_clip(path)

        apply_gray = self.force_grayscale or (
            self.split == "train" and random.random() < self.gray_prob
        )
        if apply_gray:
            gray = np.dot(clip[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            clip = np.stack([gray, gray, gray], axis=-1)

        if self.split == "train" and random.random() < 0.5:
            clip = np.ascontiguousarray(clip[:, :, ::-1, :])

        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, torch.tensor(label, dtype=torch.long)


def build_model(device: str) -> nn.Module:
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def train_model(
    data_dir: str,
    save_path: str,
    checkpoint_path: str,
    gray_prob: float,
    epochs: int = 12,
    batch_size: int = 16,
    seed: int = 42,
) -> None:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = AblationRWFDataset(data_dir, split="train", gray_prob=gray_prob)
    val_ds = AblationRWFDataset(data_dir, split="val", force_grayscale=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_acc = 0.0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        print(f"[RESUME] Resuming from epoch {start_epoch} (best_val_acc={best_val_acc:.2f}%)")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = 100.0 * val_correct / max(1, val_total)
        print(f"[gray_p={gray_prob}] Epoch {epoch:02d}/{epochs:02d} | Train: {100.0*correct/max(1,total):.2f}% | Val RGB: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best weights ({val_acc:.2f}%) to {save_path}")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            },
            checkpoint_path,
        )


def get_predictions(model_path: str, loader: DataLoader, device: str):
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    preds_all, targets_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            preds_all.append(preds)
            targets_all.append(y.numpy())
    return np.concatenate(preds_all), np.concatenate(targets_all)


def bootstrap_ci(preds: np.ndarray, targets: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    correct = (preds == targets).astype(np.float32)
    acc = 100.0 * correct.mean()
    n = len(correct)
    boot_means = [100.0 * correct[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)]
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return acc, float(low), float(high)


def mcnemar_exact_p(preds_a: np.ndarray, preds_b: np.ndarray, targets: np.ndarray) -> Tuple[int, int, float]:
    from math import comb
    a_ok = preds_a == targets
    b_ok = preds_b == targets
    b = int(np.sum(a_ok & ~b_ok))  # A right, B wrong
    c = int(np.sum(~a_ok & b_ok))  # A wrong, B right
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p_val = 2.0 * sum(comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return b, c, min(1.0, float(p_val))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--aug-weights", type=str, default="weights/fight_r3d18.pth")
    parser.add_argument("--rgb-weights", type=str, default="weights/fight_r3d18_rgb_only.pth")
    parser.add_argument("--ckpt-dir", type=str, default="weights/checkpoints")
    parser.add_argument("--out-csv", type=str, default="data/outputs/ir_ablation_results.csv")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    # 1. Train RGB-only baseline (0% grayscale augmentation) if not already trained
    if not os.path.exists(args.rgb_weights):
        print("\n=== Training Baseline Model (0% Grayscale Augmentation) ===")
        train_model(
            data_dir=args.data_dir,
            save_path=args.rgb_weights,
            checkpoint_path=os.path.join(args.ckpt_dir, "rgb_only_last.pt"),
            gray_prob=0.0,
            epochs=args.epochs,
            seed=args.seed,
        )
    else:
        print(f"[INFO] Found existing RGB-only baseline weights at {args.rgb_weights}")

    # 2. Evaluate both models on RGB Val and Simulated IR (Grayscale) Val
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_rgb_loader = DataLoader(AblationRWFDataset(args.data_dir, split="val", force_grayscale=False), batch_size=16, shuffle=False, num_workers=2)
    val_ir_loader = DataLoader(AblationRWFDataset(args.data_dir, split="val", force_grayscale=True), batch_size=16, shuffle=False, num_workers=2)

    rows = []
    for mode_name, loader in [("RGB (Daylight)", val_rgb_loader), ("Simulated IR (Grayscale)", val_ir_loader)]:
        preds_rgb_model, targets = get_predictions(args.rgb_weights, loader, device)
        preds_aug_model, _ = get_predictions(args.aug_weights, loader, device)

        acc_rgb, lo_rgb, hi_rgb = bootstrap_ci(preds_rgb_model, targets)
        acc_aug, lo_aug, hi_aug = bootstrap_ci(preds_aug_model, targets)
        b, c, p_val = mcnemar_exact_p(preds_aug_model, preds_rgb_model, targets)

        print(f"\n--- Evaluation on {mode_name} ---")
        print(f"Baseline (0% Gray Aug) : {acc_rgb:.2f}% (95% CI: [{lo_rgb:.2f}%, {hi_rgb:.2f}%])")
        print(f"Ours (50% Gray Aug)    : {acc_aug:.2f}% (95% CI: [{lo_aug:.2f}%, {hi_aug:.2f}%])")
        print(f"Delta                  : {acc_aug - acc_rgb:+.2f}% | McNemar (b={b}, c={c}, p={p_val:.4f})")

        rows.append({
            "eval_domain": mode_name,
            "baseline_acc": round(acc_rgb, 2),
            "baseline_95ci": f"[{lo_rgb:.2f}, {hi_rgb:.2f}]",
            "ours_aug_acc": round(acc_aug, 2),
            "ours_95ci": f"[{lo_aug:.2f}, {hi_aug:.2f}]",
            "delta_acc": round(acc_aug - acc_rgb, 2),
            "mcnemar_b_ours_only": b,
            "mcnemar_c_base_only": c,
            "mcnemar_p_value": round(p_val, 5),
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SAVED] Full ablation table with 95% CIs and McNemar p-values saved to {args.out_csv}")