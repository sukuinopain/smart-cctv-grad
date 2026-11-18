import argparse
import csv
import json
import os
from pathlib import Path
import random
import re
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
import scipy.io
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor


UPPER_COLORS = ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"]
LOWER_COLORS = ["black", "white", "pink", "purple", "yellow", "gray", "blue", "green", "brown"]

HSV_REFS = {
    "black": (0, 0, 35),
    "white": (0, 0, 225),
    "gray": (0, 0, 130),
    "red": (0, 200, 180),
    "yellow": (28, 200, 200),
    "green": (60, 180, 160),
    "blue": (110, 200, 180),
    "purple": (140, 160, 160),
    "pink": (165, 130, 210),
    "brown": (15, 160, 100),
}


def load_market_annotations(mat_path: str, test_dir: str) -> List[Dict]:
    """Load official Market-1501 test identity attributes and match to test gallery images."""
    mat = scipy.io.loadmat(mat_path)
    attr_struct = mat["market_attribute"][0, 0]["test"][0, 0]
    field_names = attr_struct.dtype.names
    attr_arrays = {name: attr_struct[name].squeeze() for name in field_names}
    image_indices = [int(x) for x in attr_arrays["image_index"]]

    id_to_attrs: Dict[int, Dict] = {}
    for idx, pid in enumerate(image_indices):
        up_cols = [c for c in UPPER_COLORS if f"up{c}" in attr_arrays and int(attr_arrays[f"up{c}"][idx]) == 2]
        down_cols = [c for c in LOWER_COLORS if f"down{c}" in attr_arrays and int(attr_arrays[f"down{c}"][idx]) == 2]
        has_bag = any(
            k in attr_arrays and int(attr_arrays[k][idx]) == 2
            for k in ("backpack", "bag", "handbag")
        )
        if up_cols and down_cols:
            id_to_attrs[pid] = {
                "upper_color": up_cols[0],
                "lower_color": down_cols[0],
                "has_bag": has_bag,
            }

    samples = []
    seen_pids = set()
    for img_path in sorted(Path(test_dir).glob("*.jpg")):
        pid_str = img_path.name.split("_")[0]
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        if pid in id_to_attrs and pid not in seen_pids:
            seen_pids.add(pid)
            samples.append({"pid": pid, "path": str(img_path), **id_to_attrs[pid]})
    return samples


def hsv_predict_color(img_bgr: np.ndarray, region: str = "upper", allowed: List[str] = UPPER_COLORS) -> str:
    """Classical HSV dominant-color baseline on torso or legs crop."""
    h, w, _ = img_bgr.shape
    crop = img_bgr[int(h * 0.20):int(h * 0.52), int(w * 0.25):int(w * 0.75)] if region == "upper" else img_bgr[int(h * 0.55):int(h * 0.88), int(w * 0.25):int(w * 0.75)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    med_h, med_s, med_v = np.median(hsv, axis=0)

    if med_s < 42:
        if med_v < 70:
            cand = "black"
        elif med_v > 185:
            cand = "white"
        else:
            cand = "gray"
        if cand in allowed:
            return cand

    best_c, best_d = allowed[0], 1e9
    for c in allowed:
        rh, rs, rv = HSV_REFS[c]
        dh = min(abs(med_h - rh), 180 - abs(med_h - rh)) * 2.0
        ds = abs(med_s - rs) * 0.5
        dv = abs(med_v - rv) * 0.5
        dist = dh + ds + dv
        if dist < best_d:
            best_d, best_c = dist, c
    return best_c


def normalize_vlm_color(raw_text: str, allowed: List[str]) -> str:
    txt = raw_text.lower()
    if any(w in txt for w in ["grayscale", "black and white", "monochrome", "cannot determine", "unclear", "unknown", "no color"]):
        return "unknown_ir"
    aliases = {"grey": "gray", "dark": "black", "light": "white", "navy": "blue", "maroon": "red", "khaki": "brown", "tan": "brown"}
    for k, v in aliases.items():
        if re.search(rf"\b{k}\b", txt) and v in allowed:
            return v
    for c in allowed:
        if re.search(rf"\b{c}\b", txt):
            return c
    return "other"


def bootstrap_ci(hits: List[int], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    arr = np.array(hits, dtype=np.float32)
    rng = np.random.default_rng(seed)
    mean_val = 100.0 * float(arr.mean())
    n = len(arr)
    boots = [100.0 * float(arr[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean_val, float(lo), float(hi)


def eval_vlm_vs_hsv(samples: List[Dict], max_eval: int = 250, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    subset = rng.sample(samples, min(max_eval, len(samples)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[VLM] Loading Moondream2 on {device}...")
    vlm = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    hsv_up_hits, hsv_dn_hits = [], []
    vlm_up_hits, vlm_dn_hits = [], []
    ir_abstain_hits, hsv_ir_abstain_hits = [], []

    for i, item in enumerate(subset):
        img_bgr = cv2.imread(item["path"])
        pil_rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # 1. RGB Evaluation (HSV Baseline vs Moondream2)
        hsv_up = hsv_predict_color(img_bgr, "upper", UPPER_COLORS)
        hsv_dn = hsv_predict_color(img_bgr, "lower", LOWER_COLORS)
        hsv_up_hits.append(int(hsv_up == item["upper_color"]))
        hsv_dn_hits.append(int(hsv_dn == item["lower_color"]))

        ans_up = vlm.query(pil_rgb, "What is the main color of the person's upper clothing/shirt? Answer with one color word, or 'unknown' if grayscale/unclear.")["answer"]
        ans_dn = vlm.query(pil_rgb, "What is the main color of the person's lower clothing/pants? Answer with one color word, or 'unknown' if grayscale/unclear.")["answer"]
        vlm_up = normalize_vlm_color(ans_up, UPPER_COLORS)
        vlm_dn = normalize_vlm_color(ans_dn, LOWER_COLORS)
        vlm_up_hits.append(int(vlm_up == item["upper_color"]))
        vlm_dn_hits.append(int(vlm_dn == item["lower_color"]))

        # 2. Simulated IR (Grayscale) Hallucination Test on first 100 samples
        if i < 100:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            pil_ir = Image.fromarray(gray_bgr)

            hsv_ir = hsv_predict_color(gray_bgr, "upper", UPPER_COLORS)
            hsv_ir_abstain_hits.append(int(hsv_ir == "unknown_ir"))  # Always 0 (hallucinates black/gray/white)

            ans_ir = vlm.query(pil_ir, "This is a surveillance crop. What color is the person's shirt? If the image is black-and-white/grayscale so true color cannot be known, answer 'grayscale'.")["answer"]
            vlm_ir = normalize_vlm_color(ans_ir, UPPER_COLORS)
            ir_abstain_hits.append(int(vlm_ir == "unknown_ir"))

    del vlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = []
    for name, h_list in [
        ("HSV Baseline — Upper Color (RGB)", hsv_up_hits),
        ("Moondream2 VLM — Upper Color (RGB)", vlm_up_hits),
        ("HSV Baseline — Lower Color (RGB)", hsv_dn_hits),
        ("Moondream2 VLM — Lower Color (RGB)", vlm_dn_hits),
        ("HSV Baseline — IR Non-Hallucination Rate", hsv_ir_abstain_hits),
        ("Moondream2 VLM — IR Non-Hallucination Rate", ir_abstain_hits),
    ]:
        acc, lo, hi = bootstrap_ci(h_list)
        results.append({"metric": name, "score_pct": round(acc, 2), "ci_95": f"[{lo:.2f}, {hi:.2f}]", "n_samples": len(h_list)})
    return results


def eval_siglip_retrieval(samples: List[Dict], save_weights_dir: str) -> List[Dict]:
    """Evaluate SigLIP zero-shot text-to-crop retrieval using Hit@5, Precision@10, and nDCG@10."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/siglip-base-patch16-224"
    print(f"[SigLIP] Loading {model_id} on {device}...")
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Save local copy of SigLIP weights to Drive for offline/local deployment
    Path(save_weights_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_weights_dir)
    processor.save_pretrained(save_weights_dir)

    img_embeds = []
    batch_size = 64
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        imgs = [Image.open(x["path"]).convert("RGB") for x in batch]
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            img_embeds.append(feats.cpu())
    gallery_emb = torch.cat(img_embeds, dim=0)  # [N, D]

    # Auto-generate attribute queries that have >= 10 matching identities in gallery
    queries = []
    for up in UPPER_COLORS:
        for dn in LOWER_COLORS:
            rel = [1 if (s["upper_color"] == up and s["lower_color"] == dn) else 0 for s in samples]
            if sum(rel) >= 10:
                queries.append((f"a pedestrian wearing a {up} shirt and {dn} pants", rel))

    hit5_list, prec10_list, ndcg10_list = [], [], []
    idcg10 = sum(1.0 / np.log2(r + 2) for r in range(10))

    for q_text, rel in queries:
        t_in = processor(text=[q_text], padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            t_feat = model.get_text_features(**t_in)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        sims = (gallery_emb @ t_feat.cpu().T).squeeze(1).numpy()
        top10_idx = np.argsort(-sims)[:10]
        top10_rel = [rel[i] for i in top10_idx]

        hit5_list.append(int(sum(top10_rel[:5]) > 0))
        prec10_list.append(sum(top10_rel) / 10.0)
        dcg10 = sum(top10_rel[r] / np.log2(r + 2) for r in range(10))
        ndcg10_list.append(dcg10 / idcg10)

    return [
        {"metric": "SigLIP Retrieval — Hit@5", "score_pct": round(100.0 * float(np.mean(hit5_list)), 2), "ci_95": "N/A", "n_samples": len(queries)},
        {"metric": "SigLIP Retrieval — Precision@10", "score_pct": round(100.0 * float(np.mean(prec10_list)), 2), "ci_95": "N/A", "n_samples": len(queries)},
        {"metric": "SigLIP Retrieval — nDCG@10", "score_pct": round(100.0 * float(np.mean(ndcg10_list)), 2), "ci_95": "N/A", "n_samples": len(queries)},
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-dir", type=str, required=True)
    parser.add_argument("--mat-path", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="data/outputs/market1501_vlm_results.csv")
    parser.add_argument("--siglip-dir", type=str, default="weights/siglip-base")
    parser.add_argument("--max-vlm-samples", type=int, default=250)
    args = parser.parse_args()

    samples = load_market_annotations(args.mat_path, args.market_dir)
    print(f"[INFO] Loaded {len(samples)} unique annotated identities from Market-1501 test gallery.")

    vlm_rows = eval_vlm_vs_hsv(samples, max_eval=args.max_vlm_samples)
    siglip_rows = eval_siglip_retrieval(samples, save_weights_dir=args.siglip_dir)
    all_rows = vlm_rows + siglip_rows

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "score_pct", "ci_95", "n_samples"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== Market-1501 VLM, HSV Baseline & SigLIP Search Results ===")
    for r in all_rows:
        print(f"{r['metric']:45s} | {r['score_pct']:6.2f}% | CI: {r['ci_95']} (n={r['n_samples']})")