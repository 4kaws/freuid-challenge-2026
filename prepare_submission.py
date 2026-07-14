#!/usr/bin/env python3
"""FREUID Challenge 2026 - inference entrypoint (team junesdata).

Organizer sandbox contract:
  /data/         read-only, flat directory of test images ({id}.jpeg etc.)
  /submissions/  read-write; must contain submission.csv on exit

Pipeline (frozen weights, inference only):
  - each image is covered by an overlapping ROWS x COLS grid of 512x512 crops
    at native resolution (no resizing; small images are zero-padded);
  - per model, the image score is the mean of the top-K crop logits;
  - the final score is the mean of the per-model ranks, normalized to [0, 1].
"""
import argparse, os, sys
from pathlib import Path

import cv2, numpy as np, pandas as pd, timm, torch
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(os.environ.get("FREUID_DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("FREUID_OUTPUT_DIR", "/submissions"))
SUBMISSION_PATH = Path(os.environ.get("FREUID_SUBMISSION_PATH", OUTPUT_DIR / "submission.csv"))
WEIGHTS_DIR = Path(os.environ.get("FREUID_WEIGHTS_DIR", "/app/weights"))
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CROP = 512
TOPK = 2
BATCH_IMAGES = 8

# Inference variants (one image, one frozen weight set; selected via env flag):
#   VARIANT=ens2  (default) - Pick 1: 2-model rank ensemble, public-LB optimized
#   VARIANT=hedge           - Pick 2: adds non-pseudo-labeled models for OOD robustness
VARIANT = os.environ.get("FREUID_VARIANT", os.environ.get("VARIANT", "ens2"))
VARIANTS = {
    "ens2": {"rows": 5, "cols": 6,
             "models": [("pl5_effnetv2s.pt", 1.0), ("pl5m_effnetv2m.pt", 1.0)]},
    "hedge": {"rows": 3, "cols": 4,
              "models": [("pl5_effnetv2s.pt", 1.0), ("pl5m_effnetv2m.pt", 1.0),
                         ("pl1_effnetv2s.pt", 0.5), ("base_effnetv2s.pt", 0.5)]},
}
_cfg = VARIANTS[VARIANT]
ROWS, COLS = _cfg["rows"], _cfg["cols"]
MODELS = _cfg["models"]

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def discover_images(data_dir: Path):
    pairs = [(p.stem, p) for p in sorted(data_dir.iterdir())
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not pairs:
        raise FileNotFoundError(f"No images found in {data_dir}")
    return pairs


class GridDS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"unreadable image: {self.paths[i]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h < CROP or w < CROP:
            img = cv2.copyMakeBorder(img, 0, max(0, CROP - h), 0, max(0, CROP - w),
                                     cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape[:2]
        ys = np.linspace(0, h - CROP, ROWS).round().astype(int)
        xs = np.linspace(0, w - CROP, COLS).round().astype(int)
        crops = []
        for y in ys:
            for x in xs:
                c = img[y:y + CROP, x:x + CROP].astype(np.float32) / 255.0
                crops.append(torch.from_numpy(((c - MEAN) / STD).transpose(2, 0, 1)))
        return torch.stack(crops), i


def safe_workers() -> int:
    """DataLoader workers need /dev/shm; fall back to 0 when it is tiny (default
    Docker gives 64 MB, which crashes workers with SIGBUS)."""
    try:
        import shutil
        shm_free = shutil.disk_usage("/dev/shm").free
    except OSError:
        return 0
    if shm_free < 1 << 30:  # < 1 GiB
        return 0
    return min(8, os.cpu_count() or 1)


@torch.no_grad()
def score_model(weight_path: Path, paths, device):
    ck = torch.load(weight_path, map_location="cpu")
    # older checkpoints (pl1/base, July 11) lack the model_name key; both are V2-S
    name = ck.get("model_name", "tf_efficientnetv2_s.in21k_ft_in1k")
    model = timm.create_model(name, pretrained=False, num_classes=1)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.to(device).eval()
    dl = DataLoader(GridDS(paths), batch_size=BATCH_IMAGES,
                    num_workers=safe_workers(), pin_memory=device == "cuda")
    out = np.zeros(len(paths), np.float32)
    autocast = torch.autocast(device) if device == "cuda" else torch.no_grad()
    with autocast:
        for crops, idx in dl:
            b, n, c, h, w = crops.shape
            logits = model(crops.view(b * n, c, h, w).to(device, non_blocking=True))
            logits = logits.view(b, n).float()
            out[idx.numpy()] = logits.topk(TOPK, dim=1).values.mean(1).cpu().numpy()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def predict_labels(image_rows):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"variant={VARIANT} device={device} images={len(image_rows)} "
          f"grid={ROWS}x{COLS} topk={TOPK}", file=sys.stderr)
    ids = [rid for rid, _ in image_rows]
    paths = [p for _, p in image_rows]
    n = len(paths)
    rank_sum = np.zeros(n, np.float64)
    wsum = 0.0
    for name, weight in MODELS:
        logits = score_model(WEIGHTS_DIR / name, paths, device)
        order = np.argsort(logits, kind="mergesort")
        ranks = np.empty(n, np.float64)
        ranks[order] = np.arange(1, n + 1)
        rank_sum += weight * ranks / n
        wsum += weight
        print(f"model {name} (w={weight}) done", file=sys.stderr)
    label = np.clip(rank_sum / wsum, 1e-6, 1 - 1e-6)
    return pd.DataFrame({"id": ids, "label": label})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    args = ap.parse_args()
    image_rows = discover_images(args.data_dir.resolve())
    submission = predict_labels(image_rows)
    expected = {rid for rid, _ in image_rows}
    got = set(submission["id"].astype(str))
    if expected != got:
        raise ValueError(f"id mismatch: missing={len(expected-got)} extra={len(got-expected)}")
    if not np.isfinite(submission["label"].to_numpy(float)).all():
        raise ValueError("non-finite labels")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
