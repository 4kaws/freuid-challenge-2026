"""Shared: metric, folds, patch dataset, crop-grid inference."""
import os, glob, math
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader

DATA = '/content/freuid/data'
MODELS = '/content/freuid/models'
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# ---------- metric (verified vs kernel irelic1/freuid-score) ----------
def det_curve(y_true, y_score):
    n_pos = int((y_true == 1).sum()); n_neg = int((y_true == 0).sum())
    order = np.argsort(-y_score, kind='mergesort')
    s_sorted = y_score[order]; y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted == 1); fp_cum = np.cumsum(y_sorted == 0)
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tp_cum = tp_cum[distinct]; fp_cum = fp_cum[distinct]
    bpcer = fp_cum / n_neg; apcer = 1.0 - tp_cum / n_pos
    return np.concatenate(([0.0], bpcer)), np.concatenate(([1.0], apcer))

def freuid_components(y_true, y_score, bpcer_target=0.01):
    y_true = np.asarray(y_true).astype(int); y_score = np.asarray(y_score, float)
    bpcer, apcer = det_curve(y_true, y_score)
    a = float(np.trapz(apcer, bpcer))
    feasible = bpcer <= bpcer_target + 1e-12
    p = float(apcer[np.flatnonzero(feasible).max()]) if feasible.any() else 1.0
    g_a, g_p = 1 - a, 1 - p
    f = 1.0 if g_a + g_p <= 0 else 1.0 - 2 * g_a * g_p / (g_a + g_p)
    return {'audet': a, 'apcer@1%bpcer': p, 'freuid': f}

# ---------- folds (leave-one-type-out; must match ckpt fold ids) ----------
def load_folds():
    from sklearn.model_selection import GroupKFold
    df = pd.read_csv(f'{DATA}/train_labels.csv')
    gkf = GroupKFold(n_splits=5)
    df['fold'] = -1
    for i, (_, vi) in enumerate(gkf.split(df, groups=df['type'])):
        df.loc[vi, 'fold'] = i
    return df

def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def pad_to(img, size):
    h, w = img.shape[:2]
    if h >= size and w >= size:
        return img
    return cv2.copyMakeBorder(img, 0, max(0, size - h), 0, max(0, size - w),
                              cv2.BORDER_CONSTANT, value=0)

def norm_tensor(crop):
    x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))

# ---------- grid-crop inference dataset ----------
def grid_offsets(h, w, size, rows, cols):
    ys = np.linspace(0, max(0, h - size), rows).round().astype(int)
    xs = np.linspace(0, max(0, w - size), cols).round().astype(int)
    return [(y, x) for y in ys for x in xs]

class GridDS(Dataset):
    """Yields all grid crops of one image per index; collate stacks them."""
    def __init__(self, paths, size=512, rows=3, cols=4):
        self.paths, self.size, self.rows, self.cols = paths, size, rows, cols
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = pad_to(imread(self.paths[i]), self.size)
        h, w = img.shape[:2]
        crops = [norm_tensor(img[y:y+self.size, x:x+self.size])
                 for y, x in grid_offsets(h, w, self.size, self.rows, self.cols)]
        return torch.stack(crops), i

def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    name = ck.get('model_name', 'tf_efficientnetv2_s.in21k_ft_in1k')
    m = timm.create_model(name, pretrained=False, num_classes=1)
    m.load_state_dict(ck['model'] if 'model' in ck else ck)
    return m.cuda().eval(), ck

@torch.no_grad()
def predict_grid(model, paths, size=512, rows=3, cols=4, topk=2, bs=8, workers=8):
    """Returns per-image aggregated logit (mean of top-k crop logits)."""
    dl = DataLoader(GridDS(paths, size, rows, cols), batch_size=bs,
                    num_workers=workers, pin_memory=True)
    out = np.zeros(len(paths), np.float32)
    with torch.autocast('cuda'):
        for crops, idx in dl:
            b, n, c, h, w = crops.shape
            logits = model(crops.view(b * n, c, h, w).cuda(non_blocking=True))
            logits = logits.view(b, n).float()
            agg = logits.topk(min(topk, n), dim=1).values.mean(1)
            out[idx.numpy()] = agg.cpu().numpy()
    return out
