"""Native-res 512px patch training. Usage:
python train_patch.py --fold 2 --epochs 4 --aug recap --tag recap_f2 [--pl pl.csv]
fold -1 = train on all types (no val)."""
import argparse, os, sys, time, random
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fr_common import (DATA, MODELS, load_folds, imread, pad_to, norm_tensor,
                       freuid_components, predict_grid)

def jpeg(img, q):
    return cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)

def recapture_aug(img, rng):
    """Simulate screen/print recapture: resample, moire, noise, recompress."""
    h, w = img.shape[:2]
    if rng.random() < 0.9:  # down-up resample
        s = rng.uniform(0.45, 0.9)
        interp = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA])
        img = cv2.resize(cv2.resize(img, (int(w*s), int(h*s)), interpolation=interp),
                         (w, h), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if rng.random() < 0.5:  # moire-like sinusoidal luminance pattern
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = rng.uniform(0.2, 1.5); ang = rng.uniform(0, np.pi)
        pat = np.sin(2*np.pi*f*(xx*np.cos(ang) + yy*np.sin(ang)) / 8.0)
        img = np.clip(img.astype(np.float32) + pat[..., None]*rng.uniform(2, 8), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:  # sensor noise
        img = np.clip(img.astype(np.float32) + np.random.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.5:  # gamma / color balance drift
        g = rng.uniform(0.75, 1.3)
        img = np.clip(255.0*(img.astype(np.float32)/255.0)**g, 0, 255).astype(np.uint8)
    if rng.random() < 0.9:  # recompress
        img = jpeg(img, int(rng.integers(45, 92)))
    return img

class PatchDS(Dataset):
    def __init__(self, df, size=512, aug='base', root=DATA):
        self.df = df.reset_index(drop=True); self.size = size; self.aug = aug; self.root = root
    def __len__(self):
        return len(self.df)
    def path(self, r):
        p = r.image_path
        if p.startswith('train/'):
            return f'{self.root}/train/{p}'
        return f'{self.root}/public_test/public_test/{p}'
    def __getitem__(self, i):
        r = self.df.iloc[i]
        rng = np.random.default_rng()
        img = pad_to(imread(self.path(r)), self.size)
        h, w = img.shape[:2]
        y = rng.integers(0, h - self.size + 1); x = rng.integers(0, w - self.size + 1)
        crop = img[y:y+self.size, x:x+self.size]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
        if rng.random() < 0.5:
            crop = crop[::-1]
        if self.aug == 'recap' and rng.random() < 0.3:
            crop = recapture_aug(np.ascontiguousarray(crop), rng)
        elif self.aug == 'base' and rng.random() < 0.3:
            crop = jpeg(np.ascontiguousarray(crop), int(rng.integers(55, 95)))
        return norm_tensor(np.ascontiguousarray(crop)), np.float32(r.label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--aug', default='base')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--pl', default='')  # csv: id,label -> public_test pseudo-labels
    ap.add_argument('--model', default='tf_efficientnetv2_s.in21k_ft_in1k')
    a = ap.parse_args()

    df = load_folds()
    tr = df[df.fold != a.fold] if a.fold >= 0 else df
    va = df[df.fold == a.fold] if a.fold >= 0 else None
    if a.pl:
        pl = pd.read_csv(a.pl)
        pl['image_path'] = pl['id'] + '.jpeg'
        pl['fold'] = -9
        tr = pd.concat([tr, pl[['id', 'image_path', 'label', 'fold']]], ignore_index=True)
    print(f'train={len(tr)} val={0 if va is None else len(va)} aug={a.aug}', flush=True)

    tdl = DataLoader(PatchDS(tr, aug=a.aug), batch_size=a.bs, shuffle=True, num_workers=8,
                     pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2)
    model = timm.create_model(a.model, pretrained=True, num_classes=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=len(tdl)*a.epochs, pct_start=0.1)
    scaler = torch.amp.GradScaler(); crit = nn.BCEWithLogitsLoss()
    best = 9.9
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for i, (x, y) in enumerate(tdl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast('cuda'):
                loss = crit(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run = 0.98*run + 0.02*loss.item() if i else loss.item()
            if i % 200 == 0:
                print(f'ep{ep} it{i}/{len(tdl)} loss={run:.4f} {time.time()-t0:.0f}s', flush=True)
        score = -1.0
        if va is not None:
            model.eval()
            paths = [f'{DATA}/train/{p}' for p in va.image_path]
            logit = predict_grid(model, paths)
            comp = freuid_components(va.label.values, logit)
            score = comp['freuid']
            print(f"ep{ep} VAL freuid={score:.5f} audet={comp['audet']:.5f} apcer={comp['apcer@1%bpcer']:.5f}", flush=True)
        ck = {'model_name': a.model, 'size': 512, 'model': model.state_dict()}
        torch.save(ck, f'{MODELS}/{a.tag}_last.pt')
        if va is None or score < best:
            best = score
            torch.save(ck, f'{MODELS}/{a.tag}_best.pt')
    print(f'DONE {a.tag} best={best}', flush=True)

if __name__ == '__main__':
    main()
