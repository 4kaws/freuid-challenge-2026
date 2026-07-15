"""Build final submissions (ens2, hedge) from cached crop logits.

Reproduces prepare_submission.py exactly: ids sorted lexicographically over the
union of public+private images, per-model image score = mean of top-2 crop
logits, per-model ranks (1..n)/n via mergesort argsort, weighted rank mean,
clip to [1e-6, 1-1e-6]. Output CSV: id,label sorted by id.
"""
import hashlib
import numpy as np
import pandas as pd

import os

# Cached per-crop logits (float32, one row per image, 12 columns for the 3x4 grid),
# produced by the frozen weights on an NVIDIA L4:
#   privL12_{tag}_chunkNN.npy + priv_ids.csv — Kaggle dataset junesdata/freuid-ckpts
#   pubL12_*.npy                             — same dataset (earlier version)
DPRIV = 'inputs/priv'
DPUB = 'inputs'

def find(f):
    for d in (DPRIV, DPUB):
        if os.path.exists(os.path.join(d, f)):
            return os.path.join(d, f)
    raise FileNotFoundError(f)

pub_ids = sorted(open('pub_ids.txt').read().split())
priv_ids = pd.read_csv(find('priv_ids.csv'))['id'].astype(str).tolist()  # already sorted

MODELS = {
    'pl5':  ('pubL12_pl5.npy',  'privL12_pl5_chunk{:02d}.npy'),
    'pl5m': ('pubL12_pl5m.npy', 'privL12_pl5m_chunk{:02d}.npy'),
    'pl1':  ('pubL12_pl1_f-1.npy', 'privL12_pl1_chunk{:02d}.npy'),
    'base': ('pubL12_patch512_tf_efficientnetv2_s_f-1.npy', 'privL12_base_chunk{:02d}.npy'),
}
CH = 15000

def model_scores(tag):
    pub_f, priv_pat = MODELS[tag]
    Lpub = np.load(find(pub_f))
    chunks = []
    for c in range((len(priv_ids) + CH - 1) // CH):
        chunks.append(np.load(find(priv_pat.format(c))))
    Lpriv = np.concatenate(chunks)
    assert Lpub.shape[0] == len(pub_ids) and Lpriv.shape[0] == len(priv_ids), \
        (tag, Lpub.shape, Lpriv.shape)
    s = pd.Series(
        np.concatenate([np.sort(Lpub, 1)[:, -2:].mean(1),
                        np.sort(Lpriv, 1)[:, -2:].mean(1)]).astype(np.float32),
        index=pub_ids + priv_ids)
    return s

def build(name, parts):
    all_ids = sorted(pub_ids + priv_ids)
    n = len(all_ids)
    rank_sum = np.zeros(n, np.float64)
    wsum = 0.0
    for tag, w in parts:
        logits = model_scores(tag).reindex(all_ids).to_numpy(np.float32)
        assert not np.isnan(logits).any(), tag
        order = np.argsort(logits, kind='mergesort')
        ranks = np.empty(n, np.float64)
        ranks[order] = np.arange(1, n + 1)
        rank_sum += w * ranks / n
        wsum += w
    label = np.clip(rank_sum / wsum, 1e-6, 1 - 1e-6)
    out = pd.DataFrame({'id': all_ids, 'label': label})
    out.to_csv(name, index=False)
    h = hashlib.sha256(open(name, 'rb').read()).hexdigest()
    print(f'{name}: rows={len(out)} sha256={h}')
    return h

if __name__ == '__main__':
    build('sub_final_ens2.csv', [('pl5', 1.0), ('pl5m', 1.0)])
    build('sub_final_hedge.csv', [('pl5', 1.0), ('pl5m', 1.0), ('pl1', 0.5), ('base', 0.5)])
