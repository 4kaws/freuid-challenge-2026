# FREUID Challenge 2026 — Identity Document Fraud Detection

Solution by team **junesdata** for [The FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai).

## Method overview

Fraud in this dataset is dominated by **local, pixel-level manipulation**, so the
solution avoids full-image downscaling entirely:

1. **Native-resolution patch classifiers.** EfficientNetV2-S and EfficientNetV2-M
   (timm, ImageNet-21k pretrained) are fine-tuned as binary classifiers on random
   512x512 crops taken from the original-resolution document images (no resizing).
2. **Recapture augmentation.** A screen/print recapture simulation (down/up
   resampling, moire-like luminance patterns, sensor noise, gamma drift, JPEG
   recompression) is applied to 30% of training crops, targeting the
   capture-pipeline shift emphasized by the test set.
3. **Iterative pseudo-labeling on the public test images** (competition data only,
   no external data). Confident predictions (fraud: p > 0.98; genuine: lowest-ranked
   1150 images) are added as pseudo-labeled training data and the model is retrained.
   Five rounds were run, each labeled by the previous round's best public-LB model.
4. **Grid-crop max-style inference.** Each test image is covered by an overlapping
   grid of 512x512 crops at native resolution; the per-image score is the mean of the
   top-2 crop logits (fraud is local, so max-pooling across crops is essential).
5. **Two-model rank ensemble.** The final score is the mean of the two models'
   per-image rank (rank-normalized over the full test set). The FREUID metric is
   ranking-based, so rank averaging is scale-free.

Training data: only the official competition training set (69,352 images, 5 document
types) plus pseudo-labels on the official public test images. Pretrained backbones:
publicly available timm ImageNet checkpoints.

## Repository layout

| Path | Purpose |
| ---- | ------- |
| `prepare_submission.py` | Inference entrypoint (organizer sandbox contract) |
| `src/fr_common.py` | Metric, dataset, grid-crop inference utilities |
| `src/train_patch.py` | Patch-classifier training (incl. recapture augmentation and pseudo-label ingestion) |
| `weights/` | Frozen model weights (Git LFS): `pl5_effnetv2s.pt`, `pl5m_effnetv2m.pt` |
| `docker/` | Dockerfile + requirements for the no-network verification sandbox |
| `report/` | Technical report (PDF) |

## Reproducing the submission

```bash
docker build -f docker/Dockerfile -t freuid-solution .

docker run --rm --network none --gpus all \
  -v /path/to/test_images:/data:ro \
  -v $(pwd)/_local_out:/submissions \
  freuid-solution
```

`/data` must be a flat directory of test images; the container writes
`/submissions/submission.csv` with schema `id,label` (higher label = more likely fraud).

Hardware: single GPU with >= 16 GB VRAM; tested targeting one NVIDIA A100 40GB
(see technical report for measured runtime).

## Training (for provenance)

```bash
# folds + base model (all 5 document types)
python src/train_patch.py --fold -1 --epochs 4 --bs 48 --aug recap --tag base_f-1
# pseudo-label round k (pl_k.csv built from the previous round's public-test scores)
python src/train_patch.py --fold -1 --epochs 4 --bs 48 --aug recap --tag pl5_f-1 --pl pl5.csv
python src/train_patch.py --fold -1 --epochs 4 --bs 24 --lr 2e-4 --aug recap \
  --model tf_efficientnetv2_m.in21k_ft_in1k --tag pl5m_f-1 --pl pl5.csv
```

Model weights in `weights/` were finalized and archived before the private test
release on 2026-07-13 (see Kaggle dataset `junesdata/freuid-ckpts` version history
for timestamps).

## License

MIT — see `LICENSE`.
