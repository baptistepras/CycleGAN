# CycleGAN (NumPy from scratch)

A **from-scratch** implementation of **CycleGAN** (Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, arXiv:1703.10593) written **only with NumPy** (no PyTorch / no TensorFlow).

This repository is mainly an educational project: it re-implements the paper’s building blocks (ResNet generator, 70×70 PatchGAN discriminator, LSGAN loss, cycle-consistency + identity losses) with explicit forward/backward passes.

> Disclaimer: because everything is implemented “by hand” in NumPy, training is **slow**, memory-inefficient compared to modern frameworks, and the goal is **not** state-of-the-art performance.

## Repository structure

```
CycleGAN/
├─ 1703.10593v7.pdf           # Original paper (reference)
├─ report.pdf                 # Project report / notes
├─ download_data.sh           # Dataset downloader (Berkeley mirror)
├─ train.py                   # Training script (CycleGAN loop)
├─ test.py                    # Inference + simple test metrics
├─ data.py                    # Data loading / preprocessing utilities
├─ layers.py                  # NumPy layers + autograd-style backward
├─ models.py                  # Generator / Discriminator architectures
├─ optim.py                   # Adam optimizer
└─ *.png                      # Example outputs
```

## Setup

### Requirements

- Python 3.10+ (should work with earlier 3.x too)
- NumPy
- Pillow
- tqdm
- curl + unzip (for dataset download)

Install Python dependencies:

```bash
pip install numpy pillow tqdm
```

## Download a dataset

Datasets are downloaded from the official Berkeley CycleGAN mirror.

```bash
./download_data.sh apple2orange
```

Available datasets (see `download_data.sh`):

- `apple2orange`, `summer2winter_yosemite`, `horse2zebra`
- `monet2photo`, `cezanne2photo`, `ukiyoe2photo`, `vangogh2photo`
- `maps`, `cityscapes`, `facades`, `iphone2dslr_flower`

The script creates:

```
datasets/<name>/{trainA,trainB,testA,testB}
```

## Train

Example (same pipeline / parameters as below):

```bash
python train.py \
  --data datasets/apple2orange \
  --size 64 --ngf 64 --ndf 64 --n_res 6 \
  --epochs 20 --decay_start 10 \
  --max_per_side 150 \
  --out runs/apple2orange_64
```

Notes:

- Checkpoints are saved to `--out/ckpt/` (`last.pkl` + per-epoch).
- Sample grids are periodically saved to `--out/samples/`.
- `--resume` can be used to resume from a checkpoint (`.pkl`).

## Test / inference

Generate translated images (and print simple cycle/identity L1 metrics):

```bash
python test.py \
  --ckpt runs/apple2orange_64/ckpt/last.pkl \
  --data datasets/apple2orange \
  --size 64 --ngf 64 --n_res 6 \
  --n_samples 20 --direction both \
  --out results_apple2orange
```

This writes images like:

- `results_apple2orange/AtoB_*.png` (A → B → A triplets)
- `results_apple2orange/BtoA_*.png` (B → A → B triplets)

Optional: add `--with_d` to also load discriminators and report their mean outputs.

## Implementation details (quick)

- **Generator**: Johnson-style ResNet with reflection padding + instance norm; upsampling is done with nearest-neighbor + conv (instead of transposed conv). See `models.py`.
- **Discriminator**: 70×70 **PatchGAN**. See `models.py`.
- **Losses**: **LSGAN** for GAN losses, plus cycle-consistency and identity losses. See `train.py`.

## License / usage

This code is provided for learning and experimentation. If you use it for anything beyond that, please cite the original CycleGAN paper.
