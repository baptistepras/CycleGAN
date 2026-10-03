# Usage

Setup, then every command with its options and outputs. How the code works is in [implementation.md](implementation.md).

## Setup

The project has its own environment, `cyclegan-numpy`, defined in [`environment.yml`](../environment.yml).

```bash
mamba env create -f environment.yml          # create the environment once
mamba activate cyclegan-numpy                # activate it in every new terminal
mamba env update -f environment.yml --prune  # after environment.yml changes
```

Every command below runs from the project root.

## Commands

| Command | What it does | Outputs |
| --- | --- | --- |
| `./download_data.sh apple2orange` | Downloads a dataset from the official Berkeley mirror. Also available: `horse2zebra`, `summer2winter_yosemite`, `monet2photo`, `cezanne2photo`, `ukiyoe2photo`, `vangogh2photo`, `maps`, `cityscapes`, `facades`, `iphone2dslr_flower`. | `datasets/<name>/{trainA,trainB,testA,testB}` |
| `python train.py --data datasets/apple2orange --n_res 6 --max_per_side 150 --out runs/apple2orange_64` | Trains CycleGAN with the setup of our results. | A checkpoint per epoch and `last.pkl` in `<out>/ckpt/`, sample grids every `--sample_every` steps in `<out>/samples/`. |
| `python test.py --ckpt runs/apple2orange_64/ckpt/last.pkl --data datasets/apple2orange --n_res 6 --out results_apple2orange` | Translates test images and measures cycle consistency. | `AtoB_*.png` and `BtoA_*.png` strips in `--out`, and the cycle and identity ℓ1 errors in the terminal. |

## Options

**`train.py`**: `--data`, `--size` (64), `--ngf` (64), `--ndf` (64), `--n_res` (3), `--epochs` (20), `--decay_start` (10), `--lr` (2e-4), `--lambda_cyc` (10), `--lambda_id` (0.5), `--max_per_side` (all images), `--out`, `--sample_every` (200), `--seed` (0), `--resume` (a checkpoint to restart from).

**`test.py`**: `--ckpt` (required), `--data`, `--size`, `--ngf`, `--ndf`, `--n_res`, `--out`, `--n_samples` (20), `--direction` (`AB`, `BA`, or `both`), `--with_d` (also load the discriminators and report their mean outputs).

`--size`, `--ngf`, `--ndf`, and `--n_res` must match between training and testing.
