"""Generate translated images and compute test-set metrics"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from data import load_image, to_uint8
from models import Discriminator, Generator
from train import load_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to checkpoint .pkl")
    p.add_argument("--data", default="datasets/horse2zebra")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_res", type=int, default=3)
    p.add_argument("--out", default="results")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--direction", choices=["AB", "BA", "both"], default="both")
    p.add_argument("--with_d", action="store_true", help="also load discriminators and report scores")
    args = p.parse_args()

    G = Generator(3, 3, args.ngf, args.n_res)
    F_ = Generator(3, 3, args.ngf, args.n_res)
    if args.with_d:
        DA = Discriminator(3, args.ndf)
        DB = Discriminator(3, args.ndf)
        load_ckpt(args.ckpt, G=G, F=F_, DA=DA, DB=DB)
    else:
        load_ckpt(args.ckpt, G=G, F=F_)

    os.makedirs(args.out, exist_ok=True)

    files_A = sorted(Path(f"{args.data}/testA").iterdir())
    files_B = sorted(Path(f"{args.data}/testB").iterdir())

    cyc_a, cyc_b, id_a, id_b = [], [], [], []
    d_real_a, d_fake_a, d_real_b, d_fake_b = [], [], [], []

    if args.direction in ("AB", "both"):
        for i, f in enumerate(tqdm(files_A, desc="A->B")):
            x = load_image(f, args.size)[None]
            y = G(x)
            rec = F_(y)
            cyc_a.append(np.abs(rec - x).mean())
            id_x = F_(x)
            id_a.append(np.abs(id_x - x).mean())
            if args.with_d:
                d_real_a.append(DA(x).mean())
                d_fake_a.append(DB(y).mean())
            if i < args.n_samples:
                grid = np.concatenate([to_uint8(x[0]), to_uint8(y[0]), to_uint8(rec[0])], axis=1)
                Image.fromarray(grid).save(f"{args.out}/AtoB_{f.stem}.png")

    if args.direction in ("BA", "both"):
        for i, f in enumerate(tqdm(files_B, desc="B->A")):
            x = load_image(f, args.size)[None]
            y = F_(x)
            rec = G(y)
            cyc_b.append(np.abs(rec - x).mean())
            id_y = G(x)
            id_b.append(np.abs(id_y - x).mean())
            if args.with_d:
                d_real_b.append(DB(x).mean())
                d_fake_b.append(DA(y).mean())
            if i < args.n_samples:
                grid = np.concatenate([to_uint8(x[0]), to_uint8(y[0]), to_uint8(rec[0])], axis=1)
                Image.fromarray(grid).save(f"{args.out}/BtoA_{f.stem}.png")

    print(f"\nsaved {args.n_samples} images per direction to {args.out}/")
    print("---- test-set metrics (mean over all test images) ----")
    if cyc_a:
        print(f"A->B->A cycle L1 :  {np.mean(cyc_a):.4f}  (n={len(cyc_a)})")
        print(f"F(x) identity L1 :  {np.mean(id_a):.4f}")
    if cyc_b:
        print(f"B->A->B cycle L1 :  {np.mean(cyc_b):.4f}  (n={len(cyc_b)})")
        print(f"G(y) identity L1 :  {np.mean(id_b):.4f}  (note: y is from B)")
    if args.with_d and d_real_a:
        print(f"D_A(real_a) mean : {np.mean(d_real_a):.3f}  | D_B(fake_b) mean : {np.mean(d_fake_b):.3f}")
        print(f"D_B(real_b) mean : {np.mean(d_real_b):.3f}  | D_A(fake_a) mean : {np.mean(d_fake_a):.3f}")
        print("(LSGAN equilibrium target: real ~1, fake ~0; values close to each other = D fooled)")


if __name__ == "__main__":
    main()
