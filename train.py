"""Train CycleGAN"""
import argparse
import os
import pickle
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

from data import UnpairedDataset, to_uint8
from models import Discriminator, Generator
from optim import Adam


class ImageBuffer:
    """Stores last `size` generated samples; with prob 0.5 returns one of them, else current"""

    def __init__(self, size: int = 50):
        self.size = size
        self.imgs: list[np.ndarray] = []

    def query(self, img: np.ndarray) -> np.ndarray:
        if self.size == 0:
            return img
        if len(self.imgs) < self.size:
            self.imgs.append(img.copy())
            return img
        if np.random.rand() < 0.5:
            i = np.random.randint(self.size)
            old = self.imgs[i]
            self.imgs[i] = img.copy()
            return old
        return img


def lsgan_g(d_fake: np.ndarray) -> tuple[float, np.ndarray]:
    loss = 0.5 * ((d_fake - 1) ** 2).mean()
    grad = (d_fake - 1) / d_fake.size
    return float(loss), grad


def lsgan_d(d_real: np.ndarray, d_fake: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    loss = 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
    grad_real = (d_real - 1) / d_real.size
    grad_fake = d_fake / d_fake.size
    return float(loss), grad_real, grad_fake


def l1(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean absolute error and its gradient w.r.t. a"""
    diff = a - b
    return float(np.abs(diff).mean()), np.sign(diff) / diff.size


def save_grid(path: str, *rows: tuple[np.ndarray, ...]) -> None:
    grid = np.concatenate([np.concatenate([to_uint8(im[0]) for im in row], axis=1) for row in rows], axis=0)
    Image.fromarray(grid).save(path)


def save_ckpt(path: str, **modules) -> None:
    state = {name: [p.value.copy() for p in m.parameters()] for name, m in modules.items()}
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_ckpt(path: str, **modules) -> None:
    with open(path, "rb") as f:
        state = pickle.load(f)
    for name, m in modules.items():
        for p, v in zip(m.parameters(), state[name]):
            p.value[:] = v


def train_step(a: np.ndarray, b: np.ndarray,
               G, F_, DA, DB,
               opt_G, opt_DA, opt_DB,
               buf_A: ImageBuffer, buf_B: ImageBuffer,
               lambda_cyc: float, lambda_id: float) -> dict[str, float]:
    """One CycleGAN step. Returns scalar losses."""
    opt_G.zero_grad()

    # ----- A -> B -> A -----
    fake_b = G(a)
    rec_a = F_(fake_b)
    d_fake_b = DB(fake_b)
    g_loss_b, dD = lsgan_g(d_fake_b)
    cyc_a_loss, drec_a = l1(rec_a, a)
    # backward through F (cycle path) and DB (gan path), both end at fake_b
    grad_fb_cyc = F_.backward(lambda_cyc * drec_a)
    grad_fb_gan = DB.backward(dD)
    G.backward(grad_fb_cyc + grad_fb_gan)

    # ----- B -> A -> B -----
    fake_a = F_(b)
    rec_b = G(fake_a)
    d_fake_a = DA(fake_a)
    g_loss_a, dD = lsgan_g(d_fake_a)
    cyc_b_loss, drec_b = l1(rec_b, b)
    grad_fa_cyc = G.backward(lambda_cyc * drec_b)
    grad_fa_gan = DA.backward(dD)
    F_.backward(grad_fa_cyc + grad_fa_gan)

    # ----- identity -----
    id_loss_val = 0.0
    if lambda_id > 0:
        id_b = G(b)
        id_b_loss, did_b = l1(id_b, b)
        G.backward(lambda_id * lambda_cyc * did_b)
        id_a = F_(a)
        id_a_loss, did_a = l1(id_a, a)
        F_.backward(lambda_id * lambda_cyc * did_a)
        id_loss_val = id_a_loss + id_b_loss

    opt_G.step()

    # ----- DA: real_a vs buffered fake_a -----
    fake_a_buf = buf_A.query(fake_a)
    opt_DA.zero_grad()
    d_real = DA(a)
    DA.backward((d_real - 1) / d_real.size)
    d_fake = DA(fake_a_buf)
    DA.backward(d_fake / d_fake.size)
    da_loss = 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
    # paper: divide D loss by 2 -> scale grads by 0.5 instead of full step
    for p in DA.parameters():
        p.grad *= 0.5
    opt_DA.step()

    # ----- DB -----
    fake_b_buf = buf_B.query(fake_b)
    opt_DB.zero_grad()
    d_real = DB(b)
    DB.backward((d_real - 1) / d_real.size)
    d_fake = DB(fake_b_buf)
    DB.backward(d_fake / d_fake.size)
    db_loss = 0.5 * (((d_real - 1) ** 2).mean() + (d_fake ** 2).mean())
    for p in DB.parameters():
        p.grad *= 0.5
    opt_DB.step()

    return {
        "g": g_loss_a + g_loss_b,
        "cyc": cyc_a_loss + cyc_b_loss,
        "id": id_loss_val,
        "da": float(da_loss),
        "db": float(db_loss),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/horse2zebra")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_res", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--decay_start", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_cyc", type=float, default=10.0)
    p.add_argument("--lambda_id", type=float, default=0.5)
    p.add_argument("--max_per_side", type=int, default=None)
    p.add_argument("--out", default="runs/horse2zebra")
    p.add_argument("--sample_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(f"{args.out}/samples", exist_ok=True)
    os.makedirs(f"{args.out}/ckpt", exist_ok=True)

    ds = UnpairedDataset(f"{args.data}/trainA", f"{args.data}/trainB", size=args.size, max_per_side=args.max_per_side)
    print(f"loaded {len(ds.A)} A / {len(ds.B)} B images, epoch length = {len(ds)}")

    G = Generator(3, 3, args.ngf, args.n_res)
    F_ = Generator(3, 3, args.ngf, args.n_res)
    DA = Discriminator(3, args.ndf)
    DB = Discriminator(3, args.ndf)

    opt_G = Adam(G.parameters() + F_.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_DA = Adam(DA.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_DB = Adam(DB.parameters(), lr=args.lr, betas=(0.5, 0.999))

    if args.resume:
        load_ckpt(args.resume, G=G, F=F_, DA=DA, DB=DB)
        print(f"resumed from {args.resume}")

    buf_A, buf_B = ImageBuffer(50), ImageBuffer(50)
    global_step = 0

    for epoch in range(args.epochs):
        if epoch >= args.decay_start:
            decay = max(0.0, 1.0 - (epoch - args.decay_start) / max(1, args.epochs - args.decay_start))
            for opt in (opt_G, opt_DA, opt_DB):
                opt.lr = args.lr * decay

        bar = tqdm(ds.epoch(), total=len(ds), desc=f"ep {epoch + 1}/{args.epochs} lr={opt_G.lr:.1e}")
        running = {"g": 0.0, "cyc": 0.0, "id": 0.0, "da": 0.0, "db": 0.0}
        n = 0
        t0 = time.time()

        for a, b in bar:
            losses = train_step(a, b, G, F_, DA, DB, opt_G, opt_DA, opt_DB,
                                buf_A, buf_B, args.lambda_cyc, args.lambda_id)
            for k in running:
                running[k] += losses[k]
            n += 1
            global_step += 1

            if n % 10 == 0:
                bar.set_postfix({k: f"{v / n:.3f}" for k, v in running.items()})

            if global_step % args.sample_every == 0:
                fake_b = G(a); rec_a = F_(fake_b)
                fake_a = F_(b); rec_b = G(fake_a)
                save_grid(f"{args.out}/samples/step{global_step:06d}.png",
                          (a, fake_b, rec_a), (b, fake_a, rec_b))

        dt = time.time() - t0
        avg = {k: v / max(1, n) for k, v in running.items()}
        print(f"epoch {epoch + 1} done in {dt:.0f}s | "
              + " | ".join(f"{k}={v:.3f}" for k, v in avg.items()))
        save_ckpt(f"{args.out}/ckpt/epoch{epoch + 1:03d}.pkl", G=G, F=F_, DA=DA, DB=DB)
        save_ckpt(f"{args.out}/ckpt/last.pkl", G=G, F=F_, DA=DA, DB=DB)


if __name__ == "__main__":
    main()
