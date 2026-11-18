"""Download EuroSAT (RGB, 27k images) and cache CLIP ViT-B/32 image features.

The image encoder is frozen in every method we study (CoOp, zero-shot, Tip-Adapter,
linear probe), and M-CLIP's XLM-R-Large-ViT-B/32 text encoder is aligned to the same
OpenAI ViT-B/32 image encoder, so one feature file serves all conditions.

Usage: python extract_features.py --root data --out features/eurosat_vitb32.pt
"""
import argparse
import os
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from coopt.common import device, peak_mb, reset_peak, sync
from coopt.data import FOLDERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="features/eurosat_vitb32.pt")
    ap.add_argument("--backbone", default="ViT-B/32")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    import clip
    dev = device()
    model, preprocess = clip.load(args.backbone, device=dev, jit=False)
    model = model.float().eval()

    ds = torchvision.datasets.EuroSAT(args.root, download=True, transform=preprocess)
    assert list(ds.classes) == FOLDERS, ds.classes
    rel_paths = ["/".join(p.replace("\\", "/").split("/")[-2:]) for p, _ in ds.samples]
    loader = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, shuffle=False)

    feats, labels = [], []
    reset_peak()
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="encode"):
            feats.append(model.encode_image(x.to(dev)).float().cpu())
            labels.append(y)
    sync()
    elapsed = time.perf_counter() - t0
    feats = torch.cat(feats)
    labels = torch.cat(labels)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "feats": feats, "labels": labels, "rel_paths": rel_paths, "classes": list(ds.classes),
        "backbone": args.backbone, "torchvision_version": torchvision.__version__,
        "encode_time_s": elapsed, "ms_per_image": 1000 * elapsed / len(ds),
        "peak_mem_mb": peak_mb(),
    }, args.out)
    print(f"saved {tuple(feats.shape)} -> {args.out}  ({1000 * elapsed / len(ds):.2f} ms/img)")


if __name__ == "__main__":
    main()
