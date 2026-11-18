"""Shared helpers: seeding, feature loading, metrics, timing, JSON output."""
import json
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from .data import make_splits, sample_few_shot


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(path, split_json=None, max_test=None):
    """Load cached image features; return L2-normalised tensors plus split indices."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    feats = F.normalize(d["feats"].float(), dim=-1)
    labels = d["labels"].long()
    splits = make_splits(labels.numpy(), d.get("rel_paths"), split_json=split_json)
    if max_test:  # only for smoke tests: a random subset that keeps all classes
        rng = np.random.RandomState(0)
        splits["test"] = np.sort(rng.permutation(splits["test"])[:max_test])
        splits["val"] = np.sort(rng.permutation(splits["val"])[:max_test])
    return feats, labels, splits, d


def few_shot(labels, splits, shots, seed):
    """Few-shot train set and a few-shot val set of min(shots, 4) per class (as in CoOp)."""
    lab = labels.numpy()
    tr = sample_few_shot(splits["train"], lab, shots, seed)
    va = sample_few_shot(splits["val"], lab, min(shots, 4), seed)
    return tr, va


def metrics(logits_or_preds, y, n_classes):
    t = logits_or_preds
    preds = t.argmax(-1) if t.dim() == 2 else t
    preds = preds.cpu().numpy()
    y = y.cpu().numpy()
    return {
        "acc": float((preds == y).mean() * 100),
        "macro_f1": float(f1_score(y, preds, average="macro") * 100),
        "confusion": confusion_matrix(y, preds, labels=list(range(n_classes))).tolist(),
        "preds": preds.tolist(),
    }


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timer(store, key):
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    store[key] = time.perf_counter() - t0


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**20
    return None


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
