"""Training/evaluation for CoOp, zero-shot, Tip-Adapter (training-free) and linear probe.

All methods work on cached, L2-normalised CLIP ViT-B/32 image features.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from .common import peak_mb, reset_peak, sync

# CoOp optimisation defaults (CoOp configs/trainers/CoOp/*.yaml + Dassl):
# SGD lr 0.002, momentum 0.9, wd 5e-4, batch 32, cosine schedule,
# 1 warm-up epoch at constant lr 1e-5; 50 epochs for 1-2 shots, 100 for 4-8, 200 for 16.
COOP_EPOCHS = {1: 50, 2: 50, 4: 100, 8: 100, 16: 200}


def coop_lr(epoch, max_epoch, base_lr=0.002, warmup_lr=1e-5):
    if epoch < 1:
        return warmup_lr
    return 0.5 * base_lr * (1 + math.cos(math.pi * (epoch - 1) / max_epoch))


def train_coop(enc, learner, X, y, epochs, bs=32, lr=0.002, seed=0, log=None):
    """X: (N, D) normalised image features on enc.device, y: (N,) labels."""
    opt = torch.optim.SGD(learner.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    drop_last = n >= bs  # Dassl only drops the last batch if the set is larger than a batch
    history, step_times = [], []
    reset_peak()
    sync()
    import time
    t0 = time.perf_counter()
    learner.train()
    for ep in range(epochs):
        for pg in opt.param_groups:
            pg["lr"] = coop_lr(ep, epochs, lr)
        perm = torch.randperm(n, generator=g)
        n_batches = n // bs if drop_last else math.ceil(n / bs)
        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * bs:(b + 1) * bs].to(X.device)
            ts = time.perf_counter()
            text = F.normalize(learner(enc), dim=-1)
            logits = enc.logit_scale * X[idx] @ text.t()
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sync()
            step_times.append(time.perf_counter() - ts)
            ep_loss += loss.item()
        history.append(ep_loss / max(n_batches, 1))
        if log and (ep == 0 or (ep + 1) % max(1, epochs // 5) == 0):
            log(f"  epoch {ep + 1}/{epochs} loss {history[-1]:.4f}")
    sync()
    return {
        "train_time_s": time.perf_counter() - t0,
        "mean_step_ms": 1000 * float(np.mean(step_times)) if step_times else None,
        "peak_mem_mb": peak_mb(),
        "loss_history": history,
    }


@torch.no_grad()
def coop_text_features(enc, learner):
    learner.eval()
    return F.normalize(learner(enc), dim=-1)


@torch.no_grad()
def zeroshot_weights(enc, classnames, template):
    texts = [template.format(c) for c in classnames]
    return F.normalize(enc.encode_texts(texts), dim=-1)


@torch.no_grad()
def classify(X, W, scale):
    return scale * X @ W.t()


# ---------------------------------------------------------------- Tip-Adapter
@torch.no_grad()
def tip_adapter_logits(X, W, keys, values, alpha, beta, scale=100.0):
    """Training-free Tip-Adapter (Zhang et al., ECCV 2022):
    logits = scale * X W^T + alpha * exp(-beta * (1 - X K^T)) V
    """
    affinity = X @ keys.t()
    cache = torch.exp(-beta * (1 - affinity)) @ values
    return scale * X @ W.t() + alpha * cache


@torch.no_grad()
def tip_adapter_search(Xval, yval, W, keys, values, scale=100.0,
                       alphas=None, betas=None):
    """Grid-search alpha, beta on the few-shot validation set (never on test)."""
    alphas = np.linspace(0.1, 3.0, 15) if alphas is None else alphas
    betas = np.linspace(0.1, 7.0, 24) if betas is None else betas
    best = (-1.0, 1.0, 1.0)
    for a in alphas:
        for b in betas:
            acc = (tip_adapter_logits(Xval, W, keys, values, a, b, scale).argmax(-1) == yval).float().mean().item()
            if acc > best[0]:
                best = (acc, float(a), float(b))
    return {"val_acc": best[0] * 100, "alpha": best[1], "beta": best[2]}


# ---------------------------------------------------------------- linear probe
def linear_probe(Xtr, ytr, Xval, yval, Xte, Cs=None):
    """Logistic regression on image features; C chosen on few-shot val (CLIP/CoOp protocol)."""
    Cs = np.logspace(-3, 3, 13) if Cs is None else Cs
    Xtr, ytr, Xval, yval, Xte = (t.cpu().numpy() for t in (Xtr, ytr, Xval, yval, Xte))
    best_c, best_acc = Cs[0], -1
    for c in Cs:
        clf = LogisticRegression(C=c, max_iter=2000)
        clf.fit(Xtr, ytr)
        acc = (clf.predict(Xval) == yval).mean()
        if acc > best_acc:
            best_c, best_acc = c, acc
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(Xtr, ytr)
    return torch.from_numpy(clf.predict(Xte)), {"C": float(best_c), "val_acc": float(best_acc * 100)}
