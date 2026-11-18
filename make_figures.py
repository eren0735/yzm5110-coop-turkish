"""Report figures in results/figures/.

  acc_vs_shots.png     accuracy vs. shots: CoOp and Tip-Adapter for the 4 encoder/language cells,
                       zero-shot at 0 shots, linear probe as a text-free reference
  confusion_coop.png   CoOp confusion matrices (largest shot count, first seed), 2x2 cells
  perclass_delta.png   per-class accuracy change TR − EN, for each encoder (zero-shot and CoOp)
  loss_curves.png      CoOp training loss (largest shot count, first seed)
Colour = encoder (CLIP blue, M-CLIP orange); line style = language (EN solid, TR dashed),
so identity never relies on colour alone.
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from coopt.data import CLASSNAMES

COLOR = {"clip": "#2a78d6", "mclip": "#eb6834", "none": "#6f6e69"}
STYLE = {"en": "-", "tr": "--"}
LABEL = {"clip": "CLIP", "mclip": "M-CLIP", "en": "EN", "tr": "TR"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "savefig.dpi": 200, "savefig.bbox": "tight"})


def load(results, method):
    out = []
    for p in sorted(glob.glob(os.path.join(results, method, "*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def by(runs, **kw):
    return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]


def mean_std(runs):
    a = np.array([r["test"]["acc"] for r in runs])
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def acc_vs_shots(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, method, title in zip(axes, ("coop", "tip_adapter"), ("CoOp", "Tip-Adapter (training-free)")):
        shots = sorted({r["shots"] for r in R[method]})
        if not shots:
            ax.set_visible(False)
            continue
        for enc in ("clip", "mclip"):
            for lang in ("en", "tr"):
                zs = by(R["zeroshot"], encoder=enc, lang=lang)
                xs, ys, es = [], [], []
                if zs:
                    xs.append(0); ys.append(zs[0]["test"]["acc"]); es.append(0)
                for k in shots:
                    rs = by(R[method], encoder=enc, lang=lang, shots=k)
                    if rs:
                        m, s = mean_std(rs)
                        xs.append(k); ys.append(m); es.append(s)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=COLOR[enc], ls=STYLE[lang], lw=2,
                                marker="o", ms=5, capsize=3,
                                label=f"{LABEL[enc]} {LABEL[lang]}")
        lp = [(k, *mean_std(by(R["linear_probe"], shots=k))) for k in shots if by(R["linear_probe"], shots=k)]
        if lp:
            k, m, s = zip(*lp)
            ax.errorbar(k, m, yerr=s, color=COLOR["none"], ls=":", lw=1.5, marker="s", ms=4,
                        capsize=3, label="linear probe (no text)")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xticks([0] + shots)
        ax.set_xticklabels(["0"] + [str(s) for s in shots])
        ax.set_xlabel("shots per class (0 = zero-shot)")
        ax.set_title(title)
    axes[0].set_ylabel("test accuracy (%)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[e], ls=STYLE[l], lw=2, marker="o", ms=5,
                      label=f"{LABEL[e]} {LABEL[l]}") for e in ("clip", "mclip") for l in ("en", "tr")]
    handles.append(Line2D([], [], color=COLOR["none"], ls=":", lw=1.5, marker="s", ms=4,
                          label="linear probe (no text)"))
    axes[0].legend(handles=handles, frameon=False, fontsize=8, handlelength=3)
    fig.savefig(os.path.join(out, "acc_vs_shots.png"))
    plt.close(fig)


def confusion(R, out, names):
    if not R["coop"]:
        return
    k = max(r["shots"] for r in R["coop"])
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for ax, (enc, lang) in zip(axes.flat, [("clip", "en"), ("clip", "tr"), ("mclip", "en"), ("mclip", "tr")]):
        rs = sorted(by(R["coop"], encoder=enc, lang=lang, shots=k), key=lambda r: r["seed"])
        if not rs:
            ax.set_visible(False)
            continue
        cm = np.array(rs[0]["test"]["confusion"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                if cmn[i, j] >= 0.05:
                    ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if cmn[i, j] > 0.5 else "#2b2b29")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        ax.grid(False)
        ax.set_title(f"CoOp {k}-shot, {LABEL[enc]} {LABEL[lang]}, seed {rs[0]['seed']}"
                     f"  (acc {rs[0]['test']['acc']:.1f}%)", fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_coop.png"))
    plt.close(fig)


def per_class_acc(run):
    cm = np.array(run["test"]["confusion"], dtype=float)
    return 100 * np.diag(cm) / cm.sum(1).clip(min=1)


def perclass_delta(R, out, names):
    panels = [("zeroshot", 0)]
    if R["coop"]:
        panels.append(("coop", max(r["shots"] for r in R["coop"])))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(names))
    for ax, (method, k) in zip(axes, panels):
        for i, enc in enumerate(("clip", "mclip")):
            en = by(R[method], encoder=enc, lang="en", shots=k)
            tr = by(R[method], encoder=enc, lang="tr", shots=k)
            if not (en and tr):
                continue
            d = np.mean([per_class_acc(r) for r in tr], 0) - np.mean([per_class_acc(r) for r in en], 0)
            ax.barh(y + (i - 0.5) * 0.38, d, height=0.36, color=COLOR[enc], label=LABEL[enc],
                    edgecolor="white", linewidth=1)
        ax.axvline(0, color="#8a8984", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("per-class accuracy, TR − EN (points)")
        ax.set_title("zero-shot" if method == "zeroshot" else f"CoOp {k}-shot (mean over seeds)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "perclass_delta.png"))
    plt.close(fig)


def loss_curves(R, out):
    if not R["coop"]:
        return
    k = max(r["shots"] for r in R["coop"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for enc in ("clip", "mclip"):
        for lang in ("en", "tr"):
            rs = sorted(by(R["coop"], encoder=enc, lang=lang, shots=k), key=lambda r: r["seed"])
            if rs:
                ax.plot(rs[0]["loss_history"], color=COLOR[enc], ls=STYLE[lang], lw=2,
                        label=f"{LABEL[enc]} {LABEL[lang]}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title(f"CoOp {k}-shot training loss (first seed)")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out, "loss_curves.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--names", choices=["en", "tr"], default="en")
    args = ap.parse_args()
    R = {m: load(args.results, m) for m in ("coop", "zeroshot", "tip_adapter", "linear_probe")}
    out = os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    names = CLASSNAMES[args.names]
    acc_vs_shots(R, out)
    confusion(R, out, names)
    perclass_delta(R, out, names)
    loss_curves(R, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
