"""Zero-shot classification with hand-crafted prompts (EN / TR) and CLIP / M-CLIP text encoders.

Also measures the one-off cost of encoding the class prompts (latency, peak memory).
"""
import argparse
import os

import torch

from coopt.common import (device, load_features, metrics, peak_mb, reset_peak, save_json,
                          timer)
from coopt.data import CLASSNAMES, TEMPLATES
from coopt.methods import classify, zeroshot_weights
from coopt.text_encoders import load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["clip", "mclip"])
    ap.add_argument("--langs", nargs="+", default=["en", "tr"])
    ap.add_argument("--features", default="features/eurosat_vitb32.pt")
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    dev = device()
    feats, labels, splits, _ = load_features(args.features, args.split_json, args.max_test)
    te = torch.as_tensor(splits["test"])
    Xte, yte = feats[te].to(dev), labels[te].to(dev)

    for e in args.encoders:
        enc = load_encoder(e, dev)
        for lang in args.langs:
            names, tpl = CLASSNAMES[lang], TEMPLATES[lang]
            zeroshot_weights(enc, names, tpl)  # warm-up
            t = {}
            reset_peak()
            with timer(t, "total"):
                for _ in range(args.repeats):
                    W = zeroshot_weights(enc, names, tpl)
            m = metrics(classify(Xte, W, enc.logit_scale), yte, len(names))
            print(f"[zero-shot] {e}/{lang}: acc {m['acc']:.2f}  macro-F1 {m['macro_f1']:.2f}")
            save_json({
                "method": "zeroshot", "encoder": e, "lang": lang, "shots": 0, "seed": 0,
                "template": tpl, "classnames": names,
                "text_encode_ms_all_classes": 1000 * t["total"] / args.repeats,
                "text_encode_peak_mem_mb": peak_mb(),
                "text_encoder_params": enc.text_params(), "test": m,
            }, os.path.join(args.results, "zeroshot", f"{e}_{lang}.json"))
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
