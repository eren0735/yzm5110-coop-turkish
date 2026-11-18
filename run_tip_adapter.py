"""Training-free Tip-Adapter on top of the zero-shot classifier of each encoder/language.

alpha and beta are searched on the few-shot validation set (min(k,4) per class), never on test.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from coopt.common import device, few_shot, load_features, metrics, save_json
from coopt.data import CLASSNAMES, TEMPLATES
from coopt.methods import tip_adapter_logits, tip_adapter_search, zeroshot_weights
from coopt.text_encoders import load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["clip", "mclip"])
    ap.add_argument("--langs", nargs="+", default=["en", "tr"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--features", default="features/eurosat_vitb32.pt")
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-test", type=int, default=None)
    args = ap.parse_args()

    dev = device()
    feats, labels, splits, _ = load_features(args.features, args.split_json, args.max_test)
    feats, labels = feats.to(dev), labels.to(dev)
    te = torch.as_tensor(splits["test"], device=dev)
    n_cls = len(CLASSNAMES["en"])

    for e in args.encoders:
        enc = load_encoder(e, dev)
        for lang in args.langs:
            W = zeroshot_weights(enc, CLASSNAMES[lang], TEMPLATES[lang])
            for k in args.shots:
                for seed in args.seeds:
                    tr_idx, va_idx = few_shot(labels.cpu(), splits, k, seed)
                    tr = torch.as_tensor(tr_idx, device=dev)
                    va = torch.as_tensor(va_idx, device=dev)
                    keys = feats[tr]
                    values = F.one_hot(labels[tr], n_cls).float()
                    hp = tip_adapter_search(feats[va], labels[va], W, keys, values, enc.logit_scale)
                    logits = tip_adapter_logits(feats[te], W, keys, values, hp["alpha"], hp["beta"],
                                                enc.logit_scale)
                    m = metrics(logits, labels[te], n_cls)
                    print(f"[tip] {e}/{lang} {k}-shot seed {seed}: acc {m['acc']:.2f}")
                    save_json({"method": "tip_adapter", "encoder": e, "lang": lang, "shots": k,
                               "seed": seed, **hp, "test": m},
                              os.path.join(args.results, "tip_adapter", f"{e}_{lang}_{k}shot_seed{seed}.json"))
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
