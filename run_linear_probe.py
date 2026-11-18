"""Linear probe on frozen image features. Uses no text, so it is language-independent:
it shows what the few-shot images alone can reach."""
import argparse
import os

import torch

from coopt.common import few_shot, load_features, metrics, save_json, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--features", default="features/eurosat_vitb32.pt")
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-test", type=int, default=None)
    args = ap.parse_args()

    feats, labels, splits, _ = load_features(args.features, args.split_json, args.max_test)
    te = torch.as_tensor(splits["test"])
    from coopt.methods import linear_probe
    for k in args.shots:
        for seed in args.seeds:
            tr, va = few_shot(labels, splits, k, seed)
            t = {}
            with timer(t, "fit"):
                preds, hp = linear_probe(feats[tr], labels[tr], feats[va], labels[va], feats[te])
            m = metrics(preds, labels[te], 10)
            print(f"[linear probe] {k}-shot seed {seed}: acc {m['acc']:.2f} (C={hp['C']:.3g})")
            save_json({"method": "linear_probe", "encoder": "none", "lang": "none", "shots": k,
                       "seed": seed, **hp, "train_time_s": t["fit"], "test": m},
                      os.path.join(args.results, "linear_probe", f"{k}shot_seed{seed}.json"))


if __name__ == "__main__":
    main()
