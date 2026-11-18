"""Train and evaluate CoOp with a given text encoder and class-name language.

Example:
  python run_coop.py --encoder clip  --langs en tr --shots 1 4 16 --seeds 1 2 3
  python run_coop.py --encoder mclip --langs en tr --shots 1 4 16 --seeds 1 2 3

Writes results/coop/<encoder>_<lang>_<k>shot_seed<s>.json (+ learned context in results/ckpt).
Model selection: none. As in CoOp, the model after the last epoch is evaluated on test.
"""
import argparse
import os

import torch

from coopt.common import (device, few_shot, load_features, metrics, n_params, save_json,
                          set_seed)
from coopt.data import CLASSNAMES, TEMPLATES
from coopt.methods import COOP_EPOCHS, coop_text_features, classify, train_coop
from coopt.text_encoders import load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["clip", "mclip"], required=True)
    ap.add_argument("--langs", nargs="+", default=["en", "tr"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-ctx", type=int, default=16)
    ap.add_argument("--ctx-init", action="store_true",
                    help="initialise context from the zero-shot template of the same language "
                         "(CoOp's optional variant; default = random init as in CoOp's main results)")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _init")
    ap.add_argument("--epochs", type=int, default=None, help="override CoOp's per-shot schedule")
    ap.add_argument("--features", default="features/eurosat_vitb32.pt")
    ap.add_argument("--split-json", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--max-test", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    dev = device()
    feats, labels, splits, _ = load_features(args.features, args.split_json, args.max_test)
    feats, labels = feats.to(dev), labels.to(dev)
    enc = load_encoder(args.encoder, dev)
    te = torch.as_tensor(splits["test"], device=dev)

    for lang in args.langs:
        names = CLASSNAMES[lang]
        for k in args.shots:
            epochs = args.epochs or COOP_EPOCHS[k]
            for seed in args.seeds:
                set_seed(seed)
                tr_idx, _ = few_shot(labels.cpu(), splits, k, seed)
                tr = torch.as_tensor(tr_idx, device=dev)
                init = TEMPLATES[lang].split("{}")[0].strip() if args.ctx_init else None
                learner = enc.prompt_learner(names, args.n_ctx, init)
                print(f"[coop] {args.encoder}/{lang} {k}-shot seed {seed}: {len(tr)} imgs, {epochs} epochs")
                res = train_coop(enc, learner, feats[tr], labels[tr], epochs, seed=seed, log=print)
                W = coop_text_features(enc, learner)
                m = metrics(classify(feats[te], W, enc.logit_scale), labels[te], len(names))
                print(f"  test acc {m['acc']:.2f}  macro-F1 {m['macro_f1']:.2f}")
                tag = f"{args.encoder}_{lang}_{k}shot_seed{seed}{args.tag}"
                ck = os.path.join(args.results, "ckpt", f"coop_{tag}.pt")
                os.makedirs(os.path.dirname(ck), exist_ok=True)
                torch.save(learner.ctx.detach().cpu(), ck)
                save_json({
                    "method": "coop" + args.tag, "encoder": args.encoder, "ctx_init": init, "lang": lang, "shots": k,
                    "seed": seed, "n_ctx": learner.ctx.shape[0], "epochs": epochs,
                    "n_train": len(tr), "train_idx": tr_idx.tolist(),
                    "classnames": names, "trainable_params": n_params(learner),
                    "text_encoder_params": enc.text_params(),
                    **res, "test": m, "ctx_ckpt": ck,
                }, os.path.join(args.results, "coop" + args.tag, f"{tag}.json"))


if __name__ == "__main__":
    main()
