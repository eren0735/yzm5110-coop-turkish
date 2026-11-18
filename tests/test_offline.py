"""Offline end-to-end test with tiny randomly-initialised CLIP and XLM-R models.

Checks that every script runs, CoOp's loss decreases for both encoders, only the context
vectors receive gradients, and tables/figures are produced. Needs no downloads or GPU.
Run from the repo root:  python -m tests.test_offline
"""
import os
import shutil
import sys

import numpy as np
import torch

RES = "results_test"
FEAT = os.path.join(RES, "fake_features.pt")


class StubTokenizer:
    """Minimal XLM-R-like tokenizer (ids derived from characters)."""
    cls_token_id, pad_token_id, sep_token_id, unk_token_id = 0, 1, 2, 3
    bos_token_id, eos_token_id = 0, 2

    def _ids(self, text):
        return [4 + (ord(ch) % 90) for ch in text]

    def tokenize(self, text):
        return list(text)

    def __call__(self, texts, add_special_tokens=True, padding=False, return_tensors=None):
        single = isinstance(texts, str)
        seqs = [self._ids(t) for t in ([texts] if single else texts)]
        if add_special_tokens:
            seqs = [[0] + s + [2] for s in seqs]
        if return_tensors != "pt":
            return {"input_ids": seqs[0] if single else seqs}
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), 1)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def tiny_clip():
    from clip.model import CLIP
    from coopt.text_encoders import CLIPText
    m = CLIP(embed_dim=32, image_resolution=32, vision_layers=1, vision_width=64, vision_patch_size=16,
             context_length=77, vocab_size=49408, transformer_width=64, transformer_heads=2,
             transformer_layers=2)
    return CLIPText(m)


def tiny_mclip():
    from transformers import XLMRobertaConfig, XLMRobertaModel
    from coopt.text_encoders import MCLIPText
    cfg = XLMRobertaConfig(vocab_size=100, hidden_size=48, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=96, max_position_embeddings=80, pad_token_id=1)
    return MCLIPText(XLMRobertaModel(cfg), torch.nn.Linear(48, 32), StubTokenizer())


def fake_encoder(name, device):
    return (tiny_clip() if name == "clip" else tiny_mclip()).to(device)


def make_features(n_per_class=60, d=32):
    rng = np.random.RandomState(0)
    centers = rng.randn(10, d) * 2
    labels = np.repeat(np.arange(10), n_per_class)
    feats = centers[labels] + rng.randn(len(labels), d)
    os.makedirs(RES, exist_ok=True)
    torch.save({"feats": torch.tensor(feats, dtype=torch.float32), "labels": torch.tensor(labels),
                "rel_paths": [f"C{l}/img_{i}.jpg" for i, l in enumerate(labels)]}, FEAT)


def run(module, argv):
    sys.argv = [module.__name__] + argv
    module.main()


def main():
    shutil.rmtree(RES, ignore_errors=True)
    make_features()
    import make_figures
    import make_tables
    import run_coop
    import run_linear_probe
    import run_tip_adapter
    import run_zeroshot
    for mod in (run_coop, run_zeroshot, run_tip_adapter):
        mod.load_encoder = fake_encoder
    common = ["--features", FEAT, "--results", RES]

    # gradient check: only the context vectors are trainable
    from coopt.data import CLASSNAMES
    for enc in (tiny_clip(), tiny_mclip()):
        lr = enc.prompt_learner(CLASSNAMES["tr"], n_ctx=4)
        assert not any(p.requires_grad for p in enc.parameters())
        out = lr(enc)
        assert out.shape == (10, 32), out.shape
        out.sum().backward()
        assert lr.ctx.grad is not None and lr.ctx.grad.abs().sum() > 0
        print(f"[ok] {enc.name} prompt learner: grads only on ctx {tuple(lr.ctx.shape)}")

    run(run_zeroshot, common + ["--repeats", "2"])
    run(run_linear_probe, common + ["--shots", "1", "4", "--seeds", "1", "2"])
    run(run_tip_adapter, common + ["--shots", "1", "4", "--seeds", "1", "2"])
    for e in ("clip", "mclip"):
        run(run_coop, common + ["--encoder", e, "--shots", "1", "4", "--seeds", "1", "2",
                                "--epochs", "30", "--n-ctx", "4"])

    import json
    import glob
    for p in glob.glob(os.path.join(RES, "coop", "*4shot*.json")):
        h = json.load(open(p))["loss_history"]
        assert h[-1] < h[0], (p, h[0], h[-1])
    print("[ok] CoOp loss decreases for every 4-shot run")

    run(make_tables, ["--results", RES])
    run(make_figures, ["--results", RES])
    for f in ("acc_vs_shots", "confusion_coop", "perclass_delta", "loss_curves"):
        assert os.path.exists(os.path.join(RES, "figures", f + ".png")), f
    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
