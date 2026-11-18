"""Frozen text encoders (OpenAI CLIP, M-CLIP) and CoOp prompt learners for each.

The CLIP prompt learner and text-encoder forward pass are adapted from the official
CoOp implementation (KaiyangZhou/CoOp, trainers/coop.py, MIT licence):
  prompt = [SOS] [V]_1 ... [V]_M [CLASS] . [EOS]
with M learnable context vectors shared across classes ("unified context"),
class token at the end, random init N(0, 0.02).

The M-CLIP variant is our extension: the same M learnable vectors are inserted
after XLM-R's <s> token, and the output is mean-pooled and linearly projected
exactly as in multilingual_clip.pt_multilingual_clip.MultilingualCLIP.forward.
Both encoders map into the same 512-d space as OpenAI CLIP ViT-B/32 image features,
so image features are shared between conditions.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------- OpenAI CLIP
class CLIPText(nn.Module):
    name = "clip"

    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model.float().eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        self.width = self.clip.ln_final.weight.shape[0]
        self.logit_scale = float(self.clip.logit_scale.exp().item())

    @classmethod
    def load(cls, backbone="ViT-B/32", device="cuda"):
        import clip
        model, _ = clip.load(backbone, device=device, jit=False)
        return cls(model)

    @property
    def device(self):
        return self.clip.token_embedding.weight.device

    def text_params(self):
        m = self.clip
        mods = [m.token_embedding, m.transformer, m.ln_final]
        return sum(p.numel() for mod in mods for p in mod.parameters()) + \
            m.positional_embedding.numel() + m.text_projection.numel()

    @torch.no_grad()
    def encode_texts(self, texts):
        import clip
        tok = clip.tokenize(texts, truncate=True).to(self.device)
        return self.clip.encode_text(tok).float()

    def encode_prompts(self, prompts, tokenized):
        m = self.clip
        x = prompts + m.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = m.transformer(x)
        x = x.permute(1, 0, 2)
        x = m.ln_final(x)
        # features at the EOT token (highest token id in each sequence)
        return x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ m.text_projection

    def prompt_learner(self, classnames, n_ctx=16, ctx_init=None):
        return CLIPPromptLearner(self, classnames, n_ctx, ctx_init)


class CLIPPromptLearner(nn.Module):
    def __init__(self, enc, classnames, n_ctx, ctx_init=None):
        import clip
        super().__init__()
        if ctx_init:  # CoOp's optional initialisation from a hand-written phrase
            from clip.simple_tokenizer import SimpleTokenizer
            n_ctx = len(SimpleTokenizer().encode(ctx_init))
            placeholder = ctx_init
        else:
            placeholder = " ".join(["X"] * n_ctx)
        prompts = [f"{placeholder} {name}." for name in classnames]
        tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(enc.device)
        with torch.no_grad():
            emb = enc.clip.token_embedding(tokenized)
        self.register_buffer("tokenized", tokenized)
        self.register_buffer("prefix", emb[:, :1, :].clone())            # SOS
        self.register_buffer("suffix", emb[:, 1 + n_ctx:, :].clone())    # CLS . EOS pad
        init = emb[0, 1:1 + n_ctx, :].clone() if ctx_init else \
            torch.empty(n_ctx, emb.shape[-1], device=enc.device).normal_(std=0.02)
        self.ctx = nn.Parameter(init)
        self.n_cls = len(classnames)

    def forward(self, enc):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.prefix, ctx, self.suffix], dim=1)
        return enc.encode_prompts(prompts, self.tokenized)


# ---------------------------------------------------------------- M-CLIP
class MCLIPText(nn.Module):
    """M-CLIP text tower = XLM-R transformer -> masked mean pooling -> Linear(1024 -> 512)."""
    name = "mclip"

    def __init__(self, transformer, proj, tokenizer, logit_scale=100.0):
        super().__init__()
        self.transformer = transformer.eval()
        self.proj = proj.eval()
        self.tok = tokenizer
        for p in self.parameters():
            p.requires_grad_(False)
        self.width = self.transformer.get_input_embeddings().weight.shape[1]
        # M-CLIP is distilled into CLIP's embedding space, so CLIP's logit scale (100) is reused.
        self.logit_scale = logit_scale

    @classmethod
    def load(cls, model_name="M-CLIP/XLM-Roberta-Large-Vit-B-32", device="cuda"):
        import transformers
        from multilingual_clip import pt_multilingual_clip
        m = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
        tok = transformers.AutoTokenizer.from_pretrained(model_name)
        if not (hasattr(m, "transformer") and hasattr(m, "LinearTransformation")):
            raise RuntimeError("Unexpected multilingual_clip model layout; check package version.")
        return cls(m.transformer, m.LinearTransformation, tok).to(device)

    @property
    def device(self):
        return self.proj.weight.device

    def text_params(self):
        return sum(p.numel() for p in self.parameters())

    def _pool_project(self, hidden, mask):
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1)
        return self.proj(pooled)

    @torch.no_grad()
    def encode_texts(self, texts):
        t = self.tok(texts, padding=True, return_tensors="pt")
        t = {k: v.to(self.device) for k, v in t.items() if k in ("input_ids", "attention_mask")}
        h = self.transformer(**t)[0]
        return self._pool_project(h, t["attention_mask"]).float()

    def encode_prompts(self, prompts, mask):
        h = self.transformer(inputs_embeds=prompts, attention_mask=mask)[0]
        return self._pool_project(h, mask)

    def prompt_learner(self, classnames, n_ctx=16, ctx_init=None):
        return MCLIPPromptLearner(self, classnames, n_ctx, ctx_init)


class MCLIPPromptLearner(nn.Module):
    """<s> [V]_1..[V]_M  class name .  </s>  (pad)"""

    def __init__(self, enc, classnames, n_ctx, ctx_init=None):
        super().__init__()
        tok = enc.tok
        bos = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        eos = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
        if ctx_init:
            ctx_ids = list(tok(ctx_init, add_special_tokens=False)["input_ids"])
            n_ctx = len(ctx_ids)
        else:
            ctx_ids = [tok.unk_token_id] * n_ctx  # placeholders; embeddings replaced by ctx
        seqs = []
        for name in classnames:
            ids = tok(f"{name}.", add_special_tokens=False)["input_ids"]
            seqs.append([bos] + ctx_ids + list(ids) + [eos])
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        ids, mask = ids.to(enc.device), mask.to(enc.device)
        with torch.no_grad():
            emb = enc.transformer.get_input_embeddings()(ids)
        self.register_buffer("mask", mask)
        self.register_buffer("prefix", emb[:, :1, :].clone())
        self.register_buffer("suffix", emb[:, 1 + n_ctx:, :].clone())
        init = emb[0, 1:1 + n_ctx, :].clone() if ctx_init else \
            torch.empty(n_ctx, emb.shape[-1], device=enc.device).normal_(std=0.02)
        self.ctx = nn.Parameter(init)
        self.n_cls = len(classnames)

    def forward(self, enc):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.prefix, ctx, self.suffix], dim=1)
        return enc.encode_prompts(prompts, self.mask)


def load_encoder(name, device):
    if name == "clip":
        return CLIPText.load("ViT-B/32", device)
    if name == "mclip":
        return MCLIPText.load(device=device)
    raise ValueError(name)
