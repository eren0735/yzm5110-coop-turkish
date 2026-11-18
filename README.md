# Language mismatch in CLIP few-shot adaptation (YZM5110 Research Project)

**Question.** CoOp learns the context words of CLIP's prompt from a few labelled images, but it
was only evaluated with English class names. How much accuracy is lost when the class names
are Turkish and the text encoder is CLIP's English-centric one, and does swapping in a
multilingual text encoder (M-CLIP, XLM-R-Large distilled into CLIP ViT-B/32's space) recover it?

**Reproduction target.** Zhou et al., *Learning to Prompt for Vision-Language Models* (CoOp),
IJCV 2022, arXiv:2109.01134. Claim tested: with a few labelled images per class, learned
context vectors outperform the hand-crafted zero-shot prompt.

**Intervention.** Replace CLIP's text encoder with M-CLIP's (image encoder unchanged) and run
the same CoOp procedure with Turkish class names.

## Design

All conditions share the same frozen OpenAI CLIP ViT-B/32 image features, the same few-shot
images per seed, and the same test split, so only the text side changes.

|                       | English names | Turkish names |
|-----------------------|---------------|---------------|
| CLIP text encoder     | (a) reference | (b) mismatch  |
| M-CLIP text encoder   | (d) ablation  | (c) intervention |

* `b − a`: cost of Turkish names. `c − b`: gain from the intervention.
* `d − a` (ablation): does M-CLIP help even in English? If so, gains in (c) may come from a
  different/larger encoder rather than from multilinguality.
* Controls: zero-shot (hand-written prompt), training-free Tip-Adapter, and a linear probe on
  image features (uses no text, so it is the language-free reference).
* Resource measures: text-encoder parameters, latency of encoding the class prompts,
  CoOp training time, time per step and peak GPU memory.

Dataset: EuroSAT RGB (10 classes, 27,000 images), as used in CoOp. Shots 1/4/16, seeds 1/2/3.
Test accuracy is reported for the model after the last epoch (CoOp's protocol); hyper-parameters
of Tip-Adapter and the linear probe are chosen on a few-shot validation set, never on test.

## Setup (Google Colab, GPU runtime)

```bash
git clone https://github.com/<user>/<repo>.git && cd <repo>
pip install -r requirements.txt
python -m tests.test_offline      # 1 min, CPU, tiny random models: checks the code paths
bash smoke_test.sh                 # ~5-10 min, real models, small test subset
bash run_all.sh                    # full experiment
```

Or open `colab_run.ipynb`.

**Data.** `extract_features.py` downloads EuroSAT through `torchvision.datasets.EuroSAT`
(dataset licence: MIT, Helber et al. 2019). Images are not stored in this repo. If the
download fails, place the extracted images at `data/eurosat/2750/<ClassName>/*.jpg`.

**Official split (optional).** To use CoOp's split, download `split_zhou_EuroSAT.json`
(link in CoOp's `DATASETS.md`) to `splits/`. Without it, a stratified split with the same sizes
(13,500 / 5,400 / 8,100) and a fixed seed is used. The report states which one was used.

## Outputs

| Path | Content |
|---|---|
| `results/env.json` | Python/torch/CUDA/GPU, git commit, full `pip freeze` |
| `results/zeroshot/`, `coop/`, `tip_adapter/`, `linear_probe/` | one JSON per run: config, few-shot indices, accuracy, macro-F1, confusion matrix, predictions, timing, memory |
| `results/ckpt/` | learned CoOp context vectors (tiny) |
| `results/tables/` | `reproduction`, `two_by_two`, `resources`, `tokens` (Markdown + CSV), `all_runs.csv` |
| `results/figures/` | accuracy vs. shots, confusion matrices, per-class TR−EN deltas, loss curves |

`paper_reported.csv` holds numbers copied from the CoOp paper; they are shown in their own
column and never mixed with our measurements.

Optional variant: `python run_coop.py --encoder mclip --ctx-init --tag _init` initialises the
context from the zero-shot template (CoOp's own optional variant). Results go to `results/coop_init/`.

## Deviations from CoOp (also listed in the report)

1. Backbone: ViT-B/32 (needed to pair with M-CLIP's XLM-R-Large-ViT-B/32). Compare only with
   the paper's numbers for the same backbone, or compare trends only.
2. No training augmentation: image features are cached once (the image encoder is frozen in
   CoOp too), so CoOp's random-resized-crop/flip augmentation is not applied.
3. Shots 1/4/16 (paper: 1/2/4/8/16), 3 seeds each.
4. Re-implemented without Dassl; the prompt learner and text-encoder forward pass are adapted
   from CoOp's `trainers/coop.py` (MIT). Optimiser settings follow CoOp's configs
   (SGD 0.002, momentum 0.9, wd 5e-4, batch 32, cosine, 1 constant warm-up epoch at 1e-5,
   50/100/200 epochs for 1-2/4-8/16 shots); fp32 instead of fp16.
5. CoOp for M-CLIP is our extension: learnable vectors after `<s>`, then XLM-R, masked mean
   pooling and M-CLIP's linear projection. Mean pooling averages the 16 shared context tokens
   with the few class-name tokens, which may dilute class information (check in the results).
6. Turkish class names and template are our translations (`coopt/data.py`); the Turkish
   template uses a colon so names need no case suffix.

## Credits and AI use

CoOp (Zhou et al.), CLIP (Radford et al.), Multilingual-CLIP (Carlsson et al.),
Tip-Adapter (Zhang et al.). AI assistance (Claude, Anthropic) was used for code drafting,
experiment-design discussion and editing; all code was reviewed, run and verified by the author.
See the report for details.
