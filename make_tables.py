"""Aggregate results/*/*.json into the report tables (Markdown + CSV) in results/tables/.

Tables
  all_runs.csv        one row per run
  reproduction.md     CoOp reproduction (CLIP, English) vs. zero-shot / linear probe / Tip-Adapter,
                      with the paper's reported numbers kept in a separate column
                      (fill paper_reported.csv yourself from the CoOp paper)
  two_by_two.md       {CLIP, M-CLIP} x {EN, TR} per method and shot count, plus paired deltas:
                        TR cost (b-a), intervention gain (c-b), encoder-only ablation (d-a)
  resources.md        parameters, text-encoding latency, training time, step time, peak memory
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

CELLS = {"a": ("clip", "en"), "b": ("clip", "tr"), "c": ("mclip", "tr"), "d": ("mclip", "en")}


def load(results):
    rows = []
    for p in glob.glob(os.path.join(results, "*", "*.json")):
        if os.sep + "tables" + os.sep in p:
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if "method" not in r:
            continue
        rows.append({
            "method": r["method"], "encoder": r["encoder"], "lang": r["lang"],
            "shots": r["shots"], "seed": r["seed"],
            "acc": r["test"]["acc"], "macro_f1": r["test"]["macro_f1"],
            "train_time_s": r.get("train_time_s"), "mean_step_ms": r.get("mean_step_ms"),
            "peak_mem_mb": r.get("peak_mem_mb"), "trainable_params": r.get("trainable_params"),
            "text_encoder_params": r.get("text_encoder_params"),
            "text_encode_ms": r.get("text_encode_ms_all_classes"),
        })
    return pd.DataFrame(rows)


def ms(x, digits=2):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return "–"
    if len(x) == 1:
        return f"{x.iloc[0]:.{digits}f}"
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def cell(df, method, enc, lang, shots):
    q = df[(df.method == method) & (df.encoder == enc) & (df.lang == lang) & (df.shots == shots)]
    return q.set_index("seed")["acc"]


def reproduction(df, paper_csv):
    shots = sorted(s for s in df[df.method == "coop"].shots.unique())
    zs = cell(df, "zeroshot", "clip", "en", 0)
    paper = {}
    if os.path.exists(paper_csv):
        p = pd.read_csv(paper_csv, comment="#").dropna()
        paper = {(r.method, int(r.shots)): r.acc for r in p.itertuples()}
    rows = []
    for k in shots:
        rows.append({
            "shots": k,
            "zero-shot (ours)": ms(zs),
            "CoOp (ours)": ms(cell(df, "coop", "clip", "en", k)),
            "linear probe (ours)": ms(cell(df, "linear_probe", "none", "none", k)),
            "Tip-Adapter (ours)": ms(cell(df, "tip_adapter", "clip", "en", k)),
            "CoOp (paper, reported)": paper.get(("coop", k), "fill in"),
            "zero-shot (paper, reported)": paper.get(("zeroshot", 0), "fill in"),
        })
    return pd.DataFrame(rows)


def paired_delta(x, y):
    common = x.index.intersection(y.index)
    return (y[common] - x[common]) if len(common) else pd.Series(dtype=float)


def two_by_two(df):
    rows = []
    methods = sorted(m for m in df.method.unique() if m.startswith("coop")) + ["tip_adapter"]
    specs = [("zeroshot", 0)] + [(m, k) for m in methods
                                 for k in sorted(df[df.method == m].shots.unique())]
    for m, k in specs:
        c = {key: cell(df, m, *ek, k) for key, ek in CELLS.items()}
        if all(len(v) == 0 for v in c.values()):
            continue
        rows.append({
            "method": m, "shots": k,
            "(a) CLIP EN": ms(c["a"]), "(b) CLIP TR": ms(c["b"]),
            "(c) M-CLIP TR": ms(c["c"]), "(d) M-CLIP EN": ms(c["d"]),
            "TR cost b−a": ms(paired_delta(c["a"], c["b"])),
            "gain c−b": ms(paired_delta(c["b"], c["c"])),
            "gap left c−a": ms(paired_delta(c["a"], c["c"])),
            "ablation d−a": ms(paired_delta(c["a"], c["d"])),
        })
    return pd.DataFrame(rows)


def resources(df):
    rows = []
    for enc in ("clip", "mclip"):
        z = df[(df.method == "zeroshot") & (df.encoder == enc)]
        r = {"encoder": enc,
             "text encoder params (M)": f"{z.text_encoder_params.iloc[0] / 1e6:.1f}" if len(z) else "–",
             "encode 10 prompts (ms)": ms(z.text_encode_ms) if len(z) else "–"}
        for k in sorted(df[df.method == "coop"].shots.unique()):
            q = df[(df.method == "coop") & (df.encoder == enc) & (df.shots == k)]
            r[f"CoOp {k}-shot train time (s)"] = ms(q.train_time_s, 1)
            r[f"CoOp {k}-shot step (ms)"] = ms(q.mean_step_ms, 1)
            r[f"CoOp {k}-shot peak mem (MB)"] = ms(q.peak_mem_mb, 0)
        q = df[(df.method == "coop") & (df.encoder == enc)]
        r["trainable params"] = int(q.trainable_params.iloc[0]) if len(q) else "–"
        rows.append(r)
    return pd.DataFrame(rows).set_index("encoder").T.reset_index().rename(columns={"index": "measure"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", default="paper_reported.csv")
    args = ap.parse_args()
    df = load(args.results)
    if df.empty:
        raise SystemExit("no results found")
    out = os.path.join(args.results, "tables")
    os.makedirs(out, exist_ok=True)
    df.sort_values(["method", "encoder", "lang", "shots", "seed"]).to_csv(os.path.join(out, "all_runs.csv"), index=False)
    for name, t in [("reproduction", reproduction(df, args.paper)),
                    ("two_by_two", two_by_two(df)), ("resources", resources(df))]:
        t.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        with open(os.path.join(out, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(md(t) + "\n")
        print(f"\n## {name}\n" + md(t))
    print("\nAccuracy is top-1 on the test split (%), mean ± sample std over seeds; "
          "deltas are paired by seed (same few-shot images).")


if __name__ == "__main__":
    main()
