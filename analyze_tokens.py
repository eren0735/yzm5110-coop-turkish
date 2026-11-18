"""How do the two tokenizers split the English and Turkish class names?

Writes results/tables/tokens.md. Mechanistic evidence for the language-mismatch hypothesis:
if CLIP's English BPE cuts Turkish words into many meaningless pieces, its text features
for Turkish names are expected to be less aligned with the images.
"""
import argparse
import os

from coopt.data import CLASSNAMES


def clip_pieces(text):
    from clip.simple_tokenizer import SimpleTokenizer
    t = clip_pieces.tok = getattr(clip_pieces, "tok", None) or SimpleTokenizer()
    out = []
    for i in t.encode(text):  # map byte-level symbols back to readable UTF-8
        piece = t.decoder[i].replace("</w>", "_")
        raw = bytearray(t.byte_decoder[c] for c in piece if c != "_" or False)
        out.append(raw.decode("utf-8", errors="replace") + ("_" if piece.endswith("_") else ""))
    return out


def xlmr_pieces(text, name="M-CLIP/XLM-Roberta-Large-Vit-B-32"):
    import transformers
    t = xlmr_pieces.tok = getattr(xlmr_pieces, "tok", None) or transformers.AutoTokenizer.from_pretrained(name)
    return t.tokenize(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-xlmr", action="store_true")
    args = ap.parse_args()
    rows = ["| lang | class name | CLIP BPE pieces | # | XLM-R pieces | # |", "|---|---|---|---|---|---|"]
    totals = {}
    for lang, names in CLASSNAMES.items():
        for n in names:
            c = clip_pieces(n)
            x = [] if args.no_xlmr else xlmr_pieces(n)
            totals.setdefault(lang, [0, 0, 0])
            totals[lang][0] += len(c)
            totals[lang][1] += len(x)
            totals[lang][2] += len(n.split())
            rows.append(f"| {lang} | {n} | {' '.join(c)} | {len(c)} | {' '.join(x)} | {len(x)} |")
    rows.append("")
    for lang, (c, x, w) in totals.items():
        rows.append(f"- **{lang}**: {w} words -> CLIP {c} pieces ({c / w:.2f}/word), "
                    f"XLM-R {x} pieces ({x / w:.2f}/word)")
    out = os.path.join(args.results, "tables", "tokens.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
