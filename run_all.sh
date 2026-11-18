#!/usr/bin/env bash
# Full experiment. Every number in the report comes from the files this script writes.
set -euo pipefail
SHOTS="${SHOTS:-1 4 16}"
SEEDS="${SEEDS:-1 2 3}"
F=features/eurosat_vitb32.pt
SPLIT=""; [ -f splits/split_zhou_EuroSAT.json ] && SPLIT="--split-json splits/split_zhou_EuroSAT.json"

python env_info.py
python -c "import torch,sys; ok=torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'NONE'); sys.exit(0 if ok or '${ALLOW_CPU:-}' else 1)" || { echo "No GPU. Switch the Colab runtime to GPU (or set ALLOW_CPU=1)."; exit 1; }
[ -f "$F" ] || python extract_features.py --out "$F"
python analyze_tokens.py
python run_zeroshot.py      --features $F $SPLIT
python run_linear_probe.py  --features $F $SPLIT --shots $SHOTS --seeds $SEEDS
python run_tip_adapter.py   --features $F $SPLIT --shots $SHOTS --seeds $SEEDS
python run_coop.py --encoder clip  --features $F $SPLIT --shots $SHOTS --seeds $SEEDS
python run_coop.py --encoder mclip --features $F $SPLIT --shots $SHOTS --seeds $SEEDS
python make_tables.py
python make_figures.py
echo "DONE: see results/tables and results/figures"
