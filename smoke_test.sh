#!/usr/bin/env bash
# ~5 minutes on a Colab T4. Loads the real models, runs every method on a small test subset.
set -euo pipefail
R=results_smoke
F=features/eurosat_vitb32.pt
[ -f "$F" ] || python extract_features.py --out "$F"
python analyze_tokens.py --results $R
python run_zeroshot.py   --features $F --results $R --max-test 500 --repeats 2
python run_linear_probe.py --features $F --results $R --max-test 500 --shots 4 --seeds 1
python run_tip_adapter.py  --features $F --results $R --max-test 500 --shots 4 --seeds 1
python run_coop.py --encoder clip  --features $F --results $R --max-test 500 --shots 4 --seeds 1 --epochs 5
python run_coop.py --encoder mclip --features $F --results $R --max-test 500 --shots 4 --seeds 1 --epochs 5
python make_tables.py  --results $R
python make_figures.py --results $R
echo "SMOKE TEST OK"
