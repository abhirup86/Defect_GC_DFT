#!/bin/bash
# Regenerate every number, SI table and figure of the paper from the stored DFT results
# (no DFT needed; ~1 min). Output goes to work/paper/.
set -eu
cd "$(dirname "$0")"
PY=${PYTHON:-python}
rm -rf work && mkdir -p work/paper/figs
cd work
ln -s ../data/results_json/* ../data/structures/* .
rm -f assembled64.json species_Si_64.csv      # rebuilt below
ln -s ../code/render render
$PY ../code/assemble64.py
$PY ../code/make_figs_jpcc.py
echo "Done: work/paper/numbers.tex, work/paper/si_table_*.tex, work/paper/figs/"
