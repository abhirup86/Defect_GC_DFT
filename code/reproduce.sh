#!/bin/bash
# Reproduce all DFT data, aggregation, figures and LaTeX inputs for the JPCC manuscript.
# Resumable: any step whose output already exists is skipped.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GP=${GPAW_PYTHON:-python}   # python with GPAW + ASE
MACE=${MACE_PYTHON:-python}   # python with mace-torch (pre-relaxation only)

echo "### Step 1c: bulk equations of state (lattice constants used by run64.py) ###"
for s in si sic; do
  [ -f "bulkres_${s}.json" ] && { echo "skip bulkres_${s}"; continue; }
  $GP bench_bulk.py "$s" > "blog_${s}.out" 2>&1
done

echo "### Step 0: starting geometries + MACE pre-relaxation ###"
for spec in "si V" "si VH" "si VH2" "si VH3" "si VH4" "si VO" "si VOH" "sic VC" "sic VSi" "sic VCH" "sic VSiH"; do
  set -- $spec
  [ -f "init64_${1}_${2}_pre.xyz" ] || DRY=1 $GP run64.py "$1" "$2" 0 > /dev/null 2>&1
done
$MACE prerelax64.py > prerelax64.out 2>&1

echo "### Step 1a/1b/1d: 64-atom campaign (neutral, charged, references, LVMs; parallel) ###"
[ -f res64_ref_C_diamond_q0.json ] || $GP run64.py ref C_diamond 0 > log64_ref_C_diamond_q0.out 2>&1
MAXJOBS=6 bash launch64.sh >> launch64.out 2>&1

echo "### Step 2: aggregate -> assembled64.json, paper/numbers.tex, SI tables ###"
$GP assemble64.py

echo "### Step 3: figures -> paper/figs/ ###"
$GP make_figs_jpcc.py

echo "### DONE.  Compile: ###"
echo "  cd paper && pdflatex paper_jpcc && bibtex paper_jpcc && pdflatex paper_jpcc && pdflatex paper_jpcc"
echo "  cd paper && pdflatex si_jpcc && bibtex si_jpcc && pdflatex si_jpcc && pdflatex si_jpcc"
