#!/bin/bash
# Oct 2026 follow-up campaign (referee-driven checks).  Same conventions as launch64.sh:
# resumable, skips finished/running cells, MAXJOBS concurrent single-core jobs.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=${GPAW_PYTHON:-python}   # python of an environment with GPAW + ASE
MAXJOBS=${MAXJOBS:-6}
nrun(){ pgrep -f "python[0-9.]* (run64|run_vib64|mol_vib|sp700)\.py" | wc -l | tr -d ' '; }
throttle(){ while [ "$(nrun)" -ge "$MAXJOBS" ]; do sleep 30; done; }
stamp(){ echo "[$(date '+%F %T')] $*"; }

# priority order: Si map (charged O/H complexes, JT/smearing test) first, then SiC
CELLS="si V 0 sharp|si VO 1|si VO -1|si VOH 1|si VOH -1|si VH4 1|si VH4 -1|si V -2 sharp|sic VC -2|sic VCH -1|sic VCH 1|sic VCO 0|sic VCOH 0|sic VSiH2 0|sic VSiH3 0|sic VSiH4 0"

IFS='|'
for spec in $CELLS; do
  IFS=' ' read -r sys cfg q opt <<< "$spec"
  suf=""; env=""
  [ "$opt" = "sharp" ] && { suf="_sharp"; env="VARIANT=sharp"; }
  tag="${sys}_${cfg}_q${q}${suf}"; f="res64_${tag}.json"
  if [ -f "$f" ] && $P -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if d.get('fmax_final', 0) < 0.05 else 1)"; then stamp "skip $tag (done)"; continue; fi
  if [ -n "$(find . -maxdepth 1 -name "gp64_${tag}.txt" -mmin -15 2>/dev/null)" ]; then stamp "skip $tag (running)"; continue; fi
  throttle
  stamp "launch $tag"
  env $env $P run64.py "$sys" "$cfg" "$q" > "log64_${tag}.out" 2>&1 &
  sleep 5
done
unset IFS

for m in H2 H2O; do
  [ -f "molvib_${m}.json" ] && { stamp "skip molvib $m"; continue; }
  pgrep -f "mol_vib.py ${m}\$" >/dev/null && { stamp "skip molvib $m (running)"; continue; }
  throttle; stamp "launch molvib $m"
  $P mol_vib.py "$m" > "log_molvib_${m}.out" 2>&1 &
  sleep 5
done
for c in VO VOH; do
  [ -f "vibres64_${c}.json" ] && { stamp "skip vibres64_$c"; continue; }
  [ -n "$(find . -maxdepth 1 -name "vgp64_${c}.txt" -mmin -15 2>/dev/null)" ] && { stamp "skip vib64 $c (running)"; continue; }
  pgrep -f "run_vib64.py ${c}\$" >/dev/null && { stamp "skip vib64 $c (queued in chain)"; continue; }
  throttle; stamp "launch vib64 $c"
  $P run_vib64.py "$c" > "vlog64_${c}.out" 2>&1 &
  sleep 5
done
# 700 eV single points for the new neutral O-containing / hydrogenated SiC cells
for t in sic_VCO_q0 sic_VCOH_q0 sic_VSiH2_q0 sic_VSiH3_q0 sic_VSiH4_q0; do
  while [ ! -f "res64_${t}.json" ]; do sleep 120; done
  [ -f "res700_${t}.json" ] && { stamp "skip sp700 $t"; continue; }
  throttle; stamp "launch sp700 $t"
  $P sp700.py "$t" > "log700_${t}.out" 2>&1 &
  sleep 5
done
wait
stamp "ALL DONE"
