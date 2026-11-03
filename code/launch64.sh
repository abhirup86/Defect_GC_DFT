#!/bin/bash
# Production 64-atom campaign.  Resumable and duplicate-safe: a cell is skipped if its
# result exists with |F|max < 0.05 eV/A, or if its GPAW log was touched in the last 15 min
# (i.e. it is running).  A 4th field "hs" runs the high-spin (ferromagnetic) seed, "ideal"
# starts from the ideal capped geometry without MACE pre-relaxation (variant check).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=${GPAW_PYTHON:-python}   # python of an environment with GPAW + ASE
MAXJOBS=${MAXJOBS:-6}
nrun(){ pgrep -f "run64.py|run_vib64.py" | wc -l | tr -d ' '; }
throttle(){ while [ "$(nrun)" -ge "$MAXJOBS" ]; do sleep 30; done; }
stamp(){ echo "[$(date '+%F %T')] $*"; }

TIER1="si V 0|sic VC 0|sic VSi 0 hs|sic VCH 0 ideal|si VH 0 fsm|sic VCH 0 fsm|si VH 0|si VH2 0|si VH3 0|si VH4 0|sic VSi 0|sic VCH 0|sic VSiH 0|si perfectG 0|sic perfectG 0|si VO 0|si VOH 0"
TIER2="si V 1|si V -1|si V -2|sic VC 1|sic VC -1|si VH 1|si VH -1|si VH3 1|si VH3 -1"
# ---- tier 3 (after the LVMs): fixed-spin-moment checks of the odd-electron charged cells
TIER3="si V 1 fsm|si V -1 fsm|sic VC 1 fsm|sic VC -1 fsm"

launch_list(){
IFS='|'
for spec in $1; do
  IFS=' ' read -r sys cfg q opt <<< "$spec"
  suf=""; env=""
  [ "$opt" = "hs" ] && { suf="_hs"; env="SPINSEED=high"; }
  [ "$opt" = "ideal" ] && { suf="_ideal"; env="VARIANT=ideal"; }
  [ "$opt" = "fsm" ] && { suf="_fsm"; env="VARIANT=fsm"; }
  tag="${sys}_${cfg}_q${q}${suf}"; f="res64_${tag}.json"
  if [ -f "$f" ] && $P -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if d.get('fmax_final', 0) < 0.05 else 1)"; then stamp "skip $tag (done)"; continue; fi
  if [ -n "$(find . -maxdepth 1 -name "gp64_${tag}.txt" -mmin -15 2>/dev/null)" ]; then stamp "skip $tag (running)"; continue; fi
  throttle
  stamp "launch $tag"
  env $env $P run64.py "$sys" "$cfg" "$q" > "log64_${tag}.out" 2>&1 &
  sleep 5
done
unset IFS
}
launch_list "$TIER1|$TIER2"

for c in VH VH2 VH3 VH4; do
  [ -f "vibres64_${c}.json" ] && { stamp "skip vibres64_$c"; continue; }
  while [ ! -f "res64_si_${c}_q0.json" ]; do sleep 60; done
  throttle
  stamp "launch vib64 $c"
  $P run_vib64.py "$c" > "vlog64_${c}.out" 2>&1 &
  sleep 5
done
launch_list "$TIER3"
wait
stamp "ALL DONE"
