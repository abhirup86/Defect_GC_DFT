"""
prerelax64.py -- MACE-MP-0 pre-relaxation of every neutral 64-atom defect geometry.
Runs in the mace_mps env.  Reads init64_<sys>_<cfg>_q0.xyz (written by
`DRY=1 python run64.py <sys> <cfg> 0`), relaxes with the universal MACE potential
(fmax 0.02 eV/A, 300 steps, neutral, no spin), and writes init64_<sys>_<cfg>_pre.xyz,
which run64.py picks up as its starting geometry.  Only a starting point: every
structure is subsequently relaxed with DFT.
"""
import sys, glob, os
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_mp
calc = mace_mp(model='medium', default_dtype='float64', device='cpu')
for f in sorted(glob.glob('init64_*_q0.xyz')):
    if 'perfect' in f or f.startswith('init64_mol') or f.startswith('init64_ref'): continue
    out = f.replace('_q0.xyz', '_pre.xyz')
    if os.path.exists(out): continue
    a = read(f); a.calc = calc
    opt = BFGS(a, logfile=f.replace('.xyz', '_mace.log'), maxstep=0.1); opt.run(fmax=0.02, steps=300)
    write(out, a); print(f'{f}: {opt.get_number_of_steps()} steps, E={a.get_potential_energy():.3f}', flush=True)
