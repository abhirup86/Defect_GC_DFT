"""sp700.py <tag> [ecut]: single-point energy at a higher cutoff on the relaxed 64-atom
geometry (final64_<tag>.xyz), same mesh/charge/spin as run64.py. Writes res<ecut>_<tag>.json."""
import sys, json, time, numpy as np
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
tag = sys.argv[1]; ecut = float(sys.argv[2]) if len(sys.argv) > 2 else 700.0
r = json.load(open(f'res64_{tag}.json')); a = read(f'final64_{tag}.xyz')
a.set_initial_magnetic_moments(r['magmoms_local'])
kpts = {'size': (4, 4, 4), 'gamma': True} if 'perfectG' in tag else tuple(r['kpts'])
t0 = time.time()
a.calc = GPAW(mode=PW(ecut), xc='PBE', kpts=kpts, spinpol=True, charge=r['q'],
              occupations=FermiDirac(r.get('smearing_eV', 0.05)), convergence={'energy': 1e-4, 'density': 1e-4},
              txt=f'gp{ecut:.0f}_{tag}.txt')
E = a.get_potential_energy()
json.dump({'tag': tag, 'ecut': ecut, 'E': E, 'E340': r['E'], 'magmom': float(a.get_magnetic_moment()),
           'walltime_s': round(time.time() - t0, 1)}, open(f'res{ecut:.0f}_{tag}.json', 'w'), indent=2)
print(f'DONE {tag} @ {ecut:.0f} eV: E={E:.4f} (340 eV: {r["E"]:.4f})', flush=True)
