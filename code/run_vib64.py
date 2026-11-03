"""
run_vib64.py <VH|VH2|VH3|VH4|VO|VOH>

Local vibrational modes of the Si vacancy-hydrogen complexes in the 64-atom
cell.  Starts from the k-point-converged relaxed geometry (final64_si_<cfg>_q0.xyz
from run64.py), re-relaxes at the Gamma point (the LVMs are localized on H, and
Gamma-only forces are what the finite-difference Hessian is built from), then
builds the partial dynamical matrix for the H atoms plus their four Si
neighbours by central finite differences (+/- 0.015 A).

Writes vibres64_<cfg>.json.
"""
import sys, json, numpy as np
from ase.io import read
from ase.optimize import BFGS
from ase.data import atomic_masses, atomic_numbers
from gpaw import GPAW, PW, FermiDirac

ECUT = 340
cfg = sys.argv[1]
nH = {'VH': 1, 'VH2': 2, 'VH3': 3, 'VH4': 4, 'VO': 0, 'VOH': 1}[cfg]   # VO/VOH: Oct 2026, for ZPE
tag = f'si_{cfg}_q0'
atoms = read(f'final64_{tag}.xyz')
rec0 = json.load(open(f'res64_{tag}.json'))
atoms.set_initial_magnetic_moments(rec0['magmoms_local'])

calc = GPAW(mode=PW(ECUT), xc='PBE', kpts=(1, 1, 1), symmetry='off', spinpol=True,
            occupations=FermiDirac(0.05),
            convergence={'energy': 1e-5, 'density': 1e-5, 'forces': 1e-3},
            txt=f'vgp64_{cfg}.txt')
atoms.calc = calc
BFGS(atoms, logfile=f'vopt64_{cfg}.log', maxstep=0.1).run(fmax=0.01, steps=100)

Hidx = [a.index for a in atoms if a.symbol == 'H']
nn = rec0['nn_indices']
Oidx = [a.index for a in atoms if a.symbol == 'O']
active = Hidx + Oidx + [int(i) for i in nn]
masses = np.array([atomic_masses[atomic_numbers[atoms[i].symbol]] for i in active])

delta = 0.015
pos0 = atoms.get_positions().copy()
n = len(active)
H = np.zeros((3 * n, 3 * n))
for a, ia in enumerate(active):
    for al in range(3):
        p = pos0.copy(); p[ia, al] += delta; atoms.set_positions(p)
        Fp = atoms.get_forces()[active].ravel()
        p = pos0.copy(); p[ia, al] -= delta; atoms.set_positions(p)
        Fm = atoms.get_forces()[active].ravel()
        H[3 * a + al, :] = -(Fp - Fm) / (2 * delta)
atoms.set_positions(pos0)
H = 0.5 * (H + H.T)
minv = np.repeat(1.0 / np.sqrt(masses), 3)
D = H * np.outer(minv, minv)
evals, evecs = np.linalg.eigh(D)
conv = 521.47  # sqrt(eV/A^2/amu) -> cm^-1
freqs = [float(np.sign(e) * np.sqrt(abs(e)) * conv) for e in evals]
# H participation of each mode
nHdof = 3 * len(Hidx)
part = [float((evecs[:nHdof, i] ** 2).sum()) for i in range(len(freqs))]
stretch = [round(f, 1) for f, p in zip(freqs, part) if f > 1500 and p > 0.5]
zpe = 0.5 * sum(f for f in freqs if f > 0) / 8065.54   # eV, harmonic, active atoms only
json.dump({'cfg': cfg, 'nH': nH, 'natoms': len(atoms), 'active_atoms': active,
           'freqs_cm': freqs, 'zpe_eV_active': zpe, 'E_gamma_relaxed': float(atoms.get_potential_energy()), 'H_participation': part, 'stretch_cm': stretch,
           'XH_bonds': [round(float(min(atoms.get_distances(h, nn, mic=True))), 4) for h in Hidx]},
          open(f'vibres64_{cfg}.json', 'w'), indent=2)
print(f"DONE vib64 {cfg}: stretch={stretch}", flush=True)
