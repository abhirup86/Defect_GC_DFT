"""
mol_vib.py <H2|H2O> -- relaxed geometry and harmonic frequencies of the molecular
references in the same setup as run64.py (PBE, PW 340 eV, spin-polarized, 0.05 eV
Fermi-Dirac, 10 A vacuum box, Gamma).  Oct 2026: zero-point energies for the
vibrational free-energy correction, and the energy change on relaxing the
(unrelaxed, ASE g2 geometry) references used in res64_mol_*.json.
Writes molvib_<mol>.json.
"""
import sys, json, numpy as np
from ase.build import molecule
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from gpaw import GPAW, PW, FermiDirac
name = sys.argv[1]
m = molecule(name); m.center(vacuum=5.0); m.pbc = True
m.calc = GPAW(mode=PW(340), xc='PBE', kpts=(1, 1, 1), spinpol=True, symmetry='off', occupations=FermiDirac(0.05),
              convergence={'energy': 1e-6, 'density': 1e-6, 'forces': 1e-4}, txt=f'gpmolvib_{name}.txt')
E_unrelaxed = m.get_potential_energy()
BFGS(m, logfile=f'optmolvib_{name}.log').run(fmax=0.005)
E = m.get_potential_energy()
vib = Vibrations(m, name=f'molvib_{name}', delta=0.01); vib.run()
freqs = [complex(f) for f in vib.get_frequencies()]
real = sorted(f.real for f in freqs if abs(f.imag) < 1e-6 and f.real > 500)   # internal modes only
zpe = 0.5 * sum(real) / 8065.54
json.dump({'mol': name, 'E_unrelaxed_340': E_unrelaxed, 'E_relaxed_340': E, 'dE_relax': E - E_unrelaxed,
           'freqs_cm_internal': real, 'zpe_eV': zpe,
           'positions': m.get_positions().tolist()}, open(f'molvib_{name}.json', 'w'), indent=2)
print(f'DONE molvib {name}: dE_relax={E - E_unrelaxed:+.4f} eV  freqs={real}  ZPE={zpe:.3f} eV', flush=True)
