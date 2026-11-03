"""Box-size convergence of the H2 / H2O reference energies (PBE, PW 340 eV, spin-polarized)."""
import sys, json
from ase.build import molecule
from gpaw import GPAW, PW, FermiDirac
name, vac = sys.argv[1], float(sys.argv[2]); ecut = float(sys.argv[3]) if len(sys.argv) > 3 else 340.0
m = molecule(name); m.center(vacuum=vac); m.pbc = True
m.calc = GPAW(mode=PW(ecut), xc='PBE', kpts=(1,1,1), spinpol=True, occupations=FermiDirac(0.05),
              convergence={'energy': 1e-5, 'density': 1e-5}, txt=f'gpbox_{name}_{vac:.0f}_{ecut:.0f}.txt')
E = m.get_potential_energy()
json.dump({'name': name, 'vacuum': vac, 'ecut': ecut, 'cell': m.cell.lengths().tolist(), 'E': E}, open(f'resbox_{name}_{vac:.0f}_{ecut:.0f}.json', 'w'))
print(f'{name} vacuum={vac} ecut={ecut:.0f} cell={m.cell.lengths().round(2)} E={E:.4f}')
