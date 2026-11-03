"""
bench_bulk.py <si|sic>
EOS fit -> equilibrium lattice constant, bulk modulus; plus PBE gap and
cohesive energy.  Writes bulkres_<sys>.json.
"""
import sys, json, numpy as np
from ase.build import bulk, molecule
from ase.eos import EquationOfState
from gpaw import GPAW, PW

sysname=sys.argv[1]
ECUT=400
if sysname=='si':
    a0=5.47; sym='diamond'; formula='Si'; kpts=(8,8,8)
else:
    a0=4.36; sym='zincblende'; formula='SiC'; kpts=(8,8,8)

vols,ens,lat=[],[],[]
for eps in np.linspace(-0.03,0.03,7):
    a=a0*(1+eps)
    at=bulk(formula,sym,a=a)
    at.calc=GPAW(mode=PW(ECUT),xc='PBE',kpts=kpts,txt=None)
    ens.append(at.get_potential_energy()); vols.append(at.get_volume()); lat.append(a)
v0,e0,B=EquationOfState(vols,ens).fit()
if sysname=='si':
    aeq=(4*v0)**(1/3)   # diamond: V=a^3/4 for 2-atom prim? use conventional relation
    # bulk('Si','diamond') is 2-atom primitive, V=a^3/4
else:
    aeq=(4*v0)**(1/3)   # zincblende primitive 2-atom, V=a^3/4
B_GPa=B*160.21766208

# gap at equilibrium
at=bulk(formula,sym,a=aeq)
calc=GPAW(mode=PW(ECUT),xc='PBE',kpts={'size':(8,8,8),'gamma':True},txt=None)
at.calc=calc; at.get_potential_energy()
nelec=int(round(calc.get_number_of_electrons()))
homo=nelec//2-1; lumo=nelec//2
nk=len(calc.get_ibz_k_points())
vbm=max(calc.get_eigenvalues(kpt=k)[homo] for k in range(nk))
cbm=min(calc.get_eigenvalues(kpt=k)[lumo] for k in range(nk))

json.dump({'sys':sysname,'a0':aeq,'B_GPa':B_GPa,'gap':cbm-vbm,'e0_per_cell':e0},
          open(f'bulkres_{sysname}.json','w'),indent=2)
print(f"DONE bulk {sysname}: a0={aeq:.4f} A  B={B_GPa:.1f} GPa  gap={cbm-vbm:.3f} eV",flush=True)
