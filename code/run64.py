"""
run64.py <system> <config> [q]

Production driver: relax one vacancy complex in a 64-atom cubic supercell
(2x2x2 conventional cell), spin-polarized with seeded initial moments, and
write res64_<system>_<config>_q<q>.json.

  system : si | sic | mol
  config : perfect V VH VH2 VH3 VH4 VO VOH          (si)
           perfect VC VSi VCH VSiH                  (sic)
           VCO VCOH VSiH2 VSiH3 VSiH4               (sic, Oct 2026 additions)
           H2 H2O                                   (mol)
           C_diamond                                (ref; EOS reference)
  q      : net charge (default 0); positive = electrons removed

Unified parameters for every solid-state calculation:
  PBE, PW(340 eV), shifted 2x2x2 Monkhorst-Pack k-mesh (band edges: 'perfectG',
  Gamma-centred 4x4x4), Fermi-Dirac 0.05 eV,
  PBE equilibrium lattice constants from the EOS fits (bulkres_*.json),
  Geometry: MACE-MP pre-relaxation (prerelax64.py) -> BFGS on the 2x2x2 mesh to |F| < 0.03 eV/A.
  Charged states start from the relaxed neutral cell; an existing final64_<tag>.xyz is
  used as a continuation start (a rerun of a cell that did not converge).

Symmetry is broken deliberately (antiferromagnetic seed on the dangling-bond
atoms plus a 0.02 A rattle) so that Jahn-Teller distortions and broken-
symmetry singlets are accessible.  For charged cells the spherically averaged
electrostatic potential at every atom is stored so that a potential-alignment
term can be evaluated in the post-processing (assemble64.py).
"""
import sys, json, time, os
import numpy as np
from ase.build import bulk, molecule
from ase import Atom, Atoms
from ase.io import write
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac

ECUT = float(os.environ.get('ECUT', 340))   # production 340 eV; ECUT=700 for the cutoff check (tag suffix _e700)
KPTS = (2, 2, 2)            # shifted Monkhorst-Pack mesh (4 irreducible points)
KPTS_BAND = (4, 4, 4)       # Gamma-centred mesh for band edges (config 'perfectG')
FMAX = 0.03
MAXSTEPS = 80
A_SI  = 5.475608518723767   # PBE EOS, bulkres_si.json
A_SIC = 4.390074298678462   # PBE EOS, bulkres_sic.json
BOND  = {'Si': 1.50, 'C': 1.10}

system, config = sys.argv[1], sys.argv[2]
q = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SPINSEED = os.environ.get('SPINSEED', 'low')       # 'low' (AFM/doublet) or 'high' (ferromagnetic)
VARIANT = os.environ.get('VARIANT', '')            # 'ideal': start from the ideal capped geometry, no MACE/previous file
                                                   # 'fsm'  : rerun tag with the fixed-spin-moment constraint (continuation)
                                                   # 'sharp': 0.01 eV smearing, unconstrained moment, D2d double-pairing seed
                                                   #          (tests whether the 0.05 eV smearing suppresses the JT distortion)
tag = f'{system}_{config}_q{q}' + ('_hs' if SPINSEED == 'high' else '') + (f'_{VARIANT}' if VARIANT else '') + (f'_e{ECUT:.0f}' if ECUT != 340 else '')
out = f'res64_{tag}.json'
t0 = time.time()

# ------------------------------------------------------------------ geometry
def mic_dist(atoms, point, idx=None):
    tmp = atoms.copy(); tmp.append(Atom('X', point))
    idx = range(len(atoms)) if idx is None else idx
    return tmp.get_distances(len(tmp) - 1, list(idx), mic=True)

def unit_toward(atoms, j, point):
    tmp = atoms.copy(); tmp.append(Atom('X', point))
    v = tmp.get_distance(j, len(tmp) - 1, mic=True, vector=True)
    return v / np.linalg.norm(v)

def build():
    if system == 'ref' and config == 'C_diamond':
        return bulk('C', 'diamond', a=3.57), [], None
    if system == 'mol':
        m = molecule(config); m.center(vacuum=5.0); m.pbc = True
        return m, [], None
    if system == 'si':
        sc = bulk('Si', 'diamond', a=A_SI, cubic=True) * (2, 2, 2)
    elif system == 'sic':
        sc = bulk('SiC', 'zincblende', a=A_SIC, cubic=True) * (2, 2, 2)
    else:
        raise SystemExit('bad system')
    assert len(sc) == 64
    if config in ('perfect', 'perfectG'):
        return sc, [], None

    if system == 'si':
        rem = 0
    elif config in ('VC', 'VCH', 'VCO', 'VCOH'):
        rem = [a.index for a in sc if a.symbol == 'C'][0]
    else:
        rem = [a.index for a in sc if a.symbol == 'Si'][0]
    center = sc[rem].position.copy()
    work = sc.copy(); del work[rem]
    d = mic_dist(work, center)
    neigh = [int(i) for i in np.argsort(d)[:4]]          # 4 dangling-bond atoms

    nH = {'V': 0, 'VH': 1, 'VH2': 2, 'VH3': 3, 'VH4': 4, 'VO': 0, 'VOH': 1,
          'VC': 0, 'VSi': 0, 'VCH': 1, 'VSiH': 1,
          'VCO': 0, 'VCOH': 1, 'VSiH2': 2, 'VSiH3': 3, 'VSiH4': 4}[config]
    capped = neigh[:nH]
    if config in ('VO', 'VOH', 'VCO', 'VCOH'):
        # oxygen starts between two of the four neighbours (A-centre-like
        # bridging start), not at the high-symmetry centre
        u1 = unit_toward(work, neigh[0], center); u2 = unit_toward(work, neigh[1], center)
        d1, d2 = d[neigh[0]], d[neigh[1]]
        opos = center - 0.5 * (u1 * d1 + u2 * d2)           # midpoint of the two neighbours (MIC)
        work.append(Atom('O', opos))
        capped = [neigh[0], neigh[1]]
        if config in ('VOH', 'VCOH'):
            # hydrogen on the third neighbour (dangling bond) and OH on O is
            # sampled separately; here H caps neighbour 3 -> V-OH = O bridging + Si-H
            j = neigh[2]
            work.append(Atom('H', work[j].position + unit_toward(work, j, center) * BOND['Si']))
            capped.append(j)
    else:
        for j in capped:
            sym = work[j].symbol
            work.append(Atom('H', work[j].position + unit_toward(work, j, center) * BOND[sym]))
    dangling = [j for j in neigh if j not in capped]
    return work, dangling, center

atoms, dangling, center = build()
if config not in ('perfect', 'perfectG') and system not in ('mol', 'ref'):
    atoms.rattle(stdev=0.02, seed=42)                        # break Td symmetry
    pre = f'init64_{system}_{config}_pre.xyz'                # MACE pre-relaxed (prerelax64.py)
    if os.path.exists(pre) and VARIANT != 'ideal':
        from ase.io import read
        p = read(pre); assert p.get_chemical_formula() == atoms.get_chemical_formula()
        atoms.set_positions(p.get_positions()); prerelaxed = True
    else:
        prerelaxed = False
else:
    prerelaxed = False
if os.environ.get('DRY'):
    print(tag, atoms.get_chemical_formula(), 'dangling', dangling, 'prerelaxed', prerelaxed)
    if center is not None:
        d = mic_dist(atoms, center)
        for a in atoms:
            if a.symbol in ('H','O') or d[a.index] < 2.5:
                print(f"  {a.index:3d} {a.symbol:2s} d_vac={d[a.index]:.3f}")
    write(f'init64_{tag}.xyz', atoms)
    raise SystemExit

# ---------------------------------------------------------------- spin seed
magmoms = np.zeros(len(atoms))
if system == 'mol':
    pass
elif dangling and SPINSEED == 'high':
    for j in dangling:                                       # high-spin (Hund) seed
        magmoms[j] = 1.0
elif dangling:
    n_el = sum(atoms.get_atomic_numbers()) - q
    if n_el % 2 == 1:
        magmoms[dangling[0]] = 1.0                           # doublet seed
        for j in dangling[1:]:
            magmoms[j] = 0.5 * (-1) ** dangling.index(j)     # AFM on the rest
    else:
        for k, j in enumerate(dangling):                     # broken-symmetry singlet
            magmoms[j] = 0.6 * (-1) ** k
atoms.set_initial_magnetic_moments(magmoms)

# ------------------------------------------------------------------ calculator
if system == 'mol':
    kpt_arg = (1, 1, 1)
elif system == 'ref':
    kpt_arg = {'size': (8, 8, 8), 'gamma': True}
elif config == 'perfectG':
    kpt_arg = {'size': KPTS_BAND, 'gamma': True}
else:
    kpt_arg = KPTS                      # shifted MP mesh; symmetry detection (incl.
                                        # time reversal) is left on: the rattle has
                                        # already removed the point-group symmetry
# Odd-electron cells are run with a fixed total moment of 1 mu_B (doublet): with
# Fermi smearing an unconstrained calculation can collapse to a delocalized,
# spin-unpolarized solution with a half-filled level at E_F.
n_el_total = int(sum(atoms.get_atomic_numbers()) - q) if system not in ('mol', 'ref') else 0
# Default: unconstrained (the seeded doublet localizes for VH3, VSiH, ...).  VARIANT=fsm
# imposes M = 1 mu_B as a check when an odd-electron cell collapses to M = 0 (VH in Si:
# the constrained doublet is delocalized and 0.07 eV higher, i.e. the PBE ground state
# of that cell is the spin-unpolarized solution).
FIXMAGMOM = (VARIANT == 'fsm') and (n_el_total % 2 == 1)
if FIXMAGMOM and abs(magmoms.sum() - 1.0) > 1e-6:
    magmoms[:] = 0.0
    if dangling:
        magmoms[dangling[0]] = 1.0
    else:
        magmoms[0] = 1.0
    atoms.set_initial_magnetic_moments(magmoms)

# The 'fsm' check also sharpens the smearing (0.01 eV): with 0.05 eV a partially filled
# degenerate level is occupied fractionally, which stabilises the symmetric geometry
# against the Jahn-Teller distortion and gives fractional total moments.
SMEAR = 0.01 if VARIANT in ('fsm', 'sharp') else 0.05
def make_calc(kpts, txt, conv=None):
    return GPAW(mode=PW(ECUT), xc='PBE', kpts=kpts, spinpol=True, charge=q,
                occupations=FermiDirac(SMEAR, fixmagmom=FIXMAGMOM), convergence=conv or {}, txt=txt)

# starting geometry: (i) a previous run of this very cell (continuation),
# (ii) the relaxed neutral cell for a charged state, (iii) the MACE pre-relaxed cell
from ase.io import read
start = None
for cand in (f'start64_{tag}.xyz',                                   # explicit start
             f'final64_{tag}.xyz',                                   # continuation
             f'final64_{system}_{config}_q{q}.xyz' if VARIANT != 'ideal' else None,   # other seed / unconstrained run
             f'final64_{system}_{config}_q0.xyz' if (q != 0 and VARIANT != 'ideal') else None):  # neutral -> charged
    if cand and os.path.exists(cand):
        start = cand; break
if start:
    p = read(start); assert p.get_chemical_formula() == atoms.get_chemical_formula()
    atoms.set_positions(p.get_positions())
rec_start = start or ('mace' if prerelaxed else 'ideal+rattle')

# Symmetry breaking AFTER the starting geometry is loaded (a pre-relaxed or
# previously relaxed structure may sit on a symmetric saddle point): random
# 0.05 A displacements plus an explicit dangling-bond pairing (Jahn-Teller) seed
# that moves the first two dangling-bond atoms 0.15 A towards each other.
if config not in ('perfect', 'perfectG') and system not in ('mol', 'ref'):
    atoms.rattle(stdev=0.05 if VARIANT != 'fsm' else 0.02, seed=7)
    if VARIANT == 'sharp' and len(dangling) == 4:
        # D2d seed: both dangling-bond pairs (0-1, 2-3) moved 0.20 A towards each other
        pos = atoms.get_positions()
        for i, j in ((dangling[0], dangling[1]), (dangling[2], dangling[3])):
            v = atoms.get_distance(i, j, mic=True, vector=True); u = v / np.linalg.norm(v)
            pos[i] += 0.20 * u; pos[j] -= 0.20 * u
        atoms.set_positions(pos)
        rec_seed = 'D2d double pairing'
    elif len(dangling) >= 2:
        i, j = dangling[0], dangling[1]
        v = atoms.get_distance(i, j, mic=True, vector=True); u = v / np.linalg.norm(v)
        pos = atoms.get_positions(); pos[i] += 0.15 * u; pos[j] -= 0.15 * u
        atoms.set_positions(pos)
        rec_seed = f'pairing {i}-{j}'
    else:
        rec_seed = 'rattle only'
else:
    rec_seed = None

calc = make_calc(kpt_arg, f'gp64_{tag}.txt', {'energy': 1e-4, 'density': 1e-4})
atoms.calc = calc

rec = {'system': system, 'config': config, 'q': q, 'ecut': ECUT,
       'kpts': (list(KPTS_BAND) if config == 'perfectG' else list(KPTS)) if system != 'mol' else [1, 1, 1],
       'a_lattice': A_SI if system == 'si' else (A_SIC if system == 'sic' else None),
       'natoms': len(atoms), 'formula': atoms.get_chemical_formula(),
       'initial_magmoms_total': float(magmoms.sum()), 'start_geometry': rec_start,
       'symmetry_seed': rec_seed, 'fixmagmom': FIXMAGMOM, 'smearing_eV': SMEAR, 'n_electrons_total': n_el_total}

if system == 'ref':
    # equation of state for the elemental reference (7 volumes, +-4 % strain)
    from ase.eos import EquationOfState
    cell0 = atoms.cell.copy(); vols, ens = [], []
    for f in np.linspace(0.96, 1.04, 7):
        atoms.set_cell(cell0 * f, scale_atoms=True)
        vols.append(atoms.get_volume()); ens.append(atoms.get_potential_energy())
    eos = EquationOfState(vols, ens, eos='birchmurnaghan'); v0, e0, B = eos.fit()
    a0 = (v0 * 4) ** (1 / 3)                      # primitive fcc -> conventional a
    rec.update({'E': e0, 'a0': a0, 'B_GPa': B / 1.602176634e-19 * 1e21 * 1e-9,
                'E_per_atom': e0 / len(atoms), 'eos_volumes': vols, 'eos_energies': ens,
                'relaxed': False, 'walltime_s': round(time.time() - t0, 1)})
    json.dump(rec, open(out, 'w'), indent=2)
    print(f"DONE {tag}: E/atom={e0/len(atoms):.4f} eV a0={a0:.4f} A", flush=True)
    raise SystemExit
E0 = atoms.get_potential_energy()
rec['E_initial'] = E0
rec['fmax_initial'] = float(np.abs(atoms.get_forces()).max())
if config in ('perfect', 'perfectG') or system == 'mol':
    E = E0; rec['relaxed'] = False
else:
    opt = BFGS(atoms, logfile=f'opt64_{tag}.log', maxstep=0.15)
    opt.run(fmax=FMAX, steps=MAXSTEPS)
    E = atoms.get_potential_energy()
    rec['relaxed'] = True
    rec['relax_steps'] = opt.get_number_of_steps()
    rec['fmax_final'] = float(np.abs(atoms.get_forces()).max())
    rec['prerelaxed_mace'] = prerelaxed
rec['E'] = E
rec['magmom'] = float(atoms.get_magnetic_moment())
rec['magmoms_local'] = [round(float(m), 3) for m in atoms.get_magnetic_moments()]
write(f'final64_{tag}.xyz', atoms)

# ------------------------------------------------------------- electronic data
try:
    nelec = calc.get_number_of_electrons()
    nk = len(calc.get_ibz_k_points())
    try:
        ef = calc.get_fermi_level()
    except Exception:                     # fixed-spin-moment: one Fermi level per spin
        ef = float(np.mean(calc.get_fermi_levels()))
    eigs = {}
    for s in range(2):
        eigs[s] = [calc.get_eigenvalues(kpt=k, spin=s).tolist() for k in range(nk)]
        occs = [calc.get_occupation_numbers(kpt=k, spin=s).tolist() for k in range(nk)]
    rec['fermi'] = float(ef); rec['nelec'] = float(nelec)
    near = sorted({round(float(e), 4) for s in eigs for col in eigs[s] for e in col
                   if abs(e - ef) < 3.0})
    rec['eigs_near_ef'] = near
    # spin-resolved eigenvalues within 3 eV of E_F (Gamma point only for plots)
    rec['eigs_gamma'] = {str(s): [round(float(e), 4) for e in eigs[s][0] if abs(e - ef) < 3.0]
                         for s in eigs}
    if config in ('perfect', 'perfectG'):
        # VBM / CBM from occupation-resolved eigenvalues over all k
        # pristine cell: E_F lies mid-gap, so the band edges are the extremal
        # eigenvalues on either side of it (occupation numbers are not needed)
        vb, cb = [], []
        for s in range(2):
            for k in range(nk):
                e = np.array(calc.get_eigenvalues(kpt=k, spin=s))
                vb.append(e[e < ef].max()); cb.append(e[e > ef].min())
        rec['vbm'] = float(max(vb)); rec['cbm'] = float(min(cb))
        rec['gap'] = rec['cbm'] - rec['vbm']
except Exception as e:
    rec['eig_error'] = str(e)

# ------------------------------------------------- potential for alignment
if system != 'mol':
    try:
        v = calc.get_electrostatic_potential()             # eV, fine grid
        cell = atoms.cell.array
        shape = np.array(v.shape)
        idx = np.indices(shape).reshape(3, -1).T / shape   # fractional grid
        cart = idx @ cell
        pos = atoms.get_positions()
        vat = []
        R = 1.2
        for p in pos:
            dv = cart - p
            dv -= np.round(dv @ np.linalg.inv(cell)) @ cell
            m = np.linalg.norm(dv, axis=1) < R
            vat.append(float(v.reshape(-1)[m].mean()))
        rec['v_atomic'] = [round(x, 4) for x in vat]
        if center is not None:
            dcen = mic_dist(atoms, center)
            rec['dist_from_vacancy'] = [round(float(x), 3) for x in dcen]
        else:
            rec['dist_from_vacancy'] = None
    except Exception as e:
        rec['pot_error'] = str(e)

# ---------------------------------------------------- local geometry summary
if center is not None:
    try:
        d = mic_dist(atoms, center)
        host = [i for i in np.argsort(d) if atoms[i].symbol in ('Si', 'C')][:4]
        pair = []
        for i in range(4):
            for j in range(i + 1, 4):
                pair.append(round(float(atoms.get_distance(host[i], host[j], mic=True)), 3))
        rec['nn_pair_distances'] = pair
        rec['nn_indices'] = [int(i) for i in host]
        Hs = [a.index for a in atoms if a.symbol == 'H']
        Os = [a.index for a in atoms if a.symbol == 'O']
        rec['XH_bonds'] = [round(float(min(atoms.get_distances(h, host, mic=True))), 3) for h in Hs]
        if Os:
            rec['O_to_nn'] = [round(float(x), 3) for x in atoms.get_distances(Os[0], host, mic=True)]
            if Hs:
                rec['OH'] = round(float(atoms.get_distance(Os[0], Hs[0], mic=True)), 3)
    except Exception as e:
        rec['geom_error'] = str(e)

rec['walltime_s'] = round(time.time() - t0, 1)
json.dump(rec, open(out, 'w'), indent=2)
print(f"DONE {tag}: E={E:.4f} eV  M={rec['magmom']:.2f}  ({rec['walltime_s']}s)", flush=True)
