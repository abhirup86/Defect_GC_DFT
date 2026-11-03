"""
assemble64.py -- aggregate the 64-atom production data (res64_*.json, vibres64_*.json,
bulkres_*.json, exp_data.json) into

    assembled64.json      all derived quantities (formation energies, corrections,
                          transition levels, binding energies, LVMs, geometry)
    species_Si_64.csv     input table for hydration_engine.py
    paper/numbers.tex     LaTeX macros so that every number in the manuscript is
                          generated from the data, never typed by hand

Formation energy of species X with charge q, m_H hydrogens, m_O oxygens:

  Ef(q; EF=0, dmu=0) = E[X^q] - E[host] + sum_s n_s mu_s - mH mu_H - mO mu_O
                       + q (E_VBM + dV_align) + E_MP(q)

  mu_H  = 1/2 E[H2]                 (H-rich reference, dmu_H = 0)
  mu_O  = E[H2O] - E[H2]            (oxygen slaved to water)
  dV    = <V_defect - V_host> on atoms farther than R_far from the vacancy
          (spherically averaged electrostatic potential, potential alignment)
  E_MP  = q^2 alpha e^2 / (2 eps L)  (Makov-Payne monopole; alpha = 2.8373)
"""
import json, os, csv, sys, glob
import numpy as np

KE2 = 14.399645          # e^2/(4 pi eps0)  [eV A]
ALPHA_SC = 2.8373        # Madelung constant, simple-cubic point-charge lattice
EPS = {'si': 11.7, 'sic': 9.72}      # experimental static dielectric constants
R_FAR_FRAC = 0.42        # atoms beyond this fraction of L are 'far' for alignment


def load(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


def rec(system, cfg, q=0):
    """Lowest-energy record among the spin seeds tried (low-spin file, optional _hs file)."""
    cands = []
    for suf, lab in (('', 'low'), ('_hs', 'high'), ('_ideal', 'low/ideal-start'), ('_hs_ideal', 'high/ideal-start'),
                     ('_fsm', 'fixed-spin-moment'), ('_ideal_fsm', 'fixed-spin-moment/ideal-start')):
        x = load(f'res64_{system}_{cfg}_q{q}{suf}.json')
        if x and (x.get('fmax_final', 0) < 0.05):
            x['spin_seed'] = lab; cands.append(x)
    if not cands:
        return None
    # lowest energy wins; the other seeds/constraints are recorded as 'variants'
    cands = sorted(cands, key=lambda x: x['E'])
    best = cands[0]
    if len(cands) > 1:
        best['E_other_spin'] = cands[1]['E']
        best['variants'] = [(c['spin_seed'], c['E']) for c in cands]
    return best


# ----------------------------------------------------------------- references
H2, H2O = rec('mol', 'H2'), rec('mol', 'H2O')
E_H2, E_H2O = H2['E'], H2O['E']
mu_H = 0.5 * E_H2
mu_O = E_H2O - E_H2
mu_OH = E_H2O - mu_H
Cdia = rec('ref', 'C_diamond')
# ---- cutoff check / correction (700 eV single points on the 340 eV geometries) ----
# The O (and, weakly, H) PAW data sets are not converged in absolute energy at 340 eV; the
# error cancels between solid and molecular reference only partially (VO: +0.30 eV).
# Where a res700_<tag>.json exists for the neutral cell AND for the host cell, the neutral
# formation energy is evaluated fully at 700 eV (700 eV references, 10 A box); charged states
# of the same decoration receive the same composition-dependent shift, so transition levels
# are unchanged.
E700 = {}
for p in glob.glob('res700_*.json'):
    d = load(p); E700[d['tag']] = d['E']
REF700 = {}
try:
    REF700 = {'H2': load('resbox_H2_5_700.json')['E'], 'H2O': load('resbox_H2O_5_700.json')['E']}
except Exception:
    pass
Cdia700 = load('res64_ref_C_diamond_q0_e700.json')
out = {'refs': {'E_H2': E_H2, 'E_H2O': E_H2O, 'mu_H_ref': mu_H, 'mu_O_ref': mu_O,
                'mu_OH_ref': mu_OH,
                'mu_C_diamond': Cdia['E_per_atom'] if Cdia else None,
                'a0_C_diamond': Cdia['a0'] if Cdia else None}}
EXP = load('exp_data.json')
BSI, BSIC = load('bulkres_si.json'), load('bulkres_sic.json')
out['bulk'] = {'si': BSI, 'sic': BSIC}

# ------------------------------------------------------------- host cells
def host(system):
    p = rec(system, 'perfect'); g = rec(system, 'perfectG')
    d = {'E': p['E'], 'N': p['natoms'], 'L': 2 * p['a_lattice'], 'a': p['a_lattice'],
         'mu_atom': p['E'] / p['natoms'], 'v_atomic': p.get('v_atomic')}
    if g and 'vbm' not in g and g.get('eigs_near_ef'):
        # fall back on the stored eigenvalues: E_F lies mid-gap for the pristine cell
        e = np.array(g['eigs_near_ef']); ef = g['fermi']
        g['vbm'] = float(e[e < ef].max()); g['cbm'] = float(e[e > ef].min()); g['gap'] = g['cbm'] - g['vbm']
    if g:
        d.update({'vbm': g['vbm'], 'cbm': g['cbm'], 'gap': g['gap'], 'kpts_band': g['kpts']})
    elif 'vbm' in p:
        d.update({'vbm': p['vbm'], 'cbm': p['cbm'], 'gap': p['gap'], 'kpts_band': p['kpts']})
    return d


def removed_index(system, cfg):
    """Index (in the perfect 64-atom cell) of the atom removed to create the vacancy."""
    if system == 'si':
        return 0
    from ase.build import bulk
    sc = bulk('SiC', 'zincblende', a=4.39, cubic=True) * (2, 2, 2)
    sym = 'C' if cfg in ('VC', 'VCH') else 'Si'
    return [a.index for a in sc if a.symbol == sym][0]


def potential_alignment(system, r, H):
    """<V_defect - V_host> over host atoms far from the vacancy (eV)."""
    if not r.get('v_atomic') or not H.get('v_atomic') or r.get('dist_from_vacancy') is None:
        return None, None
    rem = removed_index(system, r['config'])
    vd = np.array(r['v_atomic']); vh = np.array(H['v_atomic'])
    dist = np.array(r['dist_from_vacancy'])
    nhost = H['N'] - 1
    far = [i for i in range(nhost) if dist[i] > R_FAR_FRAC * H['L']]
    diffs = [vd[i] - vh[i + 1 if i >= rem else i] for i in far]
    return float(np.mean(diffs)), len(far)


def makov_payne(q, system, L):
    return q ** 2 * ALPHA_SC * KE2 / (2 * EPS[system] * L)


# ------------------------------------------------------------ species tables
SI_SPECIES = {   # cfg: (mH, mO, charges)
    'V': (0, 0, [1, 0, -1, -2]), 'VH': (1, 0, [1, 0, -1]), 'VH2': (2, 0, [0]),
    'VH3': (3, 0, [1, 0, -1]), 'VH4': (4, 0, [1, 0]), 'VO': (0, 1, [0, -1]), 'VOH': (1, 1, [0, -1]),
}
SIC_SPECIES = {'VC': (0, 0, [1, 0, -1]), 'VCH': (1, 0, [0]), 'VSi': (0, 0, [0]), 'VSiH': (1, 0, [0])}


def analyse(system, table, mu_removed, limit='Si-rich'):
    """mu_removed: dict cfg -> chemical potential of the removed host atom."""
    H = host(system)
    res = {'host': H, 'species': {}, '_limit': limit}
    for cfg, (mH, mO, charges) in table.items():
        for q in charges:
            r = rec(system, cfg, q)
            if r is None:
                continue
            dV, nfar = potential_alignment(system, r, H)
            emp = makov_payne(q, system, H['L'])
            Ef_raw = r['E'] - H['E'] + mu_removed[cfg] - mH * mu_H - mO * mu_O
            Ef = Ef_raw + q * (H['vbm'] + (dV or 0.0)) + emp
            key = f'{cfg}_q{q}'
            sym = load(f'res64_{system}_{cfg}_q{q}_sym.json')      # symmetric (saddle) reference
            res['species'][key] = {
                'E_sym': sym['E'] if sym else None,
                'E_JT': (sym['E'] - r['E']) if sym else None,
                'nn_pair_distances_sym': sym.get('nn_pair_distances') if sym else None,
                'cfg': cfg, 'q': q, 'mH': mH, 'mO': mO, 'E': r['E'],
                'Ef_raw': Ef_raw, 'dV_align': dV, 'n_far_atoms': nfar, 'E_MP': emp,
                'q_times_vbm': q * H['vbm'], 'Ef_ref': Ef,
                'magmom': r.get('magmom'), 'magmoms_local_max': (max(abs(m) for m in r['magmoms_local'])
                                                                 if r.get('magmoms_local') else None),
                'relax_steps': r.get('relax_steps'), 'fmax_final': r.get('fmax_final'),
                'nn_pair_distances': r.get('nn_pair_distances'), 'XH_bonds': r.get('XH_bonds'),
                'O_to_nn': r.get('O_to_nn'), 'OH': r.get('OH'),
                'spin_seed': r.get('spin_seed'), 'E_other_spin': r.get('E_other_spin'),
                'fermi': r.get('fermi'), 'eigs_near_ef': r.get('eigs_near_ef'),
                'eigs_gamma': r.get('eigs_gamma'), 'walltime_s': r.get('walltime_s'),
            }
    sp = res['species']
    # ---- cutoff correction per decoration (see header) ----
    host700 = E700.get(f'{system}_perfect_q0')
    res['cutoff_correction'] = {}
    if host700 is not None and REF700:
        muH7 = 0.5 * REF700['H2']; muO7 = REF700['H2O'] - REF700['H2']
        if system == 'si':
            mu_rem7 = {c: host700 / 64 for c in table}
        else:
            muSiC7 = host700 / 32
            muSi7 = E700['si_perfect_q0'] / 64 if 'si_perfect_q0' in E700 else None
            muC7 = Cdia700['E_per_atom'] if Cdia700 else None
            lim = res.get('_limit', 'Si-rich')
            if lim == 'Si-rich' and muSi7 is not None:
                mus = {'Si': muSi7, 'C': muSiC7 - muSi7}
            elif lim == 'C-rich' and muC7 is not None:
                mus = {'C': muC7, 'Si': muSiC7 - muC7}
            else:
                mus = None
            mu_rem7 = {c: (mus['C'] if c in ('VC', 'VCH') else mus['Si']) for c in table} if mus else {}
        for cfg, (mH, mO, charges) in table.items():
            k0 = f'{cfg}_q0'; t7 = f'{system}_{cfg}_q0'
            if k0 in sp and t7 in E700 and cfg in mu_rem7:
                Ef7 = E700[t7] - host700 + mu_rem7[cfg] - mH * muH7 - mO * muO7
                shift = Ef7 - sp[k0]['Ef_ref']
                res['cutoff_correction'][cfg] = {'Ef_340': sp[k0]['Ef_ref'], 'Ef_700': Ef7, 'shift': shift}
                for q in charges:
                    k = f'{cfg}_q{q}'
                    if k in sp:
                        sp[k]['Ef_340'] = sp[k]['Ef_ref']; sp[k]['Ef_ref'] = sp[k]['Ef_ref'] + shift
                        sp[k]['cutoff_shift'] = shift
    # transition levels for each decoration (eV above VBM)
    res['transition_levels'] = {}
    for cfg, (mH, mO, charges) in table.items():
        qs = sorted([q for q in charges if f'{cfg}_q{q}' in sp], reverse=True)
        lv = {}
        for a, b in zip(qs[:-1], qs[1:]):
            lv[f'({a:+d}/{b:+d})'] = (sp[f'{cfg}_q{b}']['Ef_ref'] - sp[f'{cfg}_q{a}']['Ef_ref']) / (a - b)
        if len(qs) >= 3 and 1 in qs and -1 in qs:           # negative-U check (+/-)
            lv['(+1/-1)'] = (sp[f'{cfg}_q-1']['Ef_ref'] - sp[f'{cfg}_q1']['Ef_ref']) / 2
            lv['U_eff'] = lv.get('(+0/-1)', np.nan) - lv.get('(+1/+0)', np.nan)
        if lv:
            res['transition_levels'][cfg] = lv
    # neutral sequential H binding
    order = ['V', 'VH', 'VH2', 'VH3', 'VH4']
    res['binding_neutral'] = {}
    for m in range(1, 5):
        a, b = f'{order[m-1]}_q0', f'{order[m]}_q0'
        if a in sp and b in sp:
            res['binding_neutral'][order[m]] = sp[a]['Ef_ref'] - sp[b]['Ef_ref']
    return res


# ---- silicon: host removed = mu_Si(bulk)
Hsi = host('si')
si = analyse('si', SI_SPECIES, {c: Hsi['mu_atom'] for c in SI_SPECIES})
out['si'] = si

# ---- SiC: Si-rich limit (mu_Si = bulk Si) and C-rich limit (mu_C = diamond)
Hsic = host('sic')
mu_SiC = 2 * Hsic['mu_atom']                                  # per formula unit
mu_Si_bulk = Hsi['mu_atom']
mu_C_dia = Cdia['E_per_atom'] if Cdia else None
limits = {}
if mu_C_dia is not None:
    dHf = mu_SiC - mu_Si_bulk - mu_C_dia                      # formation enthalpy of SiC (<0)
    out['refs']['dHf_SiC'] = dHf
    limits['Si-rich'] = {'mu_Si': mu_Si_bulk, 'mu_C': mu_SiC - mu_Si_bulk}
    limits['C-rich'] = {'mu_C': mu_C_dia, 'mu_Si': mu_SiC - mu_C_dia}
else:
    limits['Si-rich'] = {'mu_Si': mu_Si_bulk, 'mu_C': mu_SiC - mu_Si_bulk}
out['sic'] = {}
for lim, mus in limits.items():
    mu_removed = {c: (mus['mu_C'] if c in ('VC', 'VCH') else mus['mu_Si']) for c in SIC_SPECIES}
    out['sic'][lim] = analyse('sic', SIC_SPECIES, mu_removed, limit=lim)
out['sic']['limits'] = limits
# binding energies (limit independent)
s0 = out['sic']['Si-rich']['species']
bind_sic = {}
if 'VC_q0' in s0 and 'VCH_q0' in s0:
    bind_sic['VC'] = s0['VC_q0']['Ef_ref'] - s0['VCH_q0']['Ef_ref']
if 'VSi_q0' in s0 and 'VSiH_q0' in s0:
    bind_sic['VSi'] = s0['VSi_q0']['Ef_ref'] - s0['VSiH_q0']['Ef_ref']
out['sic']['binding_neutral'] = bind_sic

# ---- LVMs.  For VH4 (Td) the four Si-H stretches split into an IR-active T2 triplet
# and a Raman-only A1 singlet (the highest mode); only IR-active modes are compared
# with the infrared data.
out['lvm'] = {c: load(f'vibres64_{c}.json') for c in ['VH', 'VH2', 'VH3', 'VH4']}
for c, v in out['lvm'].items():
    if v:
        st = sorted(v['stretch_cm'])
        v['stretch_ir'] = st[:-1] if (c == 'VH4' and len(st) == 4) else st
        v['stretch_raman_only'] = st[-1:] if (c == 'VH4' and len(st) == 4) else []
out['lvm_exp'] = EXP['Si_VHn_LVM_cm']

# ---- vibrational free energies (Oct 2026) ------------------------------------------
# Local modes only: from each partial Hessian (H/O atoms + four Si neighbours) the 12
# lowest modes are Si-neighbour-like and common to every complex; the rest are the H and
# O modes (3 per atom).  Gas references: computed ZPE (mol_vib.py) plus the standard
# 298 K enthalpy and entropy (NIST-JANAF, as tabulated for the CHE by Norskov et al.).
# H2O(l) is H2O(g) at its 298 K vapour pressure (0.035 bar).  Charged states take the
# vibrational term of the neutral complex with the same decoration.
KB_EV, T_VIB = 8.617333e-5, 298.15
KJMOL = 1 / 96.485
def f_vib(freqs_cm, T=T_VIB):
    e = np.array(freqs_cm) / 8065.54
    return float(np.sum(0.5 * e + KB_EV * T * np.log(1 - np.exp(-e / (KB_EV * T)))))
VIB = {}
for c in ['VH', 'VH2', 'VH3', 'VH4', 'VO', 'VOH']:
    v = load(f'vibres64_{c}.json')
    if v:
        f = sorted(v['freqs_cm']); loc = f[12:]; sil = f[:12]
        VIB[c] = {'local_cm': loc, 'F_loc': f_vib(loc), 'ZPE_loc': 0.5 * sum(loc) / 8065.54,
                  'ZPE_Si12': 0.5 * sum(x for x in sil if x > 0) / 8065.54}
MV = {m: load(f'molvib_{m}.json') for m in ('H2', 'H2O')}
GAS = {'H2': {'H298': 8.468 * KJMOL, 'S298': 130.68e-3 * KJMOL, 'p_bar': 1.0},
       'H2O': {'H298': 9.904 * KJMOL, 'S298': 188.84e-3 * KJMOL, 'p_bar': 0.035}}
g = {}
for m in ('H2', 'H2O'):
    if MV[m]:
        d = GAS[m]
        g[m] = MV[m]['zpe_eV'] + d['H298'] - T_VIB * d['S298'] + KB_EV * T_VIB * np.log(d['p_bar'])
        g[m + '_relax'] = MV[m]['dE_relax']          # static energy change on relaxing the reference
if VIB and 'H2' in g and 'H2O' in g:
    dmuH_g = 0.5 * (g['H2'] + g['H2_relax'])                                  # shift of mu_H
    dmuO_g = (g['H2O'] + g['H2O_relax']) - (g['H2'] + g['H2_relax'])          # shift of mu_O = mu_H2O - 2 mu_H
    for key, d in si['species'].items():
        Floc = VIB.get(d['cfg'], {}).get('F_loc', 0.0) if d['cfg'] != 'V' else 0.0
        d['F_vib_loc'] = Floc
        d['Ef_G'] = d['Ef_ref'] + Floc - d['mH'] * dmuH_g - d['mO'] * dmuO_g
    out['vib'] = {'local': VIB, 'g_gas': g, 'dmuH_shift_ref': dmuH_g, 'dmuO_shift_ref': dmuO_g, 'T': T_VIB}
json.dump(out, open('assembled64.json', 'w'), indent=2)
print('wrote assembled64.json')

# ---------------------------------------------------- engine input (Si)
with open('species_Si_64.csv', 'w') as f:
    f.write('name,q,mH,mO,Ef_ref\n')
    for key, d in si['species'].items():
        f.write(f"{d['cfg']},{d['q']},{d['mH']},{d['mO']},{d['Ef_ref']:.4f}\n")
print('wrote species_Si_64.csv')


# ---------------------------------------------------- LaTeX macros
def macro(name, val, fmt='{:.2f}'):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        s = '--'
    else:
        s = fmt.format(val)
    return f'\\newcommand{{\\{name}}}{{{s}}}\n'

M = []
M.append(macro('aSi', BSI['a0'], '{:.3f}')); M.append(macro('aSiC', BSIC['a0'], '{:.3f}'))
M.append(macro('BSi', BSI['B_GPa'], '{:.0f}')); M.append(macro('BSiC', BSIC['B_GPa'], '{:.0f}'))
M.append(macro('EgSi', Hsi.get('gap'), '{:.2f}')); M.append(macro('EgSiC', Hsic.get('gap'), '{:.2f}'))
M.append(macro('EgSiEOS', BSI['gap'], '{:.2f}')); M.append(macro('EgSiCEOS', BSIC['gap'], '{:.2f}'))
M.append(macro('EHtwo', E_H2, '{:.3f}')); M.append(macro('EHtwoO', E_H2O, '{:.3f}'))
M.append(macro('muHref', mu_H, '{:.3f}')); M.append(macro('muOref', mu_O, '{:.3f}')); M.append(macro('muOHref', mu_OH, '{:.3f}'))
M.append(macro('muSi', Hsi['mu_atom'], '{:.3f}')); M.append(macro('muSiCfu', mu_SiC, '{:.3f}'))
M.append(macro('muCdia', mu_C_dia, '{:.3f}')); M.append(macro('aCdia', out['refs'].get('a0_C_diamond'), '{:.3f}'))
M.append(macro('dHfSiC', out['refs'].get('dHf_SiC'), '{:+.2f}'))
M.append(macro('LSi', Hsi['L'], '{:.2f}')); M.append(macro('LSiC', Hsic['L'], '{:.2f}'))
M.append(macro('EMPone', makov_payne(1, 'si', Hsi['L']), '{:.2f}')); M.append(macro('EMPtwo', makov_payne(2, 'si', Hsi['L']), '{:.2f}'))
M.append(macro('EMPoneSiC', makov_payne(1, 'sic', Hsic['L']), '{:.2f}'))
tex_name = {'V': 'V', 'VH': 'VH', 'VH2': 'VHtwo', 'VH3': 'VHthree', 'VH4': 'VHfour', 'VO': 'VO', 'VOH': 'VOH',
            'VC': 'VC', 'VCH': 'VCH', 'VSi': 'VSi', 'VSiH': 'VSiH'}
qname = {1: 'p', 0: 'z', -1: 'm', -2: 'mm'}
for key, d in si['species'].items():
    n = tex_name[d['cfg']] + qname[d['q']]
    M.append(macro('Ef' + n, d['Ef_ref'], '{:.2f}'))
    M.append(macro('Etot' + n, d['E'], '{:.3f}'))
    M.append(macro('Mag' + n, abs(d['magmom']) if d['magmom'] is not None else None, '{:.2f}'))
    M.append(macro('dV' + n, d['dV_align'], '{:+.3f}'))
    if d.get('E_JT') is not None:
        M.append(macro('EJT' + n, d['E_JT'], '{:.2f}'))
    if d['nn_pair_distances']:
        M.append(macro('nnmin' + n, min(d['nn_pair_distances']), '{:.2f}'))
        M.append(macro('nnmax' + n, max(d['nn_pair_distances']), '{:.2f}'))
    if d['XH_bonds']:
        M.append(macro('XH' + n, np.mean(d['XH_bonds']), '{:.3f}'))
        M.append(macro('XHmin' + n, min(d['XH_bonds']), '{:.3f}'))
        M.append(macro('XHmax' + n, max(d['XH_bonds']), '{:.3f}'))
    if d.get('O_to_nn'):
        so = sorted(d['O_to_nn'])
        M.append(macro('SiO' + n, so[0], '{:.2f}')); M.append(macro('SiOb' + n, so[1], '{:.2f}'))
    if d.get('OH'):
        M.append(macro('OHb' + n, d['OH'], '{:.3f}'))
for lim in ('Si-rich', 'C-rich'):
    if lim not in out['sic']:
        continue
    suf = 'Sirich' if lim == 'Si-rich' else 'Crich'
    for key, d in out['sic'][lim]['species'].items():
        n = tex_name[d['cfg']] + qname[d['q']]
        M.append(macro('Ef' + n + suf, d['Ef_ref'], '{:.2f}'))
        if lim == 'Si-rich':
            M.append(macro('Etot' + n, d['E'], '{:.3f}'))
            M.append(macro('Mag' + n, abs(d['magmom']) if d['magmom'] is not None else None, '{:.2f}'))
            if d.get('E_JT') is not None:
                M.append(macro('EJT' + n, d['E_JT'], '{:.2f}'))
            if d.get('E_other_spin') is not None:
                M.append(macro('dEspin' + n, d['E_other_spin'] - d['E'], '{:.2f}'))
            if d['nn_pair_distances']:
                M.append(macro('nnmin' + n, min(d['nn_pair_distances']), '{:.2f}'))
                M.append(macro('nnmax' + n, max(d['nn_pair_distances']), '{:.2f}'))
            if d['XH_bonds']:
                M.append(macro('XH' + n, np.mean(d['XH_bonds']), '{:.3f}'))
for m, cfg in enumerate(['VH', 'VH2', 'VH3', 'VH4'], 1):
    M.append(macro('Eb' + tex_name[cfg], si['binding_neutral'].get(cfg), '{:+.2f}'))
M.append(macro('EbVC', bind_sic.get('VC'), '{:+.2f}')); M.append(macro('EbVSi', bind_sic.get('VSi'), '{:+.2f}'))
for cfg, lv in si['transition_levels'].items():
    for k, v in lv.items():
        kk = k.replace('(+1/+0)', 'PZ').replace('(+0/-1)', 'ZM').replace('(-1/-2)', 'MMM').replace('(+1/-1)', 'PM').replace('U_eff', 'Ueff')
        M.append(macro('eps' + tex_name[cfg] + kk, v, '{:+.2f}'))
for lim in ('Si-rich',):
    for cfg, lv in out['sic'][lim]['transition_levels'].items():
        for k, v in lv.items():
            kk = k.replace('(+1/+0)', 'PZ').replace('(+0/-1)', 'ZM').replace('(+1/-1)', 'PM').replace('U_eff', 'Ueff')
            M.append(macro('eps' + tex_name[cfg] + kk, v, '{:+.2f}'))
for cfg in ['VH', 'VH2', 'VH3', 'VH4']:
    v = out['lvm'][cfg]
    if v:
        st = v['stretch_ir']
        # distinct IR-active values (degenerate T2/E modes listed once)
        uniq = []
        for x in st:
            if not uniq or abs(x - uniq[-1]) > 2.0:
                uniq.append(x)
        txt = ', '.join(f'{x:.0f}' for x in uniq)
        if v['stretch_raman_only']:
            txt += f" ({v['stretch_raman_only'][0]:.0f} A$_1$, IR-inactive)"
        M.append(f"\\newcommand{{\\LVM{tex_name[cfg]}}}{{{txt}}}\n")
        ex = out['lvm_exp'][cfg]
        M.append(f"\\newcommand{{\\LVMexp{tex_name[cfg]}}}{{{', '.join(str(x) for x in ex)}}}\n")
        M.append(macro('LVMmean' + tex_name[cfg], np.mean(st) if st else None, '{:.0f}'))
        M.append(macro('LVMerr' + tex_name[cfg], 100 * (np.mean(st) - np.mean(ex)) / np.mean(ex) if st else None, '{:+.1f}'))
# overall LVM statistics (IR-active modes, complex means)
allc, alle = [], []
for cfg in ['VH', 'VH2', 'VH3', 'VH4']:
    v = out['lvm'][cfg]
    if v and v['stretch_ir']:
        allc.append(np.mean(v['stretch_ir'])); alle.append(np.mean(out['lvm_exp'][cfg]))
if allc:
    sc = np.sum(np.array(alle) * np.array(allc)) / np.sum(np.array(allc) ** 2)
    M.append(macro('LVMscale', sc, '{:.3f}'))
    M.append(macro('LVMmad', np.mean(np.abs(np.array(allc) - np.array(alle))), '{:.0f}'))
    M.append(macro('LVMmadpct', 100 * np.mean(np.abs(np.array(allc) - np.array(alle)) / np.array(alle)), '{:.1f}'))
    M.append(macro('LVMmadscaled', np.mean(np.abs(sc * np.array(allc) - np.array(alle))), '{:.0f}'))
    M.append(macro('LVMmaxpct', 100 * np.max(np.abs(np.array(allc) - np.array(alle)) / np.array(alle)), '{:.1f}'))
    M.append(macro('LVMminpct', 100 * np.min(np.abs(np.array(allc) - np.array(alle)) / np.array(alle)), '{:.1f}'))
# cutoff-correction macros and SI table
T = [r"""\begin{table}[h]
\caption{Plane-wave cutoff check. Neutral formation energies at $\Delta\mu_{\mathrm{H}}=0$ from the production 340~eV calculations and from 700~eV single points on the same geometries with molecular references at the same cutoff. The 700~eV values are used in the main text; charged states of the same decoration receive the same shift.}
\label{tab:cutoff}
\begin{tabular}{llrrr}
\toprule
Host & Complex & $E^{f}$(340~eV) & $E^{f}$(700~eV) & shift\\
\midrule
"""]
ncorr = 0
labc = {'V': '$V$', 'VH': '$V$H', 'VH2': '$V$H$_2$', 'VH3': '$V$H$_3$', 'VH4': '$V$H$_4$', 'VO': '$V$--O', 'VOH': '$V$--OH',
        'VC': '$V_{\\mathrm{C}}$', 'VCH': '$V_{\\mathrm{C}}$H', 'VSi': '$V_{\\mathrm{Si}}$', 'VSiH': '$V_{\\mathrm{Si}}$H'}
for hostname, resd in (('Si', si), ('3C-SiC (Si-rich)', out['sic']['Si-rich'])):
    for cfg, c in resd.get('cutoff_correction', {}).items():
        T.append(f"{hostname} & {labc[cfg]} & {c['Ef_340']:.3f} & {c['Ef_700']:.3f} & {c['shift']:+.3f}\\\\\n"); ncorr += 1
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open('paper/si_table_cutoff.tex', 'w').write(''.join(T))
sh = si.get('cutoff_correction', {})
M.append(macro('cutVO', sh.get('VO', {}).get('shift'), '{:+.2f}')); M.append(macro('cutVOH', sh.get('VOH', {}).get('shift'), '{:+.2f}'))
M.append(macro('cutVHfour', sh.get('VH4', {}).get('shift'), '{:+.2f}')); M.append(macro('cutVH', sh.get('VH', {}).get('shift'), '{:+.2f}'))
M.append(macro('cutmaxH', max([abs(v['shift']) for k, v in sh.items() if k in ('V', 'VH', 'VH2', 'VH3', 'VH4')] or [float('nan')]), '{:.2f}'))
M.append(macro('ncutoff', ncorr, '{:d}'))
M.append(macro('EHtwoSeven', REF700.get('H2'), '{:.3f}')); M.append(macro('EHtwoOSeven', REF700.get('H2O'), '{:.3f}'))

# stability-map boundaries (Si) and derived quantities
sp = si['species']; Ef = lambda k: sp[k]['Ef_ref']
kT10 = 8.617333e-5 * 300 * np.log(10.0)
if all(k in sp for k in ('V_q0', 'VH4_q0', 'VO_q0', 'VOH_q0', 'VH2_q0', 'VH_q0', 'VH3_q0')):
    b_V_VH4 = -(Ef('V_q0') - Ef('VH4_q0')) / 4
    b_VOH_VH4 = (Ef('VH4_q0') - Ef('VOH_q0')) / 5
    b_VO_VOH = Ef('VOH_q0') - Ef('VO_q0')
    M.append(macro('dmuVtoVHfour', b_V_VH4, '{:+.2f}'))
    M.append(macro('dmuVOHtoVHfour', b_VOH_VH4, '{:+.2f}'))
    M.append(macro('dmuVOtoVOH', b_VO_VOH, '{:+.2f}'))
    M.append(macro('pHVHfour', -b_VOH_VH4 / kT10, '{:.0f}'))            # pH at U = 0 where VH4 takes over from VOH
    M.append(macro('UVHfour', -b_VOH_VH4 - kT10 * 7, '{:+.2f}'))         # U (V vs SHE) at pH 7 for the same boundary
    M.append(macro('pHVOH', -b_VO_VOH / kT10, '{:.0f}'))                 # pH at U = 0 where VOH gives way to VO
    M.append(macro('Edisp', 2 * Ef('VH2_q0') - Ef('V_q0') - Ef('VH4_q0'), '{:.2f}'))
    M.append(macro('EdispOdd', Ef('VH_q0') + Ef('VH3_q0') - Ef('V_q0') - Ef('VH4_q0'), '{:.2f}'))
    M.append(macro('dmuHtwoRef', -0.5 * (Ef('V_q0') - Ef('VO_q0')), '{:+.2f}'))
    M.append(macro('EbO', Ef('V_q0') - Ef('VO_q0'), '{:.2f}'))                 # O binding at dmu = 0 vs water
    M.append(macro('EbOH', Ef('V_q0') - Ef('VOH_q0'), '{:.2f}'))
    M.append(macro('nnmeanVp', np.mean(sp['V_q1']['nn_pair_distances']), '{:.2f}'))
    M.append(macro('nnmeanVmm', np.mean(sp['V_q-2']['nn_pair_distances']), '{:.2f}'))
    M.append(macro('nnideal', Hsi['a'] / np.sqrt(2), '{:.2f}'))
# vibrationally corrected boundaries (Ef_G); dmu_H is then measured from 1/2 G(H2, 298 K, 1 bar)
if 'vib' in out and all(k in sp for k in ('V_q0', 'VH4_q0', 'VO_q0', 'VOH_q0')):
    G = lambda k: sp[k]['Ef_G']
    gb_V_VH4 = -(G('V_q0') - G('VH4_q0')) / 4
    gb_VOH_VH4 = (G('VH4_q0') - G('VOH_q0')) / 5
    gb_VO_VOH = G('VOH_q0') - G('VO_q0')
    M.append(macro('dmuVtoVHfourG', gb_V_VH4, '{:+.2f}')); M.append(macro('pHVtoVHfourG', -gb_V_VH4 / kT10, '{:.0f}'))
    M.append(macro('dmuVOtoVOHG', gb_VO_VOH, '{:+.2f}')); M.append(macro('pHVOHG', -gb_VO_VOH / kT10, '{:.0f}'))
    M.append(macro('dmuVOHtoVHfourG', gb_VOH_VH4, '{:+.2f}'))
    M.append(macro('UVHfourpHzeroG', -gb_VOH_VH4, '{:+.2f}'))                 # U at pH 0 for VOH -> VH4
    M.append(macro('UVHfourG', -gb_VOH_VH4 - kT10 * 7, '{:+.2f}'))            # U at pH 7
    M.append(macro('UVOHpHsevenG', -gb_VO_VOH - kT10 * 7, '{:+.2f}'))         # U at pH 7 for VO -> VOH
    V = out['vib']
    M.append(macro('ZPEperH', np.mean([V['local'][c]['ZPE_loc'] / n for c, n in (('VH', 1), ('VH2', 2), ('VH3', 3), ('VH4', 4))]), '{:.2f}'))
    M.append(macro('FvibperH', np.mean([V['local'][c]['F_loc'] / n for c, n in (('VH', 1), ('VH2', 2), ('VH3', 3), ('VH4', 4))]), '{:.2f}'))
    M.append(macro('ZPEO', V['local']['VO']['ZPE_loc'], '{:.2f}'))
    M.append(macro('nuVOasym', max(V['local']['VO']['local_cm']), '{:.0f}'))
    M.append(macro('ZPEHtwo', MV['H2']['zpe_eV'], '{:.2f}')); M.append(macro('nuHtwo', MV['H2']['freqs_cm_internal'][0], '{:.0f}'))
    M.append(macro('ZPEHtwoO', MV['H2O']['zpe_eV'], '{:.2f}'))
    M.append(macro('gHtwo', V['g_gas']['H2'], '{:+.2f}')); M.append(macro('gHtwoO', V['g_gas']['H2O'], '{:+.2f}'))
    M.append(macro('shiftperH', np.mean([V['local'][c]['F_loc'] / n for c, n in (('VH', 1), ('VH2', 2), ('VH3', 3), ('VH4', 4))]) - V['dmuH_shift_ref'], '{:.2f}'))
    zs = [V['local'][c]['ZPE_Si12'] for c in V['local']]
    M.append(macro('ZPESispread', max(zs) - min(zs), '{:.2f}'))
    M.append(macro('EdispG', 2 * G('VH2_q0') - G('V_q0') - G('VH4_q0'), '{:.2f}'))
vc = out['sic']['Si-rich']['species']
if 'VCH_q0' in vc and 'VC_q0' in vc:
    M.append(macro('dmuVCH', vc['VCH_q0']['Ef_ref'] - vc['VC_q0']['Ef_ref'], '{:+.2f}'))
if 'VSiH_q0' in vc and 'VSi_q0' in vc:
    M.append(macro('dmuVSiH', vc['VSiH_q0']['Ef_ref'] - vc['VSi_q0']['Ef_ref'], '{:+.2f}'))
M.append(macro('nnidealSiC', Hsic['a'] / np.sqrt(2), '{:.2f}'))
M.append(macro('gapSiCG', Hsic.get('gap'), '{:.2f}'))
os.makedirs('paper', exist_ok=True)
open('paper/numbers.tex', 'w').write(''.join(M))
print(f'wrote paper/numbers.tex ({len(M)} macros)')

# ---------------------------------------------------- console report
print('\n=== Si (64-atom) ===  Ef_ref [eV] (incl. cutoff shift)   dV [eV]   E_MP [eV]   M [muB]')
print('  cutoff shifts:', {k: round(v['shift'], 3) for k, v in si.get('cutoff_correction', {}).items()})
for key, d in si['species'].items():
    print(f"  {key:8s} {d['Ef_ref']:+7.3f}   {d['dV_align'] if d['dV_align'] is not None else float('nan'):+6.3f}   "
          f"{d['E_MP']:5.3f}   {d['magmom']:5.2f}  steps={d['relax_steps']}")
print('  transition levels:', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in si['transition_levels'].items()})
print('  H binding (neutral):', {k: round(v, 3) for k, v in si['binding_neutral'].items()})
print(f"  gap(perfectG) = {Hsi.get('gap')}")
for lim in ('Si-rich', 'C-rich'):
    if lim in out['sic']:
        print(f'\n=== 3C-SiC ({lim}) ===')
        for key, d in out['sic'][lim]['species'].items():
            print(f"  {key:8s} Ef={d['Ef_ref']:+7.3f}  M={d['magmom']:5.2f}")
print('  SiC H binding:', {k: round(v, 3) for k, v in bind_sic.items()})
print('\n=== LVMs ===')
for cfg in ['VH', 'VH2', 'VH3', 'VH4']:
    v = out['lvm'][cfg]
    print(f"  {cfg:4s} calc={v['stretch_cm'] if v else None}  exp={out['lvm_exp'][cfg]}")


# ---------------------------------------------------- SI tables
def fmt(v, f='{:+.2f}'):
    return '--' if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)

lab = {'V': '$V$', 'VH': '$V$H', 'VH2': '$V$H$_2$', 'VH3': '$V$H$_3$', 'VH4': '$V$H$_4$', 'VO': '$V$--O', 'VOH': '$V$--OH',
       'VC': '$V_{\\mathrm{C}}$', 'VCH': '$V_{\\mathrm{C}}$H', 'VSi': '$V_{\\mathrm{Si}}$', 'VSiH': '$V_{\\mathrm{Si}}$H'}
def qlab(q):
    return {1: '$+1$', 0: '$0$', -1: '$-1$', -2: '$-2$'}[q]

T = []
T.append(r"""\begin{table}[h]
\caption{Silicon complexes in the 64-atom cell (PBE, spin-polarized): total energy $E$, Makov--Payne term $E_{\mathrm{MP}}$, potential-alignment shift $\Delta V$, reference formation energy $E^{f}_{\mathrm{ref}}$ (at $E_F=E_{\mathrm{VBM}}$, $\Delta\mu_{\mathrm{H}}=0$), total magnetic moment $M$, largest local moment $|m|_{\max}$, range of the six neighbor--neighbor distances $d_{\mathrm{nn}}$ (ideal: 3.87~\AA), and mean X--H bond length.}
\label{tab:si}
\footnotesize
\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrrcc}
\toprule
Complex & $q$ & $E$ (eV) & $E_{\mathrm{MP}}$ (eV) & $\Delta V$ (eV) & $E^{f}_{\mathrm{ref}}$ (eV) & $M$ ($\mu_B$) & $|m|_{\max}$ ($\mu_B$) & $d_{\mathrm{nn}}$ (\AA) & $d_{\mathrm{XH}}$ (\AA)\\
\midrule
""")
for key, d in si['species'].items():
    nn = f"{min(d['nn_pair_distances']):.2f}--{max(d['nn_pair_distances']):.2f}" if d['nn_pair_distances'] else '--'
    xh = f"{np.mean(d['XH_bonds']):.3f}" if d['XH_bonds'] else '--'
    T.append(f"{lab[d['cfg']]} & {qlab(d['q'])} & {d['E']:.3f} & {d['E_MP']:.3f} & {fmt(d['dV_align'], '{:+.3f}')} & "
             f"{d['Ef_ref']:+.3f} & {abs(d['magmom']):.2f} & {fmt(d['magmoms_local_max'], '{:.2f}')} & {nn} & {xh}\\\\\n")
T.append("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
open('paper/si_table_si.tex', 'w').write(''.join(T))

T = [r"""\begin{table}[h]
\caption{3C-SiC complexes in the 64-atom cell: total energy, formation energies at the Si-rich and C-rich limits, magnetic moment, neighbor--neighbor distance range (ideal: 3.10~\AA), and X--H bond length. $V_{\mathrm{C}}$ has four Si neighbors (Si--H bonds); $V_{\mathrm{Si}}$ has four C neighbors (C--H bonds).}
\label{tab:sic}
\footnotesize
\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrrcc}
\toprule
Complex & $q$ & $E$ (eV) & $E_{\mathrm{MP}}+q\Delta V$ (eV) & $E^{f}$ Si-rich (eV) & $E^{f}$ C-rich (eV) & $M$ ($\mu_B$) & $d_{\mathrm{nn}}$ (\AA) & $d_{\mathrm{XH}}$ (\AA)\\
\midrule
"""]
sr = out['sic']['Si-rich']['species']; cr = out['sic'].get('C-rich', {}).get('species', {})
for key, d in sr.items():
    nn = f"{min(d['nn_pair_distances']):.2f}--{max(d['nn_pair_distances']):.2f}" if d['nn_pair_distances'] else '--'
    xh = f"{np.mean(d['XH_bonds']):.3f}" if d['XH_bonds'] else '--'
    corr = d['E_MP'] + d['q'] * (d['dV_align'] or 0.0)
    T.append(f"{lab[d['cfg']]} & {qlab(d['q'])} & {d['E']:.3f} & {corr:+.3f} & {d['Ef_ref']:+.3f} & "
             f"{fmt(cr.get(key, {}).get('Ef_ref'), '{:+.3f}')} & {abs(d['magmom']):.2f} & {nn} & {xh}\\\\\n")
T.append("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
open('paper/si_table_sic.tex', 'w').write(''.join(T))

T = [r"""\begin{table}[h]
\caption{Thermodynamic charge-transition levels (eV above the PBE VBM) and effective correlation energy $U_{\mathrm{eff}}=\varepsilon(0/-)-\varepsilon(+/0)$ (negative for a negative-$U$ center). The (+/$-$) entry is the level at which $+1$ and $-1$ are degenerate, i.e. the observable level of a negative-$U$ center.}
\label{tab:levels}
\begin{tabular}{llrrrrr}
\toprule
Host & Defect & $\varepsilon(+/0)$ & $\varepsilon(0/-)$ & $\varepsilon(-/2-)$ & $\varepsilon(+/-)$ & $U_{\mathrm{eff}}$\\
\midrule
"""]
for hostname, tls in [('Si', si['transition_levels']), ('3C-SiC', out['sic']['Si-rich']['transition_levels'])]:
    for cfg, lv in tls.items():
        T.append(f"{hostname} & {lab[cfg]} & {fmt(lv.get('(+1/+0)'))} & {fmt(lv.get('(+0/-1)'))} & {fmt(lv.get('(-1/-2)'))} & "
                 f"{fmt(lv.get('(+1/-1)'))} & {fmt(lv.get('U_eff'))}\\\\\n")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open('paper/si_table_levels.tex', 'w').write(''.join(T))

T = [r"""\begin{table}[h]
\caption{All harmonic frequencies (cm$^{-1}$) of the H + four-Si-neighbor dynamical matrix for the $V$H$_m$ complexes, with the hydrogen participation $P_{\mathrm{H}}$ of each mode. Stretch modes ($>1500$~cm$^{-1}$, $P_{\mathrm{H}}>0.5$) are compared with experiment in Table~1 of the main text.}
\label{tab:lvmfull}
\small
\begin{tabular}{lp{12cm}}
\toprule
Complex & $\tilde\nu$ ($P_{\mathrm{H}}$)\\
\midrule
"""]
for cfg in ['VH', 'VH2', 'VH3', 'VH4']:
    v = out['lvm'][cfg]
    if not v: 
        T.append(f"{lab[cfg]} & -- \\\\\n"); continue
    items = [f"{f:.0f} ({p:.2f})" for f, p in zip(v['freqs_cm'], v['H_participation']) if f > 300]
    T.append(f"{lab[cfg]} & {', '.join(items)}\\\\\n")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open('paper/si_table_lvm.tex', 'w').write(''.join(T))

# reference calculations (symmetric saddle, alternative spin seeds, fixed-spin-moment checks)
T = [r"""\begin{table}[h]
\caption{Reference calculations used to test the ground states: symmetric (unseeded) solutions, alternative spin seeds, and fixed-spin-moment (FSM) checks with the total moment constrained to 1~$\mu_B$ and 0.01~eV smearing. $\Delta E$ is the energy relative to the default (unconstrained, low-spin-seeded) relaxation of the same cell (positive: higher); the lower of the two enters the main text. For $V_{\mathrm{C}}$H the two starting geometries converge to the same bridging structure.}
\label{tab:variants}
\small
\begin{tabular}{lllrl}
\toprule
Host & Complex & Reference calculation & $\Delta E$ (eV) & Outcome\\
\midrule
"""]
def var_row(host, cfg, q, suf, label, outcome_fn):
    r = load(f'res64_{host}_{cfg}_q{q}{suf}.json'); b = load(f'res64_{host}_{cfg}_q{q}.json')   # vs the unconstrained run
    if r and b:
        T.append(f"{'Si' if host=='si' else '3C-SiC'} & {lab[cfg]}$^{{{q:+d}}}$".replace('^{+0}', '^{0}') +
                 f" & {label} & {r['E'] - b['E']:+.3f} & {outcome_fn(r)}\\\\\n")
nnr = lambda r: f"{min(r['nn_pair_distances']):.2f}--{max(r['nn_pair_distances']):.2f}~\AA" if r.get('nn_pair_distances') else ''
var_row('si', 'V', 0, '_sym', 'symmetric start', lambda r: f"$M$={abs(r['magmom']):.2f}, $d_{{\mathrm{{nn}}}}$ {nnr(r)}")
var_row('si', 'V', 0, '_sharp', '$D_{2d}$ seed, 0.01 eV smearing', lambda r: f"$M$={abs(r['magmom']):.2f}, $d_{{\mathrm{{nn}}}}$ {nnr(r)}")
var_row('sic', 'VC', 0, '_sym', 'symmetric start', lambda r: f"$M$={abs(r['magmom']):.2f}, $d_{{\mathrm{{nn}}}}$ {nnr(r)}")
var_row('sic', 'VSi', 0, '_hs', 'ferromagnetic seed', lambda r: f"$M$={abs(r['magmom']):.2f}")
var_row('si', 'VH', 0, '_fsm', 'FSM doublet', lambda r: f"$M$={abs(r['magmom']):.2f}, $|m|_{{\max}}$={max(abs(x) for x in r['magmoms_local']):.2f}")
var_row('sic', 'VCH', 0, '_fsm', 'FSM doublet', lambda r: f"$M$={abs(r['magmom']):.2f}, $|m|_{{\max}}$={max(abs(x) for x in r['magmoms_local']):.2f}")
var_row('sic', 'VCH', 0, '_ideal', 'Si--H capped start', lambda r: f"same bridging structure, Si--H {min(r['XH_bonds']):.2f}~\AA")
for host, cfg in (('si', 'V'), ('sic', 'VC')):
    for q in (1, -1):
        var_row(host, cfg, q, '_fsm', 'FSM, 0.01 eV smearing', lambda r: f"$M$={abs(r['magmom']):.2f}, $d_{{\mathrm{{nn}}}}$ {nnr(r)}")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
open('paper/si_table_variants.tex', 'w').write(''.join(T))

# vibrational free energies (Oct 2026)
if 'vib' in out:
    V = out['vib']
    T = [r"""\begin{table}[h]
\caption{Local vibrational modes and vibrational free energies (harmonic, 64-atom cell, $\Gamma$ point). Of each partial Hessian (H/O atoms plus the four Si neighbors) the twelve lowest, Si-neighbor-dominated modes are common to all complexes and are excluded; the remaining modes are listed. $F_{\mathrm{vib}}$ is the harmonic free energy of these modes at 298~K (zero-point energy plus thermal term). Gas-phase references: zero-point energy from the calculated frequencies plus the standard 298~K enthalpy and entropy; $\mathrm{H_2O}$ at 0.035~bar.}
\label{tab:vib}
\small
\begin{tabular}{lp{7.2cm}rr}
\toprule
Species & local modes (cm$^{-1}$) & ZPE (eV) & $F_{\mathrm{vib}}$ (eV)\\
\midrule
"""]
    for c in ['VH', 'VH2', 'VH3', 'VH4', 'VO', 'VOH']:
        if c in V['local']:
            d = V['local'][c]
            T.append(f"{lab[c]} & {', '.join(f'{x:.0f}' for x in d['local_cm'])} & {d['ZPE_loc']:.3f} & {d['F_loc']:.3f}\\\\\n")
    for m, lm in (('H2', '$\\mathrm{H_2}$'), ('H2O', '$\\mathrm{H_2O}$')):
        mv = load(f'molvib_{m}.json')
        if mv:
            T.append(f"{lm} & {', '.join(f'{x:.0f}' for x in mv['freqs_cm_internal'])} & {mv['zpe_eV']:.3f} & $G$ corr.\\ {V['g_gas'][m]:+.3f}\\\\\n")
    T.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open('paper/si_table_vib.tex', 'w').write(''.join(T))

steps = [d['relax_steps'] for d in list(si['species'].values()) + list(sr.values()) if d.get('relax_steps')]
open('paper/si_tables.tex', 'w').write(macro('relaxstepsmin', min(steps) if steps else None, '{:d}') +
                                        macro('relaxstepsmax', max(steps) if steps else None, '{:d}'))
print('wrote SI tables')
