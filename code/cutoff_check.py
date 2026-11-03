"""Compare formation/binding energies of the H- and O-bearing Si complexes at 340 eV (production)
and 700 eV (single points on the same geometries, res700_*.json), each with molecular references
at its own cutoff (10 A box)."""
import json, glob
L = lambda p: json.load(open(p))
ref = {340: {'H2': L('res64_mol_H2_q0.json')['E'], 'H2O': L('res64_mol_H2O_q0.json')['E']},
       700: {'H2': L('resbox_H2_5_700.json')['E'], 'H2O': L('resbox_H2O_5_700.json')['E']}}
E = {340: {t: L(f'res64_{t}.json')['E'] for t in ['si_perfect_q0', 'si_V_q0', 'si_VH_q0', 'si_VH4_q0', 'si_VO_q0', 'si_VOH_q0']}}
E[700] = {t: L(f'res700_{t}.json')['E'] for t in E[340] if glob.glob(f'res700_{t}.json')}
def q(ec):
    e = E[ec]; muH = ref[ec]['H2'] / 2; muO = ref[ec]['H2O'] - ref[ec]['H2']; muSi = e['si_perfect_q0'] / 64
    Ef = lambda t, mH, mO: e[t] - e['si_perfect_q0'] + muSi - mH * muH - mO * muO
    out = {}
    if 'si_V_q0' in e: out['Ef(V)'] = Ef('si_V_q0', 0, 0)
    if 'si_VH_q0' in e: out['Ef(VH)'] = Ef('si_VH_q0', 1, 0)
    if 'si_VH4_q0' in e: out['Ef(VH4)'] = Ef('si_VH4_q0', 4, 0)
    if 'si_VO_q0' in e: out['Ef(VO)'] = Ef('si_VO_q0', 0, 1)
    if 'si_VOH_q0' in e: out['Ef(VOH)'] = Ef('si_VOH_q0', 1, 1)
    if 'Ef(V)' in out and 'Ef(VH)' in out: out['Ebind(VH)'] = out['Ef(V)'] - out['Ef(VH)']
    if 'Ef(V)' in out and 'Ef(VH4)' in out: out['Ebind/H (VH4)'] = (out['Ef(V)'] - out['Ef(VH4)']) / 4
    if 'Ef(VOH)' in out and 'Ef(VH4)' in out: out['dmu(VOH/VH4)'] = (out['Ef(VH4)'] - out['Ef(VOH)']) / 5
    if 'Ef(VO)' in out and 'Ef(VOH)' in out: out['dmu(VO/VOH)'] = out['Ef(VOH)'] - out['Ef(VO)']
    return out
a, b = q(340), q(700)
print(f"{'quantity':18s} {'340 eV':>9s} {'700 eV':>9s} {'diff':>8s}")
for k in a:
    if k in b: print(f"{k:18s} {a[k]:9.3f} {b[k]:9.3f} {b[k]-a[k]:+8.3f}")
    else: print(f"{k:18s} {a[k]:9.3f}   (700 eV pending)")
