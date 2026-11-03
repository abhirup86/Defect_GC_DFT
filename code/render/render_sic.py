"""
render_sic.py -- ray-traced (OVITO/Tachyon) SiC panels for Fig. 2d,e.
  complex_sic_<cfg>.png : relaxed V_C, V_C-H, V_Si, V_Si-H close-ups (vacancy neighbours enlarged/darkened)
  sic_cell.png          : 3C-SiC 64-atom cell with the two sublattice sites marked (V_C orange, V_Si purple).
                          Illustrative: the two vacancies were computed in separate cells.
Same camera conventions as render_complexes.py.
"""
import json, os, sys
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CreateBondsModifier
from ovito.vis import Viewport, TachyonRenderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'render')
COL = {'Si': (0.58, 0.67, 0.79), 'C': (0.36, 0.36, 0.39), 'Si_nn': (0.30, 0.44, 0.62), 'C_nn': (0.13, 0.13, 0.15),
       'H': (0.98, 0.98, 0.98), 'VC': (0.96, 0.64, 0.35), 'VSi': (0.60, 0.45, 0.78)}
RAD = {'Si': 0.36, 'C': 0.30, 'Si_nn': 0.48, 'C_nn': 0.42, 'H': 0.30, 'VC': 0.62, 'VSi': 0.62}
TACH = dict(ambient_occlusion=True, ambient_occlusion_brightness=0.85, shadows=False, antialiasing_samples=12, direct_light_intensity=0.95)

def scene(pos, kind, bonds_pairs, out, size, view_dir, fov=24, transp=None):
    sym = ['He' if k in ('VC', 'VSi') else k.split('_')[0] for k in kind]
    A = Atoms(symbols=sym, positions=np.array(pos), cell=[40, 40, 40], pbc=False)
    data = ase_to_ovito(A); data.cell_.vis.enabled = False
    data.particles_.create_property('Color', data=np.array([COL[k] for k in kind]))
    data.particles_.create_property('Radius', data=np.array([RAD[k] for k in kind]))
    data.particles_.create_property('Transparency', data=np.array(transp if transp is not None else [0.0] * len(kind)))
    pipe = Pipeline(source=StaticSource(data=data))
    b = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    for x, y, c in bonds_pairs:
        b.set_pairwise_cutoff(x, y, c)
    b.vis.width = 0.15; b.vis.use_particle_colors = True
    pipe.modifiers.append(b); pipe.add_to_scene()
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=view_dir, fov=np.radians(30))
    vp.zoom_all(); vp.fov = np.radians(fov)
    vp.render_image(size=size, filename=out, background=(1, 1, 1), alpha=True, renderer=TachyonRenderer(**TACH))
    pipe.remove_from_scene(); print('rendered', out)

def complex_png(cfg, R_CUT=3.45):
    at = read(os.path.join(ROOT, f'final64_sic_{cfg}_q0.xyz')); rr = json.load(open(os.path.join(ROOT, f'res64_sic_{cfg}_q0.json')))
    nn = rr['nn_indices']; ref = at[nn[0]].position
    cen = ref + np.mean([at.get_distance(nn[0], i, mic=True, vector=True) for i in nn], axis=0)
    pos, kind = [], []
    for a in at:
        d = at.get_distance(nn[0], a.index, mic=True, vector=True) + ref - cen
        if np.linalg.norm(d) < R_CUT or a.symbol == 'H':
            pos.append(d); kind.append(a.symbol + ('_nn' if a.index in nn else '') if a.symbol != 'H' else 'H')
    scene(pos, kind, (('Si', 'C', 2.2), ('Si', 'H', 1.8), ('C', 'H', 1.25)),
          os.path.join(OUT, f'complex_sic_{cfg}.png'), (900, 900), (-0.28, -0.40, -1.0))

def cell_png():
    sc = bulk('SiC', 'zincblende', a=4.390074298678462, cubic=True) * (2, 2, 2)
    P = sc.get_positions(); S = sc.get_chemical_symbols(); L = sc.cell[0, 0]
    iC = [i for i, s in enumerate(S) if s == 'C']; iSi = [i for i, s in enumerate(S) if s == 'Si']
    cen = np.array([L / 2, L / 2, L / 2])
    vC = min(iC, key=lambda i: np.linalg.norm(P[i] - (cen + np.array([-1.6, 1.4, 0.4]))))
    vSi = min(iSi, key=lambda i: np.linalg.norm(P[i] - (cen + np.array([1.8, -1.6, -0.6]))))
    nnC = [i for i in range(len(S)) if i != vC and np.linalg.norm(P[i] - P[vC]) < 2.0]
    nnSi = [i for i in range(len(S)) if i != vSi and np.linalg.norm(P[i] - P[vSi]) < 2.0]
    pos, kind = [], []
    for i in range(len(S)):
        if i in (vC, vSi): continue
        pos.append(P[i]); kind.append(S[i] + ('_nn' if i in nnC + nnSi else ''))
    pos += [P[vC], P[vSi]]; kind += ['VC', 'VSi']
    tr = [0.0] * (len(kind) - 2) + [0.35, 0.35]
    scene(pos, kind, (('Si', 'C', 2.2),), os.path.join(OUT, 'sic_cell.png'), (1100, 1000), (-0.62, -1.0, -0.42), fov=22, transp=tr)
    # mask: same camera, everything white except V_C (green) and V_Si (blue), to place arrows in Fig. 2d
    saved = dict(COL); COL.update({k: (1.0, 1.0, 1.0) for k in COL}); COL['VC'] = (0.0, 1.0, 0.0); COL['VSi'] = (0.0, 0.0, 1.0)
    scene(pos, kind, (('Si', 'C', 2.2),), os.path.join(OUT, 'sic_cell_mask.png'), (1100, 1000), (-0.62, -1.0, -0.42), fov=22,
          transp=[0.0] * len(kind))
    COL.clear(); COL.update(saved)
    print('V_C - V_Si separation (A):', round(np.linalg.norm(P[vC] - P[vSi]), 2))

if __name__ == '__main__':
    for c in ('VC', 'VCH', 'VSi', 'VSiH'):
        complex_png(c)
    cell_png()
