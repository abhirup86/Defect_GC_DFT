"""
render_slab.py -- Fig. 1a: the relaxed 64-atom silicon cell with a vacancy (final64_si_V_q0.xyz)
seen in perspective, the four vacancy neighbours highlighted, with an illustrative layer of
water molecules on top (water positions are random and NOT computed; the panel is a schematic).
Writes render/slab.png (transparent background).
"""
import json, os, sys
NOWATER = 'nowater' in sys.argv          # cell icon for the method schematic (Fig. 1)
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CreateBondsModifier
from ovito.vis import Viewport, TachyonRenderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
at = read(os.path.join(ROOT, 'final64_si_V_q0.xyz')); rr = json.load(open(os.path.join(ROOT, 'res64_si_V_q0.json')))
L = at.cell[0, 0]; nn = rr['nn_indices']
# shift so the vacancy sits in the middle of the cell, then wrap
ref = at[nn[0]].position
cen = ref + np.mean([at.get_distance(nn[0], i, mic=True, vector=True) for i in nn], axis=0)
at.translate(np.array([L / 2, L / 2, L / 2 - 1.0]) - cen); at.wrap()
vac = np.array([L / 2, L / 2, L / 2 - 1.0])
CAM = np.array([-0.62, -1.0, -0.42]); CAM = CAM / np.linalg.norm(CAM)
P = at.get_positions()
# prune Si left with < 2 Si neighbours after wrapping (edge artefacts), twice
keep = np.ones(len(at), bool)
for _ in range(2):
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    coord = ((D < 2.6) & (D > 0.1) & keep[None, :]).sum(1)
    keep &= (coord >= 2) | np.isin(np.arange(len(at)), nn)
# cut-away: drop atoms between the camera and the vacancy (cylinder of radius 2.4 A)
for i, p in enumerate(P):
    v = p - vac; along = -np.dot(v, CAM)            # > 0: on the camera side
    perp = np.linalg.norm(v + along * CAM)
    if i not in nn and along > 0.8 and perp < 2.4:
        keep[i] = False
pos = [P[i] for i in range(len(at)) if keep[i]]; sym = ['Si'] * len(pos)
kind = ['nn' if i in nn else 'Si' for i in range(len(at)) if keep[i]]
# illustrative water layer
rng = np.random.default_rng(3); ztop = max(p[2] for p in pos)
Os = []
for _ in range(0 if NOWATER else 20000):    # random sequential placement, capped
    if len(Os) >= 24: break
    p = np.array([rng.uniform(0, L), rng.uniform(0, L), rng.uniform(ztop + 2.3, ztop + 8.5)])
    if all(np.linalg.norm(((p - q + L / 2) % L) - L / 2) > 2.8 for q in Os):
        Os.append(p)
for p in Os:
    w = molecule('H2O'); w.rotate(rng.uniform(0, 360), rng.normal(size=3)); w.translate(p - w.positions[0])
    for s, q in zip(w.get_chemical_symbols(), w.positions):
        pos.append(q); sym.append(s); kind.append('w' + s)
pos.append(vac); sym.append('He'); kind.append('X')
A = Atoms(symbols=sym, positions=np.array(pos), cell=[L, L, 40], pbc=False)
data = ase_to_ovito(A); data.cell_.vis.enabled = False
COL = {'Si': (0.72, 0.76, 0.82), 'nn': (0.36, 0.50, 0.66), 'wO': (0.84, 0.29, 0.30), 'wH': (0.97, 0.97, 0.97), 'X': (0.96, 0.64, 0.35)}
RAD = {'Si': 0.40, 'nn': 0.50, 'wO': 0.34, 'wH': 0.20, 'X': 0.70}
TR = {'Si': 0.0, 'nn': 0.0, 'wO': 0.25, 'wH': 0.25, 'X': 0.55}
data.particles_.create_property('Color', data=np.array([COL[k] for k in kind]))
data.particles_.create_property('Radius', data=np.array([RAD[k] for k in kind]))
data.particles_.create_property('Transparency', data=np.array([TR[k] for k in kind]))
pipe = Pipeline(source=StaticSource(data=data))
b = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
b.set_pairwise_cutoff('Si', 'Si', 2.6); b.set_pairwise_cutoff('O', 'H', 1.15)
b.vis.width = 0.18; b.vis.use_particle_colors = True
pipe.modifiers.append(b); pipe.add_to_scene()
vp = Viewport(type=Viewport.Type.Perspective, camera_dir=tuple(CAM), fov=np.radians(30))
vp.zoom_all(); vp.fov = np.radians(26)
vp.render_image(size=(1200, 1100), filename=os.path.join(ROOT, 'render', 'cell_icon.png' if NOWATER else 'slab.png'), background=(1, 1, 1), alpha=True,
                renderer=TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.85, shadows=False,
                                         antialiasing_samples=12, direct_light_intensity=0.95))
# mask render: same camera, everything white and opaque except the vacancy marker in pure green,
# so the composing script can find the vacancy's position in the image
pipe.remove_from_scene()
dm = ase_to_ovito(A); dm.cell_.vis.enabled = False
dm.particles_.create_property('Color', data=np.array([(0.0, 1.0, 0.0) if k == 'X' else (1.0, 1.0, 1.0) for k in kind]))
dm.particles_.create_property('Radius', data=np.array([RAD[k] for k in kind]))
pm = Pipeline(source=StaticSource(data=dm)); pm.add_to_scene()
vp.render_image(size=(1200, 1100), filename=os.path.join(ROOT, 'render', 'cell_icon_mask.png' if NOWATER else 'slab_mask.png'), background=(1, 1, 1), alpha=True,
                renderer=TachyonRenderer(ambient_occlusion=False, shadows=False, antialiasing_samples=1, direct_light=False))
pm.remove_from_scene()
print('rendered', 'cell_icon.png' if NOWATER else 'slab.png', '(+mask); water molecules:', len(Os), 'cell', round(L, 2))
