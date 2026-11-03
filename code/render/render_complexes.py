"""
render_complexes.py -- ray-traced (OVITO/Tachyon) close-ups of the relaxed vacancy complexes
for Fig. 1b. Reads final64_si_<cfg>_q0.xyz and res64_si_<cfg>_q0.json from the project root,
keeps the atoms within R_CUT of the vacancy centre (unwrapped around it), marks the vacancy
with a translucent sphere and renders with a fixed camera. Run with the OVITO venv:
    <venv>/bin/python render/render_complexes.py
Writes render/complex_<cfg>.png (transparent background).
"""
import json, os, sys
import numpy as np
from ase.io import read
from ovito.io.ase import ase_to_ovito
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CreateBondsModifier
from ovito.vis import Viewport, TachyonRenderer
from ovito.data import DataCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'render')
R_CUT = 4.0
COL = {'Si': (0.76, 0.80, 0.86), 'Sinn': (0.36, 0.50, 0.66), 'H': (0.90, 0.90, 0.92), 'O': (0.84, 0.29, 0.30), 'X': (0.96, 0.64, 0.35)}
RAD = {'Si': 0.42, 'H': 0.33, 'O': 0.40, 'X': 0.40}

def cluster(cfg):
    at = read(os.path.join(ROOT, f'final64_si_{cfg}_q0.xyz')); rr = json.load(open(os.path.join(ROOT, f'res64_si_{cfg}_q0.json')))
    nn = rr['nn_indices']; ref = at[nn[0]].position
    vec = lambda i: at.get_distance(nn[0], i, mic=True, vector=True)
    cen = ref + np.mean([vec(i) for i in nn], axis=0)
    keep, pos, sym, hl = [], [], [], []
    for a in at:
        d = at.get_distance(nn[0], a.index, mic=True, vector=True) + ref - cen
        if np.linalg.norm(d) < R_CUT or a.symbol in ('H', 'O'):
            pos.append(d); sym.append(a.symbol); hl.append(a.index in nn)
    return np.array(pos), sym, hl

def render(cfg, size=(900, 900), view_dir=(-0.28, -0.40, -1.0)):
    from ase import Atoms
    pos, sym, hl = cluster(cfg)
    # no vacancy marker in the close-ups: it would hide the inward-pointing H; the four dark
    # neighbours mark the vacancy (the marker is shown in the crystal view, Fig. 1a)
    atoms = Atoms(symbols=[s if s != 'X' else 'He' for s in sym], positions=pos, cell=[30, 30, 30], pbc=False)
    data = ase_to_ovito(atoms)
    data.cell_.vis.enabled = False
    ptypes = data.particles_.particle_types_
    for t in ptypes.types_:
        name = {'He': 'X'}.get(t.name, t.name)
        t.radius = RAD[name]; t.color = COL[name]
    colors = np.array([COL['Sinn'] if h else COL[s] for s, h in zip(sym, hl)])
    data.particles_.create_property('Color', data=colors)
    # second shell small and light, first-shell neighbours large and dark, H/O at full size
    rad = np.array([0.46 if h else (RAD[s] if s in ('H', 'O') else 0.27) for s, h in zip(sym, hl)])
    data.particles_.create_property('Radius', data=rad)
    tr = np.zeros(len(sym))
    data.particles_.create_property('Transparency', data=tr)
    pipe = Pipeline(source=StaticSource(data=data))
    bonds = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    for a, b, c in (('Si', 'Si', 2.6), ('Si', 'H', 1.75), ('Si', 'O', 1.95), ('O', 'H', 1.15)):
        bonds.set_pairwise_cutoff(a, b, c)
    bonds.vis.width = 0.16; bonds.vis.use_particle_colors = True
    pipe.modifiers.append(bonds)
    pipe.add_to_scene()
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=view_dir, fov=np.radians(28))
    vp.zoom_all(); vp.fov = np.radians(24)
    out = os.path.join(OUT, f'complex_{cfg}.png')
    vp.render_image(size=size, filename=out, background=(1, 1, 1), alpha=True,
                    renderer=TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.85, shadows=False,
                                             antialiasing_samples=16, direct_light_intensity=0.95))
    pipe.remove_from_scene()
    print('rendered', out)

if __name__ == '__main__':
    for cfg in (sys.argv[1:] or ['V', 'VH', 'VH2', 'VH3', 'VH4', 'VO', 'VOH']):
        render(cfg)
