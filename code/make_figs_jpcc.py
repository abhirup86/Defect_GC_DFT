"""
make_figs_jpcc.py -- manuscript figures (ACS single column 3.33 in / double 7.0 in)
from assembled64.json.  Every panel is drawn from the production 64-atom data;
panels whose data are missing are skipped with a message.

Writes paper/figs/fig1_concept, fig2_validation, fig3_hydrogen, fig4_model,
figS1_sic_map, toc_graphic  (.pdf + .png).
"""
import json, os, sys, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hydration_engine import Species, stability_map, dmu_H_from_pH_U

OUT = 'paper/figs'
os.makedirs(OUT, exist_ok=True)
A = json.load(open('assembled64.json'))
EXP = json.load(open('exp_data.json'))

# Aptos (Microsoft Office font) if available, else matplotlib's default DejaVu Sans
from matplotlib import font_manager
APTOS_DIR = os.environ.get('APTOS_DIR', '/Applications/Microsoft Word.app/Contents/Resources/DFonts')   # optional; falls back to the default font
if os.path.isdir(APTOS_DIR):
    for f in os.listdir(APTOS_DIR):
        if f.startswith('Aptos') and f.endswith('.ttf') and 'Narrow' not in f:
            font_manager.fontManager.addfont(os.path.join(APTOS_DIR, f))
FONT = 'Aptos' if 'Aptos' in {f.name for f in font_manager.fontManager.ttflist} else 'DejaVu Sans'
FS = 1.3   # scale for the hard-coded annotation sizes below


def fs(x):
    return round(x * FS, 1)


mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic',
    'mathtext.bf': f'{FONT}:bold', 'mathtext.sf': FONT, 'mathtext.default': 'regular',
    'font.size': 10, 'axes.labelsize': 10.5, 'axes.titlesize': 10.5, 'legend.fontsize': 9,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3.5,
    'ytick.major.size': 3.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 600, 'lines.linewidth': 1.3,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
# Okabe-Ito, fixed order
BLU, ORA, GRN, PNK, SKY, YEL, VER, GRY = '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#D55E00', '#7f7f7f'
SPECIES_COLOR = {'V': BLU, 'VH': SKY, 'VH2': GRN, 'VH3': ORA, 'VH4': PNK, 'VO': VER, 'VOH': YEL,
                 'VC': VER, 'VCH': ORA, 'VSi': GRN, 'VSiH': SKY}
QSTYLE = {2: ':', 1: '--', 0: '-', -1: '-.', -2: (0, (1, 1))}


def panel(ax, l, dx=-0.25, dy=1.02):
    ax.text(dx, dy, l, transform=ax.transAxes, fontsize=fs(10), fontweight='bold', va='bottom', ha='left')


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight')
    plt.close(fig); print(f'{name} done')


def name_of(cfg):
    return {'V': 'V', 'VH': 'VH', 'VH2': 'VH$_2$', 'VH3': 'VH$_3$', 'VH4': 'VH$_4$', 'VO': 'VO', 'VOH': 'VOH',
            'VC': '$V_{\\mathrm{C}}$', 'VCH': '$V_{\\mathrm{C}}$H', 'VSi': '$V_{\\mathrm{Si}}$', 'VSiH': '$V_{\\mathrm{Si}}$H'}[cfg]


def label_of(cfg, q):
    sub = {'V': 'V', 'VH': 'VH', 'VH2': 'VH$_2$', 'VH3': 'VH$_3$', 'VH4': 'VH$_4$', 'VO': 'VO', 'VOH': 'VOH',
           'VC': '$V_{\\mathrm{C}}$', 'VCH': '$V_{\\mathrm{C}}$H', 'VSi': '$V_{\\mathrm{Si}}$', 'VSiH': '$V_{\\mathrm{Si}}$H'}[cfg]
    qs = {1: '$^{+}$', 0: '$^{0}$', -1: '$^{-}$', -2: '$^{2-}$', 2: '$^{2+}$'}[q]
    return sub + qs


# =====================================================================
def fig1_schematic():
    """(superseded by the rendered fig1 below) Oct 2026 redesign: (a) physical picture, (b) the computed complexes drawn from the
    relaxed 64-atom structures, (c) how minimizing Omega over the complexes gives the map
    (lower envelope of Omega vs dmu_H at mid-gap, vibrational free energies included)."""
    from ase.io import read
    from matplotlib.patches import Circle, Rectangle
    fig = plt.figure(figsize=(7.48, 2.9))
    # ---------------- (a) physical picture
    ax = fig.add_axes([0.0, 0.08, 0.33, 0.86]); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.add_patch(Rectangle((0, 0.3), 3.4, 6.2, fc='#dcebf8', ec='none'))
    ax.add_patch(Rectangle((3.4, 0.3), 6.6, 6.2, fc='#f1f1f1', ec='none'))
    ax.text(1.7, 5.95, 'water', ha='center', fontsize=fs(6.5), fontweight='bold', color='#1f4e79')
    ax.text(1.7, 4.9, 'H$_2$O $\\rightleftharpoons$ H$^+$ + OH$^-$', ha='center', fontsize=fs(5.6))
    ax.text(1.7, 4.15, 'pH, $U$  $\\rightarrow$  $\\mu_{\\mathrm{H}}$', ha='center', fontsize=fs(5.6))
    ax.text(1.7, 3.4, '$\\mu_{\\mathrm{O}}=\\mu_{\\mathrm{H_2O}}-2\\mu_{\\mathrm{H}}$', ha='center', fontsize=fs(5.6))
    ax.text(6.7, 5.95, 'Si crystal', ha='center', fontsize=fs(6.5), fontweight='bold', color='0.2')
    # lattice (projected diamond lattice as a square net) with one vacancy
    xs = np.arange(4.1, 9.8, 0.82); ys = np.arange(1.0, 5.6, 0.82)
    vac = (xs[3], ys[2])
    for x in xs:
        for y in ys:
            if abs(x - vac[0]) < 1e-6 and abs(y - vac[1]) < 1e-6: continue
            for dx, dy in ((0.82, 0), (0, 0.82)):
                x2, y2 = x + dx, y + dy
                if x2 > xs[-1] + 1e-6 or y2 > ys[-1] + 1e-6: continue
                if abs(x2 - vac[0]) < 1e-6 and abs(y2 - vac[1]) < 1e-6: continue
                ax.plot([x, x2], [y, y2], color='0.7', lw=0.8, zorder=1)
            ax.add_patch(Circle((x, y), 0.15, fc='0.55', ec='none', zorder=2))
    ax.add_patch(Circle(vac, 0.3, fc='none', ec='0.3', ls='--', lw=0.8, zorder=3))
    for dx, dy in ((0.82, 0), (-0.82, 0), (0, 0.82), (0, -0.82)):        # dangling bonds
        ax.plot([vac[0] + dx, vac[0] + 0.55 * dx], [vac[1] + dy, vac[1] + 0.55 * dy], color=VER, lw=1.6, zorder=3,
                solid_capstyle='round')
    ax.text(vac[0], vac[1] - 0.62, '$X^q$', ha='center', va='top', fontsize=fs(6.2), fontweight='bold', zorder=4,
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9))
    ax.add_patch(FancyArrowPatch((2.9, 2.75), (vac[0] - 0.35, vac[1] - 0.15), connectionstyle='arc3,rad=0.25',
                                 arrowstyle='<|-|>', mutation_scale=8, lw=1.1, color=GRN, zorder=5))
    ax.text(1.7, 1.5, 'H, O, OH\n$(m_{\\mathrm{H}}, m_{\\mathrm{O}})$', ha='center', fontsize=fs(5.4), color='#00694d')
    ax.add_patch(FancyArrowPatch((vac[0] + 1.9, 0.75), (vac[0] + 0.3, vac[1] - 0.3), connectionstyle='arc3,rad=-0.25',
                                 arrowstyle='<|-|>', mutation_scale=8, lw=1.1, color=BLU, zorder=5))
    ax.text(vac[0] + 2.0, 0.45, '$e^-$ ($E_F$): charge $q$', ha='center', fontsize=fs(5.4), color=BLU,
            bbox=dict(boxstyle='round,pad=0.12', fc='#f1f1f1', ec='none'))
    panel(ax, 'a', dx=0.0, dy=0.95)
    # ---------------- (b) the computed complexes (relaxed structures)
    def rot(v):
        th, ph = np.radians(22), np.radians(18)
        Rz = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        return v @ (Rx @ Rz).T
    sp = A['si']['species']
    charges = {}
    for d in sp.values():
        charges.setdefault(d['cfg'], []).append(d['q'])
    cells = ['V', 'VH', 'VH2', 'VH3', 'VH4', 'VO', 'VOH']
    BX0, BW, BH = 0.345, 0.083, 0.36
    slot_rect = lambda r, c: [BX0 + c * BW, 0.53 - r * 0.44, BW, BH]
    slots = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    for cfg, (r, c) in zip(cells, slots):
        axs = fig.add_axes(slot_rect(r, c)); axs.axis('off'); axs.set_aspect('equal')
        at = read(f'final64_si_{cfg}_q0.xyz'); rr = json.load(open(f'res64_si_{cfg}_q0.json'))
        nn = rr['nn_indices']; ref = at[nn[0]].position
        def mic(i):
            return ref + at.get_distance(nn[0], i, mic=True, vector=True)
        cen = np.mean([mic(i) for i in nn], axis=0)
        idx = list(nn) + [a.index for a in at if a.symbol in ('H', 'O')]
        P = {i: rot(mic(i) - cen) for i in idx}
        bonded = set()
        for i in idx:
            for j in idx:
                if i < j:
                    dij = np.linalg.norm(mic(i) - mic(j)); si, sj = at[i].symbol, at[j].symbol
                    lim = {('H', 'Si'): 1.7, ('O', 'Si'): 1.95, ('H', 'O'): 1.15}.get(tuple(sorted((si, sj))))
                    if lim and dij < lim:
                        axs.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], color='0.35', lw=1.0, zorder=1)
                        bonded |= {i, j}
        for i in nn:                                   # dangling bonds on unpassivated neighbours
            if i not in bonded or (cfg in ('VO', 'VOH') and i not in bonded):
                v = -P[i][:2]; v = v / np.linalg.norm(v)
                axs.plot([P[i][0], P[i][0] + 0.55 * v[0]], [P[i][1], P[i][1] + 0.55 * v[1]], color=VER, lw=1.4,
                         solid_capstyle='round', zorder=1)
        axs.add_patch(Circle((0, 0), 0.32, fc='none', ec='0.5', ls=':', lw=0.7))
        for i in sorted(idx, key=lambda k: P[k][2]):
            sym = at[i].symbol
            fc, rad, ec = {'Si': ('0.55', 0.36, 'none'), 'H': ('white', 0.2, '0.2'), 'O': (VER, 0.3, 'none')}[sym]
            axs.add_patch(Circle(P[i][:2], rad, fc=fc, ec=ec, lw=0.6, zorder=2 + P[i][2]))
        axs.set_xlim(-3.0, 3.0); axs.set_ylim(-3.0, 3.0)
        qs = sorted(charges.get(cfg, [0]), reverse=True)
        qtxt = ', '.join({1: '+', 0: '0', -1: '−', -2: '2−'}[q] for q in qs)
        lab = {'V': '$V$', 'VH': '$V$H', 'VH2': '$V$H$_2$', 'VH3': '$V$H$_3$', 'VH4': '$V$H$_4$', 'VO': '$V$–O', 'VOH': '$V$–OH'}[cfg]
        axs.set_title(lab, fontsize=fs(5.8), pad=0)
        axs.text(0, -3.05, f'$q$: {qtxt}', ha='center', va='top', fontsize=fs(5.0), color='0.25')
        if (r, c) == (0, 0):
            panel(axs, 'b', dx=-0.12, dy=1.08)
    axl = fig.add_axes(slot_rect(1, 3)); axl.axis('off'); axl.set_aspect('equal'); axl.set_xlim(-3, 3); axl.set_ylim(-3, 3)
    for k, (fc, ec, rad, t) in enumerate((('0.55', 'none', 0.36, 'Si'), ('white', '0.2', 0.2, 'H'), (VER, 'none', 0.3, 'O'))):
        axl.add_patch(Circle((-1.6, 2.0 - 1.35 * k), rad, fc=fc, ec=ec, lw=0.6))
        axl.text(-0.8, 2.0 - 1.35 * k, t, va='center', fontsize=fs(5.2))
    axl.plot([-2.0, -1.2], [-2.05, -2.05], color=VER, lw=1.4)
    axl.text(-0.8, -2.05, 'dangling\nbond', va='center', fontsize=fs(4.6), linespacing=0.9)
    # ---------------- (c) lower envelope of Omega vs dmu_H at mid-gap
    axc = fig.add_axes([0.76, 0.17, 0.225, 0.74])
    EFm = 0.3; mu = np.linspace(-1.45, 0.45, 600)
    lines = []
    for d in sp.values():
        G = d.get('Ef_G', d['Ef_ref']) + d['q'] * EFm - (d['mH'] - 2 * d['mO']) * mu
        lines.append((d['cfg'], d['q'], G))
        axc.plot(mu, G, color='0.78', lw=0.6, zorder=1)
    stack = np.array([g for _, _, g in lines]); imin = stack.argmin(0); env = stack.min(0)
    for k in sorted(set(imin.tolist())):
        m = imin == k; cfg, q, _ = lines[k]
        axc.plot(mu[m], env[m], color=SPECIES_COLOR[cfg], lw=2.4, zorder=3, solid_capstyle='butt')
        xm = mu[m].mean(); ym = env[m].mean()
        axc.text(xm, ym - 0.35, label_of(cfg, q), ha='center', va='top', fontsize=fs(5.6), color=SPECIES_COLOR[cfg] if cfg != 'VOH' else '#a08a00',
                 fontweight='bold', zorder=4, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
    axc.axvspan(0, 0.45, facecolor='white', alpha=0.6, edgecolor='0.4', hatch='////', lw=0, zorder=2)
    axc.axvspan(-1.45, -E_O2_H2O, facecolor='white', alpha=0.6, edgecolor='0.4', hatch='////', lw=0, zorder=2)
    for x in (0, -E_O2_H2O):
        axc.axvline(x, color='0.2', lw=0.7, ls='--', zorder=2)
    axc.set_xlim(-1.45, 0.45); axc.set_ylim(env.min() - 0.9, env.max() + 1.6)
    axc.set_xlabel(r'$\Delta\mu_{\mathrm{H}}$ (eV)'); axc.set_ylabel(r'$\Omega$ (eV), $E_F$ = 0.3 eV', labelpad=2)
    axc.text(0.5, 0.97, 'grey: all complexes\ncolor: lowest = stable', transform=axc.transAxes, ha='center', va='top', fontsize=fs(5.0), color='0.25',
             zorder=6, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
    panel(axc, 'c', dx=-0.32, dy=1.0)
    save(fig, 'fig1_concept')


def _crop_box(path, pad=8):
    from PIL import Image
    im = Image.open(path).convert('RGBA'); a = np.array(im)[:, :, 3]
    ys, xs = np.where(a > 8)
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))


def _crop(path, pad=8):
    from PIL import Image
    return np.array(Image.open(path).convert('RGBA').crop(_crop_box(path, pad)))


def _letter(ax, l, x=-0.02, y=1.0):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom', ha='left')


def _omega_panel(axc, lines_fam, EFm, title, mu_rng=(-1.45, 0.45), note=None, ylab=True):
    """lines_fam: list of (family_color, [(label, q, Ef, mH, mO), ...]); envelope drawn per family."""
    mu = np.linspace(*mu_rng, 800)
    allG = []
    for col, fam in lines_fam:
        G = np.array([Ef + q * EFm - (mH - 2 * mO) * mu for _, q, Ef, mH, mO in fam])
        for g in G:
            axc.plot(mu, g, color='0.82', lw=0.5, zorder=1)
        imin = G.argmin(0); env = G.min(0); allG.append(env)
        for k in sorted(set(imin.tolist())):
            m = imin == k; lab, q, *_ = fam[k]
            c = col(lab) if callable(col) else col
            axc.plot(mu[m], env[m], color=c, lw=2.0, zorder=3, solid_capstyle='butt')
            axc.text(mu[m].mean(), env[m].mean() - 0.3, lab + {1: '$^{+}$', 0: '$^{0}$', -1: '$^{-}$', -2: '$^{2-}$'}[q],
                     ha='center', va='top', fontsize=6.0, color=c if c != YEL else '#9a8500', fontweight='bold', zorder=4,
                     bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.85))
    for lo, hi in ((0, mu_rng[1]), (mu_rng[0], -E_O2_H2O)):
        axc.axvspan(lo, hi, facecolor='0.93', edgecolor='0.6', hatch='/////', lw=0, zorder=2)
    for x in (0, -E_O2_H2O):
        axc.axvline(x, color='0.3', lw=0.6, ls='--', zorder=2)
    lo = min(e.min() for e in allG); hi = max(e.max() for e in allG)
    axc.set_xlim(*mu_rng); axc.set_ylim(lo - 0.9, hi + 1.5)
    axc.set_xlabel(r'$\Delta\mu_{\mathrm{H}}$ (eV)', fontsize=6.5, labelpad=1.5)
    if ylab:
        axc.set_ylabel(r'$\Omega$ (eV)', fontsize=6.5, labelpad=1.5)
    axc.tick_params(labelsize=6, width=0.5, length=2.5, pad=1.5)
    for sp_ in axc.spines.values(): sp_.set_linewidth(0.5)
    axc.set_title(title, fontsize=6.3, pad=2)
    if note:
        axc.text(0.5, 0.97, note, transform=axc.transAxes, ha='center', va='top', fontsize=5.2, color='0.3', zorder=6,
                 bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9))


def _water_drop(ax, cx, cy, r, fc='#cfe6f2', ec='#5b8db3'):
    """Diagrammatic water drop (reservoir icon) in an image axis whose y axis points down."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    th = np.radians(np.linspace(135, 405, 90))
    pts = [(cx + r * np.cos(t), cy - r * np.sin(t)) for t in th]
    tip = (cx, cy - np.sqrt(2) * r * 1.25)
    verts = [tip] + pts + [tip]
    ax.add_patch(PathPatch(Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]), fc=fc, ec=ec, lw=0.8, zorder=4))
    ax.text(cx, cy + 0.1 * r, 'H$_2$O', fontsize=4.6, color='#1f4e79', ha='center', va='center', zorder=5)


def fig2_structures():
    """Fig. 2 (Oct 2026). Si row (a-c) and 3C-SiC row (d-f): (a,d) ray-traced relaxed cells, (b,e) relaxed
    complexes, (c,f) lower envelope of Omega vs dmu_H from the computed energies. Renders are produced with
    OVITO (render/render_slab.py, render_complexes.py, render_sic.py) and placed as images."""
    from PIL import Image
    RD = 'render'
    fig = plt.figure(figsize=(7.48, 6.0))
    R = lambda x, y, w, h, y0: [x, y0 + y * 0.5, w, h * 0.5]
    # ===================== silicon row (top) =====================
    y0 = 0.5
    ax = fig.add_axes(R(0.0, 0.03, 0.34, 0.86, y0)); ax.axis('off')
    box = _crop_box(f'{RD}/cell_icon.png'); arr = np.array(Image.open(f'{RD}/cell_icon.png').convert('RGBA').crop(box))
    msk = np.array(Image.open(f'{RD}/cell_icon_mask.png').convert('RGBA').crop(box)).astype(int)
    ax.imshow(arr); H, W = arr.shape[:2]
    g = (msk[:, :, 1] > msk[:, :, 0] + 25) & (msk[:, :, 1] > msk[:, :, 2] + 25) & (msk[:, :, 3] > 100)
    vy, vx = np.array(np.where(g)).mean(axis=1)
    ax.set_xlim(-0.04 * W, 1.42 * W); ax.set_ylim(1.06 * H, -0.30 * H)
    _water_drop(ax, 1.20 * W, -0.10 * H, 0.075 * W)
    ax.text(1.20 * W, 0.08 * H, 'water\n(pH, $U$)', fontsize=5.6, color='#1f4e79', ha='center', va='top', linespacing=0.95)
    ax.add_patch(FancyArrowPatch((1.11 * W, -0.06 * H), (vx + 0.04 * W, vy - 0.05 * H), connectionstyle='arc3,rad=0.25',
                                 arrowstyle='-|>', mutation_scale=7, lw=1.0, color=GRN, zorder=5))
    ax.text(0.80 * W, -0.16 * H, 'H, O, OH', fontsize=6.0, color='#00694d', ha='center', va='center')
    ax.add_patch(FancyArrowPatch((1.18 * W, 0.92 * H), (vx + 0.05 * W, vy + 0.05 * H), connectionstyle='arc3,rad=-0.25',
                                 arrowstyle='<|-|>', mutation_scale=7, lw=1.0, color=BLU, zorder=5))
    ax.text(1.20 * W, 0.92 * H, '$e^-$ ($E_F$)', fontsize=6.0, color=BLU, va='center')
    ax.text(0.0, 1.04 * H, 'Si (relaxed 64-atom cell)', fontsize=5.6, color='0.35', va='top')
    _letter(ax, 'a', x=0.0)
    sp = A['si']['species']; charges = {}
    for d in sp.values():
        charges.setdefault(d['cfg'], []).append(d['q'])
    cells = ['V', 'VH', 'VH2', 'VH3', 'VH4', 'VO', 'VOH']
    labs = {'V': '$V$', 'VH': '$V$H', 'VH2': '$V$H$_2$', 'VH3': '$V$H$_3$', 'VH4': '$V$H$_4$', 'VO': '$V$–O', 'VOH': '$V$–OH'}
    qfmt = lambda qs: '$q$ = ' + ', '.join({1: '+1', 0: '0', -1: '−1', -2: '−2'}[q] for q in sorted(qs, reverse=True))
    rect = lambda r, c: R(0.355 + c * 0.088, 0.50 - r * 0.45, 0.088, 0.36, y0)
    for k, cfg in enumerate(cells):
        r, c = divmod(k, 4)
        axs = fig.add_axes(rect(r, c)); axs.axis('off'); axs.imshow(_crop(f'{RD}/complex_{cfg}.png'))
        axs.set_title(labs[cfg], fontsize=6.5, pad=1)
        axs.text(0.5, -0.04, qfmt(charges.get(cfg, [0])), transform=axs.transAxes, ha='center', va='top', fontsize=5.2, color='0.3')
        if k == 0: _letter(axs, 'b', x=-0.08, y=1.12)
    axl = fig.add_axes(rect(1, 3)); axl.axis('off'); axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    for k, (col, sz, t) in enumerate((((0.36, 0.50, 0.66), 42, 'Si, vacancy\nneighbor'), ((0.76, 0.80, 0.86), 18, 'Si'),
                                       ((0.90, 0.90, 0.92), 14, 'H'), ((0.84, 0.29, 0.30), 26, 'O'), ((0.96, 0.64, 0.35), 34, 'vacancy site (a)'))):
        y = 0.88 - 0.19 * k
        axl.scatter([0.14], [y], s=sz, c=[col], edgecolors='0.45', linewidths=0.4, alpha=0.65 if t.startswith('vacancy') else 1)
        axl.text(0.3, y, t, va='center', fontsize=5.2, color='0.2', linespacing=0.9)
    axc = fig.add_axes(R(0.775, 0.16, 0.215, 0.74, y0))
    fam = [(d['cfg'], d['q'], d.get('Ef_G', d['Ef_ref']), d['mH'], d['mO']) for d in sp.values()]
    name = {'V': '$V$', 'VH': '$V$H', 'VH2': '$V$H$_2$', 'VH3': '$V$H$_3$', 'VH4': '$V$H$_4$', 'VO': '$V$–O', 'VOH': '$V$–OH'}
    fam = [(name[c], q, E, mH, mO) for c, q, E, mH, mO in fam]
    inv = {v: k for k, v in name.items()}
    _omega_panel(axc, [(lambda lab: SPECIES_COLOR[inv[lab]], fam)], 0.3, r'Si, $E_F$ = 0.3 eV (free energies)',
                 note='grey: all computed species\ncolored: lowest (stable)')
    _letter(axc, 'c', x=-0.3)
    # ===================== 3C-SiC row (bottom) =====================
    y0 = 0.0
    ax = fig.add_axes(R(0.0, 0.03, 0.33, 0.86, y0)); ax.axis('off')
    box = _crop_box(f'{RD}/sic_cell.png'); arr = np.array(Image.open(f'{RD}/sic_cell.png').convert('RGBA').crop(box))
    msk = np.array(Image.open(f'{RD}/sic_cell_mask.png').convert('RGBA').crop(box)).astype(int)
    ax.imshow(arr); H, W = arr.shape[:2]
    gC = (msk[:, :, 1] > msk[:, :, 0] + 25) & (msk[:, :, 1] > msk[:, :, 2] + 25) & (msk[:, :, 3] > 100)
    gS = (msk[:, :, 2] > msk[:, :, 0] + 25) & (msk[:, :, 2] > msk[:, :, 1] + 25) & (msk[:, :, 3] > 100)
    cy_, cx_ = np.array(np.where(gC)).mean(axis=1); sy_, sx_ = np.array(np.where(gS)).mean(axis=1)
    ax.set_xlim(-0.04 * W, 1.45 * W); ax.set_ylim(1.06 * H, -0.30 * H)
    _water_drop(ax, 1.20 * W, -0.10 * H, 0.075 * W)
    ax.text(1.20 * W, 0.08 * H, 'water\n(pH, $U$)', fontsize=5.6, color='#1f4e79', ha='center', va='top', linespacing=0.95)
    for (tx, ty, rad) in ((cx_, cy_, 0.15), (sx_, sy_, 0.35)):
        ax.add_patch(FancyArrowPatch((1.11 * W, -0.06 * H), (tx + 0.03 * W, ty - 0.04 * H), connectionstyle=f'arc3,rad={rad}',
                                     arrowstyle='-|>', mutation_scale=7, lw=1.0, color=GRN, zorder=5))
    ax.text(0.62 * W, -0.20 * H, 'H', fontsize=6.4, color='#00694d', ha='center', va='center', fontweight='bold')
    for k, (col, t) in enumerate((((0.96, 0.64, 0.35), '$V_{\\mathrm{C}}$ site (4 Si nbrs)'), ((0.60, 0.45, 0.78), '$V_{\\mathrm{Si}}$ site (4 C nbrs)'),
                                   ((0.58, 0.67, 0.79), 'Si'), ((0.36, 0.36, 0.39), 'C'))):
        yy = (0.52 + 0.12 * k) * H
        ax.scatter([1.06 * W], [yy], s=26 if k < 2 else 14, c=[col], edgecolors='0.45', linewidths=0.4, alpha=0.75 if k < 2 else 1)
        ax.text(1.10 * W, yy, t, fontsize=5.2, color='0.2', va='center')
    ax.text(0.0, 1.04 * H, '3C-SiC (64-atom cell)', fontsize=5.6, color='0.35', va='top')
    _letter(ax, 'd', x=0.0)
    slabs = {'VC': '$V_{\\mathrm{C}}$', 'VCH': '$V_{\\mathrm{C}}$–H', 'VSi': '$V_{\\mathrm{Si}}$', 'VSiH': '$V_{\\mathrm{Si}}$–H'}
    sub = {'VC': '$D_{2d}$ pairing', 'VCH': 'Si–H–Si bridge', 'VSi': 'C dangling bonds', 'VSiH': 'C–H bond'}
    sic = A['sic']['Si-rich']['species']; sq = {}
    for d in sic.values():
        sq.setdefault(d['cfg'], []).append(d['q'])
    for k, cfg in enumerate(['VC', 'VCH', 'VSi', 'VSiH']):
        r, c = divmod(k, 2)
        axs = fig.add_axes(R(0.34 + c * 0.17, 0.52 - r * 0.47, 0.15, 0.36, y0)); axs.axis('off')
        axs.imshow(_crop(f'{RD}/complex_sic_{cfg}.png'))
        axs.set_title(slabs[cfg] + '  ' + f'({sub[cfg]})', fontsize=6.0, pad=1)
        axs.text(0.5, -0.04, qfmt(sq.get(cfg, [0])), transform=axs.transAxes, ha='center', va='top', fontsize=5.2, color='0.3')
        if k == 0: _letter(axs, 'e', x=-0.06, y=1.12)
    axc = fig.add_axes(R(0.775, 0.16, 0.215, 0.74, y0))
    famC = [(slabs[d['cfg']], d['q'], d['Ef_ref'], d['mH'], d['mO']) for d in sic.values() if d['cfg'] in ('VC', 'VCH')]
    famS = [(slabs[d['cfg']], d['q'], d['Ef_ref'], d['mH'], d['mO']) for d in sic.values() if d['cfg'] in ('VSi', 'VSiH')]
    _omega_panel(axc, [(VER, famC), ((0.48, 0.33, 0.68), famS)], 1.25, r'3C-SiC, $E_F$ = 1.25 eV (static, Si-rich)')
    _letter(axc, 'f', x=-0.3)
    save(fig, 'fig2_structures')


def fig1_flow():
    """(superseded) Fig. 1 (Oct 2026): method schematic in five stages. Uses the rendered cell icon
    (render/render_slab.py nowater) and a miniature of the computed stability map."""
    from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle
    from hydration_engine import stability_map
    fig = plt.figure(figsize=(7.48, 2.55))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis('off'); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
    X = [0.008, 0.208, 0.408, 0.608, 0.802]; Wd = [0.185, 0.185, 0.185, 0.18, 0.19]
    fills = ['#eef3f8', '#f6f1ea', '#eef6f1', '#f4eff6', '#f2f2f2']
    titles = ['1  Supercell DFT', '2  Corrected energies', '3  Reservoirs', '4  Grand potential', '5  Stability map']
    for x, w, fc, t in zip(X, Wd, fills, titles):
        bg.add_patch(FancyBboxPatch((x, 0.06), w, 0.86, boxstyle='round,pad=0.004,rounding_size=0.018', fc=fc, ec='none'))
        bg.text(x + 0.012, 0.885, t, fontsize=7, fontweight='bold', color='0.15', va='center')
    for x, w in zip(X[:-1], Wd[:-1]):
        bg.add_patch(FancyArrowPatch((x + w + 0.002, 0.49), (x + w + 0.017, 0.49), arrowstyle='-|>', mutation_scale=9, lw=1.1, color='0.45'))
    # 1: rendered cell + text
    a1 = fig.add_axes([X[0] + 0.01, 0.36, Wd[0] - 0.02, 0.47]); a1.axis('off'); a1.imshow(_crop('render/cell_icon.png'))
    bg.text(X[0] + 0.012, 0.33, '64-atom cells, PBE,\nspin-polarized', fontsize=5.8, color='0.25', va='top', linespacing=1.1)
    bg.text(X[0] + 0.012, 0.19, 'complex $X^q$: charge $q$,\n$m_{\\mathrm{H}}$ H, $m_{\\mathrm{O}}$ O', fontsize=5.8, color='0.25', va='top', linespacing=1.1)
    # 2: three correction chips
    chips = [('finite size', '$E_{\\mathrm{MP}}+q\\,\\Delta V$'), ('basis', '700 eV single points'), ('vibrations', 'local H/O modes;\nH$_2$, H$_2$O at 298 K')]
    for k, (h, t) in enumerate(chips):
        y = 0.70 - 0.20 * k
        bg.add_patch(FancyBboxPatch((X[1] + 0.012, y - 0.075), Wd[1] - 0.024, 0.15, boxstyle='round,pad=0.003,rounding_size=0.01', fc='white', ec='#d9c9b3', lw=0.6))
        bg.text(X[1] + 0.02, y + 0.035, h, fontsize=5.6, color='#8a6a3e', fontweight='bold', va='center')
        bg.text(X[1] + 0.02, y - 0.022, t, fontsize=5.6, color='0.2', va='center', linespacing=1.0)
    bg.text(X[1] + Wd[1] / 2, 0.12, '$\\rightarrow\\ E_f^{G}(X^q)$', fontsize=7, color='0.15', ha='center', va='center')
    # 3: electron reservoir (band edges + E_F) and water reservoir (beaker)
    a3 = fig.add_axes([X[2] + 0.01, 0.10, Wd[2] - 0.02, 0.72]); a3.axis('off'); a3.set_xlim(0, 10); a3.set_ylim(0, 10)
    a3.add_patch(Rectangle((0.4, 6.0), 3.4, 1.0, fc='#c9d6e6', ec='none')); a3.add_patch(Rectangle((0.4, 8.6), 3.4, 1.0, fc='#c9d6e6', ec='none'))
    a3.text(2.1, 6.5, 'VB', fontsize=5.4, ha='center', va='center', color='0.3'); a3.text(2.1, 9.1, 'CB', fontsize=5.4, ha='center', va='center', color='0.3')
    a3.plot([0.4, 3.8], [7.8, 7.8], color=BLU, lw=1.0, ls='--'); a3.text(4.0, 7.8, '$E_F$', fontsize=6.2, color=BLU, va='center')
    a3.text(5.4, 8.6, 'electrons:\ndoping, bias', fontsize=5.6, color='0.25', va='center', linespacing=1.0)
    a3.add_patch(Polygon([[0.6, 4.6], [0.9, 0.6], [3.5, 0.6], [3.8, 4.6]], closed=True, fc='#d7ebf7', ec='#5b8db3', lw=0.8))
    for (cx, cy, r) in ((1.6, 1.6, 0.25), (2.6, 2.4, 0.2), (2.0, 3.2, 0.18), (3.0, 1.3, 0.16)):
        a3.add_patch(Circle((cx, cy), r, fc='white', ec='#5b8db3', lw=0.5))
    a3.text(2.2, 5.0, 'H$_2$O', fontsize=6.0, ha='center', color='#1f4e79', fontweight='bold')
    a3.text(4.3, 3.6, 'pH, $U$ (CHE):', fontsize=5.6, color='0.25', va='center')
    a3.text(4.3, 2.5, '$\\Delta\\mu_{\\mathrm{H}}=-eU-0.059\\,$pH', fontsize=5.6, color='0.2', va='center')
    a3.text(4.3, 1.3, '$\\mu_{\\mathrm{O}}=\\mu_{\\mathrm{H_2O}}-2\\mu_{\\mathrm{H}}$', fontsize=5.6, color='0.2', va='center')
    # 4: grand potential and minimisation
    xc = X[3] + Wd[3] / 2
    bg.add_patch(FancyBboxPatch((X[3] + 0.012, 0.50), Wd[3] - 0.024, 0.25, boxstyle='round,pad=0.003,rounding_size=0.01', fc='white', ec='#c8b4d6', lw=0.6))
    bg.text(xc, 0.66, '$\\Omega = E_f^{G} + q\\,E_F$', fontsize=6.8, ha='center', va='center', color='0.15')
    bg.text(xc, 0.57, '$-\\,(m_{\\mathrm{H}}-2m_{\\mathrm{O}})\\,\\Delta\\mu_{\\mathrm{H}}$', fontsize=6.8, ha='center', va='center', color='0.15')
    bg.text(xc, 0.40, 'minimize over', fontsize=6.0, ha='center', va='center', color='0.25')
    bg.text(xc, 0.32, '$q,\\ m_{\\mathrm{H}},\\ m_{\\mathrm{O}}$', fontsize=6.8, ha='center', va='center', color='#6b4c8a', fontweight='bold')
    bg.text(xc, 0.17, 'at every ($E_F$, $\\Delta\\mu_{\\mathrm{H}}$)', fontsize=5.8, ha='center', va='center', color='0.3')
    # 5: miniature of the computed silicon map (water stability window only)
    a5 = fig.add_axes([X[4] + 0.035, 0.16, Wd[4] - 0.05, 0.64])
    sp = A['si']['species']; gap = A['si']['host']['gap']
    S = [Species(d['cfg'], d['q'], d['mH'], d['mO'], d.get('Ef_G', d['Ef_ref'])) for d in sp.values()]
    EF = np.linspace(0, gap, 120); MU = np.linspace(-1.23, 0.3, 240); idx, _ = stability_map(S, EF, MU)
    pres = sorted(set(idx.ravel().tolist())); remap = {i: k for k, i in enumerate(pres)}
    a5.pcolormesh(EF, MU, np.vectorize(remap.get)(idx), cmap=ListedColormap([SPECIES_COLOR[S[i].name] for i in pres]),
                  vmin=-0.5, vmax=len(pres) - 0.5, shading='auto', rasterized=True)
    a5.axhspan(0, 0.3, facecolor='white', alpha=0.55, edgecolor='0.4', hatch='////', lw=0)
    for i in pres:
        m = idx == i
        if m.mean() < 0.03: continue
        yy, xx = np.where(m)
        a5.text(gap * 0.5, MU[int(np.median(yy))], label_of(S[i].name, S[i].q), fontsize=6, ha='center', va='center',
                color='white' if MU[int(np.median(yy))] < 0 else '0.15', fontweight='bold')
    a5.set_xlabel('$E_F$', fontsize=6.2, labelpad=1); a5.set_ylabel('$\\Delta\\mu_{\\mathrm{H}}$', fontsize=6.2, labelpad=1)
    a5.set_xticks([]); a5.set_yticks([])
    for sp_ in a5.spines.values(): sp_.set_linewidth(0.5)
    bg.text(X[4] + Wd[4] / 2, 0.09, 'silicon vacancy in water', fontsize=5.6, color='0.3', ha='center')
    save(fig, 'fig1_concept')


def fig1_electrochem():
    """(superseded) Fig. 1 (Oct 2026, v3). (a) Electrochemical picture: ray-traced relaxed Si cell under an illustrative
    water layer, wired as the working electrode against a reference electrode in the water, with a zoom on the
    vacancy showing candidate decorations (relaxed structures). (b) The computed Si stability map with its
    reservoirs attached to the axes: band-edge strip under E_F, pH scale (U = 0) on the right, water window."""
    from PIL import Image
    from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, ConnectionPatch
    from hydration_engine import stability_map
    RD = 'render'
    fig = plt.figure(figsize=(7.48, 3.25))
    # ---------------- (a) electrochemical picture
    ax = fig.add_axes([0.0, 0.02, 0.50, 0.92]); ax.axis('off')
    box = _crop_box(f'{RD}/slab.png'); arr = np.array(Image.open(f'{RD}/slab.png').convert('RGBA').crop(box))
    msk = np.array(Image.open(f'{RD}/slab_mask.png').convert('RGBA').crop(box)).astype(int)
    H, W = arr.shape[:2]
    g = (msk[:, :, 1] > msk[:, :, 0] + 25) & (msk[:, :, 1] > msk[:, :, 2] + 25) & (msk[:, :, 3] > 100)
    vy, vx = np.array(np.where(g)).mean(axis=1)
    ax.set_xlim(-0.05 * W, 1.95 * W); ax.set_ylim(1.08 * H, -0.12 * H)
    # electrolyte bath behind the water molecules, crystal in front
    ax.add_patch(FancyBboxPatch((-0.03 * W, -0.06 * H), 1.10 * W, 0.47 * H, boxstyle='round,pad=0,rounding_size=12',
                                fc='#e3eff8', ec='none', zorder=0))
    ax.imshow(arr, extent=(0, W, H, 0), zorder=1)
    ax.text(0.0, -0.085 * H, 'water: pH, $U$', fontsize=6.5, color='#1f4e79', fontweight='bold', va='bottom')
    ax.text(0.0, 1.06 * H, 'Si electrode (relaxed 64-atom cell)', fontsize=6.0, color='0.3', va='top')
    # external circuit: crystal -> potentiostat U -> reference electrode dipping into the water
    yb = 0.97 * H; xR = 1.04 * W
    ax.plot([0.80 * W, xR], [0.86 * H, 0.86 * H], color='0.35', lw=0.9, zorder=2)
    ax.plot([xR, xR], [0.86 * H, 0.58 * H], color='0.35', lw=0.9)
    ax.add_patch(Circle((xR, 0.50 * H), 0.065 * W, fc='white', ec='0.35', lw=0.9, zorder=3))
    ax.text(xR, 0.50 * H, '$U$', fontsize=7, ha='center', va='center', zorder=4)
    ax.plot([xR, xR], [0.425 * H, 0.06 * H], color='0.35', lw=0.9, zorder=2)
    ax.plot([xR, 0.97 * W], [0.06 * H, 0.06 * H], color='0.35', lw=0.9)
    ax.add_patch(Rectangle((0.93 * W, 0.06 * H), 0.08 * W, 0.22 * H, fc='#9aa7b6', ec='0.35', lw=0.7, zorder=2))
    ax.text(1.0 * W, 0.31 * H, 'ref.', fontsize=5.5, color='0.3', ha='center', va='top', zorder=4)
    ax.text(xR - 0.085 * W, 0.50 * H, 'vs SHE', fontsize=5.3, color='0.3', ha='right', va='center', zorder=4,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
    # zoom on the vacancy -> inset with candidate decorations
    ax.add_patch(Circle((vx, vy), 0.075 * W, fc='none', ec=VER, lw=1.1, zorder=5))
    inset = [0.315, 0.13, 0.185, 0.44]                     # figure coordinates
    axi = fig.add_axes(inset); axi.set_xlim(0, 1); axi.set_ylim(0, 1); axi.axis('off')
    axi.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle='round,pad=0,rounding_size=0.05', fc='white', ec=VER, lw=1.0,
                                 transform=axi.transAxes))
    axi.text(0.5, 0.93, 'which complex $X^{q}$?', fontsize=6.2, ha='center', va='center', fontweight='bold', color='0.2')
    sub = [('V', '$V$'), ('VH4', '$V$H$_4$'), ('VO', '$V$–O'), ('VOH', '$V$–OH')]
    for k, (cfg, lab) in enumerate(sub):
        r, c = divmod(k, 2)
        a_ = fig.add_axes([inset[0] + 0.012 + c * 0.088, inset[1] + 0.035 + (1 - r) * 0.18, 0.08, 0.15]); a_.axis('off')
        a_.imshow(_crop(f'{RD}/complex_{cfg}.png')); a_.set_title(lab, fontsize=5.8, pad=0)
    axi.text(0.5, 0.035, 'charge $q$; H and O content', fontsize=5.3, ha='center', va='bottom', color='0.35')
    for yy in (0.18, 0.80):
        con = ConnectionPatch(xyA=(vx + 0.06 * W, vy - (0.04 if yy > 0.5 else -0.04) * H), coordsA=ax.transData,
                              xyB=(0.01, yy), coordsB=axi.transAxes, color=VER, lw=0.7, ls='--')
        fig.add_artist(con)
    ax.text(-0.05 * W, -0.12 * H, 'a', fontsize=8, fontweight='bold', va='bottom')
    # ---------------- (b) stability map with reservoirs on the axes
    sp = A['si']['species']; gap = A['si']['host']['gap']
    S = [Species(d['cfg'], d['q'], d['mH'], d['mO'], d.get('Ef_G', d['Ef_ref'])) for d in sp.values()]
    mlo, mhi = -1.40, 0.35
    axm = fig.add_axes([0.605, 0.24, 0.27, 0.58])
    EF = np.linspace(0, gap, 200); MU = np.linspace(mlo, mhi, 400); idx, _ = stability_map(S, EF, MU)
    pres = sorted(set(idx.ravel().tolist())); remap = {i: k for k, i in enumerate(pres)}
    axm.pcolormesh(EF, MU, np.vectorize(remap.get)(idx), cmap=ListedColormap([SPECIES_COLOR[S[i].name] for i in pres]),
                   vmin=-0.5, vmax=len(pres) - 0.5, shading='auto', rasterized=True)
    for lo, hi in ((0, mhi), (mlo, -E_O2_H2O)):
        axm.axhspan(lo, hi, facecolor='white', alpha=0.55, edgecolor='0.4', hatch='////', lw=0)
    axm.axhline(0, color='0.2', lw=0.6, ls='--'); axm.axhline(-E_O2_H2O, color='0.2', lw=0.6, ls='--')
    for i in pres:
        m = idx == i
        if m.mean() < 0.03: continue
        yy, xx = np.where(m); yc = MU[int(np.median(yy))]
        axm.text(gap * 0.5, yc, label_of(S[i].name, S[i].q), fontsize=6.5, ha='center', va='center',
                 color='white' if yc < 0 else '0.1', fontweight='bold')
    axm.text(gap * 0.97, 0.02, 'H$_2$/H$_2$O', fontsize=5.3, ha='right', va='bottom', color='0.2')
    axm.text(gap * 0.97, -E_O2_H2O + 0.02, 'O$_2$/H$_2$O', fontsize=5.3, ha='right', va='bottom', color='0.2')
    axm.set_xlim(0, gap); axm.set_ylim(mlo, mhi); axm.set_xticks([])
    axm.set_ylabel(r'$\Delta\mu_{\mathrm{H}}$ (eV)', fontsize=6.5, labelpad=1.5)
    axm.tick_params(labelsize=6, width=0.5, length=2.5, pad=1.5)
    for sp_ in axm.spines.values(): sp_.set_linewidth(0.6)
    # right axis: pH at U = 0 (CHE), with bias arrows
    kT10 = 8.617333e-5 * 298.15 * np.log(10)
    axp = axm.twinx(); axp.set_ylim(mlo, mhi)
    axp.set_yticks([-kT10 * p for p in (0, 7, 14)]); axp.set_yticklabels(['0', '7', '14'], fontsize=6)
    axp.set_ylabel('pH at $U$ = 0', fontsize=6.5, labelpad=2); axp.tick_params(width=0.5, length=2.5, pad=1.5)
    for sp_ in axp.spines.values(): sp_.set_linewidth(0.6)
    axa = fig.add_axes([0.915, 0.24, 0.05, 0.58]); axa.axis('off'); axa.set_xlim(0, 1); axa.set_ylim(mlo, mhi)
    axa.add_patch(FancyArrowPatch((0.35, -0.65), (0.35, 0.05), arrowstyle='-|>', mutation_scale=7, lw=0.9, color=BLU))
    axa.add_patch(FancyArrowPatch((0.35, -0.75), (0.35, -1.35), arrowstyle='-|>', mutation_scale=7, lw=0.9, color=VER))
    axa.text(0.62, -0.30, 'cathodic $U$', fontsize=5.3, color=BLU, rotation=90, va='center', ha='center')
    axa.text(0.62, -1.05, 'anodic $U$', fontsize=5.3, color=VER, rotation=90, va='center', ha='center')
    # band-edge strip under the E_F axis
    axb = fig.add_axes([0.605, 0.10, 0.27, 0.10]); axb.set_xlim(-0.12, gap + 0.12); axb.set_ylim(0, 1); axb.axis('off')
    axb.add_patch(Rectangle((-0.12, 0.25), 0.12, 0.5, fc='#c9d6e6', ec='none')); axb.add_patch(Rectangle((gap, 0.25), 0.12, 0.5, fc='#c9d6e6', ec='none'))
    axb.text(-0.06, 0.5, 'VB', fontsize=5.3, ha='center', va='center', color='0.3'); axb.text(gap + 0.06, 0.5, 'CB', fontsize=5.3, ha='center', va='center', color='0.3')
    grad = np.linspace(0, 1, 200)[None, :]
    axb.imshow(grad, extent=(0, gap, 0.25, 0.75), aspect='auto', cmap=mpl.colors.LinearSegmentedColormap.from_list('pn', ['#f3d9c4', '#d6e4f2']))
    axb.text(0.01, 0.95, 'p-type', fontsize=5.3, color='0.35', va='top'); axb.text(gap - 0.01, 0.95, 'n-type', fontsize=5.3, color='0.35', va='top', ha='right')
    axb.text(gap / 2, -0.05, '$E_F$ (doping, bias)', fontsize=6.3, ha='center', va='top', color='0.15')
    # formula band
    fig.text(0.74, 0.93, r'$\Omega = E_f^{G}(X^{q}) + q\,E_F - (m_{\mathrm{H}}-2m_{\mathrm{O}})\,\Delta\mu_{\mathrm{H}}$', fontsize=7, ha='center', va='center')
    fig.text(0.74, 0.865, r'minimized over $q,\ m_{\mathrm{H}},\ m_{\mathrm{O}}$ at every point', fontsize=6, ha='center', va='center', color='0.35')
    fig.text(0.545, 0.93, 'b', fontsize=8, fontweight='bold', va='center')
    save(fig, 'fig1_concept')


def fig1_methods():
    """(superseded) Fig. 1 (Oct 2026, v4): methodology. (a) configuration space of the computed complexes (relaxed
    structures, charge states), (b) how one formation energy is built (real numbers for V-O^-),
    (c) the two reservoirs, (d) Omega planes over (E_F, dmu_H) and their lower envelope, (e) the
    projected stability map with the pH axis. Everything except the drawn band diagram is computed data."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from hydration_engine import stability_map
    RD = 'render'
    sp = A['si']['species']; gap = A['si']['host']['gap']
    fig = plt.figure(figsize=(7.48, 4.75))
    def title(x, y, l, t):
        fig.text(x, y, l, fontsize=8, fontweight='bold', va='bottom'); fig.text(x + 0.018, y, t, fontsize=7, va='bottom', color='0.15')
    # ---------------- (a) configuration space
    title(0.005, 0.955, 'a', 'Complexes and charge states')
    charges = {}
    for d in sp.values():
        charges.setdefault(d['cfg'], []).append(d['q'])
    grid = {(0, m): c for m, c in enumerate(['V', 'VH', 'VH2', 'VH3', 'VH4'])}
    grid.update({(1, 0): 'VO', (1, 1): 'VOH'})
    labs = {'V': '$V$', 'VH': '$V$H', 'VH2': '$V$H$_2$', 'VH3': '$V$H$_3$', 'VH4': '$V$H$_4$', 'VO': '$V$–O', 'VOH': '$V$–OH'}
    X0, CW, Y0, CH = 0.055, 0.084, 0.565, 0.165
    for m in range(5):
        fig.text(X0 + m * CW + CW / 2, 0.925, f'$m_{{\\mathrm{{H}}}}$ = {m}', fontsize=6, ha='center', color='0.45')
    fig.text(0.012, Y0 + 1.5 * CH + 0.02, 'no O', fontsize=6, rotation=90, va='center', color='0.3')
    fig.text(0.012, Y0 + 0.5 * CH, '+O', fontsize=6, rotation=90, va='center', color='0.3')
    QORD = [1, 0, -1, -2]
    for r in (0, 1):
        for m in range(5):
            x, y = X0 + m * CW, Y0 + (1 - r) * (CH + 0.02)
            cfg = grid.get((r, m))
            if cfg is None:
                fig.add_artist(FancyBboxPatch((x + 0.008, y + 0.02), CW - 0.016, CH - 0.03, boxstyle='round,pad=0,rounding_size=0.008',
                                              fc='none', ec='0.75', ls=(0, (2, 2)), lw=0.6, transform=fig.transFigure))
                fig.text(x + CW / 2, y + CH / 2, 'not\ncomputed', fontsize=5, ha='center', va='center', color='0.6', linespacing=0.9)
                continue
            axs = fig.add_axes([x + 0.004, y + 0.03, CW - 0.008, CH - 0.045]); axs.axis('off')
            axs.imshow(_crop(f'{RD}/complex_{cfg}.png')); axs.set_title(labs[cfg], fontsize=6, pad=0)
            for k, q in enumerate(QORD):
                on = q in charges.get(cfg, [])
                fig.add_artist(mpl.patches.Ellipse((x + CW / 2 + (k - 1.5) * 0.012, y + 0.016), 0.0084, 0.0084 * 7.48 / 4.75, transform=fig.transFigure,
                                                   fc=BLU if on else 'white', ec=BLU if on else '0.7', lw=0.5))
    fig.text(X0, 0.528, 'charge states computed:', fontsize=5.5, color='0.35', va='center')
    for k, q in enumerate(QORD):
        xx = X0 + 0.135 + k * 0.03
        fig.add_artist(mpl.patches.Ellipse((xx, 0.528), 0.0084, 0.0084 * 7.48 / 4.75, transform=fig.transFigure, fc=BLU, ec=BLU, lw=0.5))
        fig.text(xx + 0.007, 0.528, {1: '+1', 0: '0', -1: '−1', -2: '−2'}[q], fontsize=5.5, va='center', color='0.3')
    # ---------------- (b) one formation energy, step by step (V-O^-)
    title(0.475, 0.955, 'b', 'Corrected formation energy')
    d = sp['VO_q-1']
    base = d['Ef_raw'] + d['q_times_vbm']
    steps = [('DFT\n($E_F$=VBM)', base, None), ('$q\\Delta V$', d['q'] * d['dV_align'], '#8fb3d9'), ('$E_{\\mathrm{MP}}$', d['E_MP'], '#8fb3d9'),
             ('cutoff', d['cutoff_shift'], '#d9b38f'), ('vib.', d['Ef_G'] - d['Ef_ref'], '#9fcfa8')]
    axb = fig.add_axes([0.53, 0.575, 0.19, 0.33])
    run = 0.0
    for k, (lab, v, col) in enumerate(steps):
        if col is None:
            axb.bar(k, v, color='0.55', width=0.62); run = v
        else:
            axb.bar(k, v, bottom=run, color=col, width=0.62); axb.plot([k - 1 + 0.31, k - 0.31], [run, run], color='0.5', lw=0.5, ls=':')
            axb.text(k, run + v + 0.04, f'+{v:.2f}', fontsize=5.5, ha='center', va='bottom', color='0.25'); run += v
    axb.bar(len(steps), run, color=SPECIES_COLOR['VO'], width=0.62)
    axb.plot([len(steps) - 1 + 0.31, len(steps) - 0.31], [run, run], color='0.5', lw=0.5, ls=':')
    axb.text(len(steps), run + 0.04, f'{run:.2f}', fontsize=5.8, ha='center', va='bottom', fontweight='bold', color='0.15')
    axb.set_xticks(range(len(steps) + 1)); axb.set_xticklabels([s_[0] for s_ in steps] + ['$E_f^{G}$'], fontsize=5.5)
    axb.set_ylim(1.6, run + 0.45); axb.set_ylabel('$E_f$ of $V$–O$^{-}$ (eV)', fontsize=6.3, labelpad=1.5)
    axb.tick_params(axis='y', labelsize=5.8, width=0.5, length=2.5, pad=1.5); axb.tick_params(axis='x', length=0, pad=2)
    for sp_ in axb.spines.values(): sp_.set_linewidth(0.5)
    # ---------------- (c) reservoirs
    title(0.745, 0.955, 'c', 'Reservoirs')
    axe = fig.add_axes([0.755, 0.60, 0.06, 0.30]); axe.set_xlim(0, 1); axe.set_ylim(-0.4, 1.0); axe.axis('off')
    axe.add_patch(Rectangle((0.05, -0.4), 0.9, 0.4, fc='#c9d6e6', ec='none')); axe.add_patch(Rectangle((0.05, 0.62), 0.9, 0.38, fc='#c9d6e6', ec='none'))
    axe.text(0.5, -0.2, 'VB', fontsize=5.5, ha='center', va='center', color='0.3'); axe.text(0.5, 0.81, 'CB', fontsize=5.5, ha='center', va='center', color='0.3')
    axe.plot([0.05, 0.95], [0.3, 0.3], color=BLU, lw=1.0, ls='--'); axe.text(0.62, 0.36, '$E_F$', fontsize=6, ha='center', va='bottom', color=BLU)
    axe.add_patch(FancyArrowPatch((0.3, 0.08), (0.3, 0.52), arrowstyle='<|-|>', mutation_scale=6, lw=0.7, color=BLU))
    axe.text(0.5, -0.5, 'electrons', fontsize=5.5, ha='center', va='top', color='0.3')
    axh = fig.add_axes([0.875, 0.62, 0.11, 0.28])
    pH = np.linspace(0, 14, 50); kT10 = 8.617333e-5 * 298.15 * np.log(10)
    for U, col in ((-0.5, BLU), (0.0, '0.25'), (0.5, VER)):
        axh.plot(pH, -U - kT10 * pH, color=col, lw=1.0)
        axh.text(14.3, -U - kT10 * 14, f'{U:+.1f} V' if U else '0 V', fontsize=5.2, color=col, va='center')
    axh.set_xlim(0, 14); axh.set_xticks([0, 7, 14]); axh.set_xlabel('pH', fontsize=6, labelpad=1)
    axh.set_ylabel('$\\Delta\\mu_{\\mathrm{H}}$ (eV)', fontsize=6, labelpad=1)
    axh.tick_params(labelsize=5.5, width=0.5, length=2, pad=1)
    for sp_ in axh.spines.values(): sp_.set_linewidth(0.5)
    axh.set_title('water (CHE)', fontsize=6, pad=2)
    fig.text(0.86, 0.535, '$\\mu_{\\mathrm{O}}=\\mu_{\\mathrm{H_2O}}-2\\mu_{\\mathrm{H}}$', fontsize=6.2, color='0.2', va='center')
    # ---------------- (d) Omega planes and their lower envelope
    title(0.005, 0.455, 'd', 'Minimize $\\Omega$ over $q$, $m_{\\mathrm{H}}$, $m_{\\mathrm{O}}$')
    keys = [k for k in sp if k in ('V_q1', 'V_q0', 'VO_q0', 'VOH_q0', 'VH4_q0', 'V_q-1', 'VH_q0')]
    ef = np.linspace(0, gap, 30); mu = np.linspace(-1.3, 0.35, 40); EFg, MUg = np.meshgrid(ef, mu)
    Z = {k: sp[k]['Ef_G'] + sp[k]['q'] * EFg - (sp[k]['mH'] - 2 * sp[k]['mO']) * MUg for k in keys}
    stack = np.array([Z[k] for k in keys]); imin = stack.argmin(0); env = stack.min(0)
    ax3 = fig.add_axes([0.0, 0.02, 0.50, 0.42], projection='3d')
    for k in keys:
        ax3.plot_surface(EFg, MUg, Z[k], color='0.75', alpha=0.13, lw=0, antialiased=False, shade=False)
        ax3.plot_wireframe(EFg, MUg, Z[k], rstride=39, cstride=29, color='0.6', lw=0.4, alpha=0.9)
    fc = np.empty(env.shape + (4,))
    for j, k in enumerate(keys):
        fc[imin == j] = mpl.colors.to_rgba(SPECIES_COLOR[sp[k]['cfg']], 0.95)
    ax3.plot_surface(EFg, MUg, env, facecolors=fc, rstride=1, cstride=1, lw=0, antialiased=False, shade=False)
    zbot = env.min() - 1.2
    ax3.contourf(EFg, MUg, imin, zdir='z', offset=zbot, levels=np.arange(-0.5, len(keys)),
                 colors=[SPECIES_COLOR[sp[k]['cfg']] for k in keys], alpha=0.85)
    ax3.set_zlim(zbot, env.max() + 1.5); ax3.view_init(elev=22, azim=-58)
    ax3.set_xlabel('$E_F$ (eV)', fontsize=6, labelpad=-8); ax3.set_ylabel('$\\Delta\\mu_{\\mathrm{H}}$ (eV)', fontsize=6, labelpad=-6)
    ax3.set_zlabel('$\\Omega$', fontsize=6.5, labelpad=-10)
    ax3.tick_params(labelsize=5, pad=-3); ax3.set_xticks([0, 0.3, 0.6]); ax3.set_yticks([-1, -0.5, 0]); ax3.set_zticks([])
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0)); axis.pane.set_edgecolor('0.85')
    ax3.grid(False)
    fig.text(0.035, 0.40, 'grey: $\\Omega$ of every species\ncolored: lowest $\\Omega$ (stable)\nfloor: projection = map', fontsize=5.5, color='0.35', va='top', linespacing=1.1)
    # ---------------- (e) projected map with the pH axis
    title(0.555, 0.455, 'e', 'Stability map')
    axm = fig.add_axes([0.60, 0.09, 0.27, 0.32])
    S = [Species(d_['cfg'], d_['q'], d_['mH'], d_['mO'], d_.get('Ef_G', d_['Ef_ref'])) for d_ in sp.values()]
    mlo, mhi = -1.40, 0.35
    EF = np.linspace(0, gap, 200); MU = np.linspace(mlo, mhi, 400); idx, _ = stability_map(S, EF, MU)
    pres = sorted(set(idx.ravel().tolist())); remap = {i: k for k, i in enumerate(pres)}
    axm.pcolormesh(EF, MU, np.vectorize(remap.get)(idx), cmap=ListedColormap([SPECIES_COLOR[S[i].name] for i in pres]),
                   vmin=-0.5, vmax=len(pres) - 0.5, shading='auto', rasterized=True)
    for lo, hi in ((0, mhi), (mlo, -E_O2_H2O)):
        axm.axhspan(lo, hi, facecolor='white', alpha=0.55, edgecolor='0.4', hatch='////', lw=0)
    axm.axhline(0, color='0.2', lw=0.6, ls='--'); axm.axhline(-E_O2_H2O, color='0.2', lw=0.6, ls='--')
    for i in pres:
        m_ = idx == i
        if m_.mean() < 0.03: continue
        yy, xx = np.where(m_); yc = MU[int(np.median(yy))]
        axm.text(gap * 0.5, yc, label_of(S[i].name, S[i].q), fontsize=6.3, ha='center', va='center',
                 color='white' if yc < 0 else '0.1', fontweight='bold')
    axm.set_xlim(0, gap); axm.set_ylim(mlo, mhi)
    axm.set_xlabel('$E_F - E_{\\mathrm{VBM}}$ (eV)', fontsize=6.3, labelpad=1.5); axm.set_ylabel('$\\Delta\\mu_{\\mathrm{H}}$ (eV)', fontsize=6.3, labelpad=1.5)
    axm.tick_params(labelsize=5.8, width=0.5, length=2.5, pad=1.5)
    for sp_ in axm.spines.values(): sp_.set_linewidth(0.6)
    axp = axm.twinx(); axp.set_ylim(mlo, mhi)
    axp.set_yticks([-kT10 * p for p in (0, 7, 14)]); axp.set_yticklabels(['0', '7', '14'], fontsize=5.8)
    axp.set_ylabel('pH at $U$ = 0', fontsize=6.3, labelpad=2); axp.tick_params(width=0.5, length=2.5, pad=1.5)
    for sp_ in axp.spines.values(): sp_.set_linewidth(0.6)
    fig.text(0.955, 0.25, 'hatched:\nwater\nunstable', fontsize=5.3, color='0.4', va='center', linespacing=1.0)
    # ---------------- flow arrows
    for (x0, y0, x1, y1) in ((0.49, 0.25, 0.555, 0.25),):
        fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=8, lw=0.9,
                                       color='0.55', connectionstyle='arc3,rad=0.0'))
    save(fig, 'fig1_concept')


def fig1():
    """Fig. 1 (Oct 2026, v5): the original concept figure (electron + water reservoirs -> minimize
    Omega[X^q] over q, m_H, m_O -> stability map), redrawn in a cleaner style: soft cards with small
    icons, thin curved connectors, and the computed silicon map in place of the cartoon map."""
    from matplotlib.patches import FancyBboxPatch, Rectangle, PathPatch
    from matplotlib.path import Path
    from hydration_engine import stability_map
    fig = plt.figure(figsize=(3.4, 2.3))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis('off'); bg.set_xlim(0, 3.4); bg.set_ylim(0, 2.3)   # inches
    def card(x, y, w, h, fc, ec):
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08', fc=fc, ec=ec, lw=0.6))
    BLUE_D, BLUE_L, BLUE_E = '#2f5d8a', '#eaf1f8', '#b9cde2'
    TEAL_D, TEAL_L, TEAL_E = '#1f6f5c', '#e8f4ef', '#b5d9cb'
    VIO_D, VIO_L, VIO_E = '#5b4a7a', '#f1eef6', '#cfc5df'
    # ---- electron reservoir card (top left)
    card(0.05, 1.27, 1.22, 0.93, BLUE_L, BLUE_E)
    bg.add_patch(Rectangle((0.15, 1.95), 0.32, 0.13, fc='#c3d4e6', ec='none')); bg.text(0.31, 2.015, 'CB', fontsize=4.8, ha='center', va='center', color='0.3')
    bg.add_patch(Rectangle((0.15, 1.38), 0.32, 0.13, fc='#c3d4e6', ec='none')); bg.text(0.31, 1.445, 'VB', fontsize=4.8, ha='center', va='center', color='0.3')
    bg.plot([0.15, 0.47], [1.73, 1.73], color=BLUE_D, lw=0.9, ls=(0, (3, 1.5)))
    bg.add_patch(FancyArrowPatch((0.40, 1.57), (0.40, 1.89), arrowstyle='<|-|>', mutation_scale=5, lw=0.6, color=BLUE_D))
    bg.text(0.21, 1.785, '$E_F$', fontsize=5.6, color=BLUE_D, ha='center')
    bg.text(0.58, 1.93, 'Electron', fontsize=6.6, fontweight='bold', color=BLUE_D, va='center')
    bg.text(0.58, 1.79, 'reservoir', fontsize=6.6, fontweight='bold', color=BLUE_D, va='center')
    bg.text(0.58, 1.58, '$E_F$', fontsize=6.2, color='0.15', va='center')
    bg.text(0.58, 1.44, '(doping, bias)', fontsize=5.4, color='0.35', va='center')
    # ---- water reservoir card (bottom left)
    card(0.05, 0.10, 1.22, 1.05, TEAL_L, TEAL_E)
    # water-drop icon
    cx, cy, r = 0.31, 0.55, 0.15
    th = np.radians(np.linspace(135, 405, 90))          # lower arc between the two tangent points
    pts = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in th]
    tip = (cx, cy + np.sqrt(2) * r * 1.25)                  # tangent lines meet above the circle
    drop = [tip] + pts + [tip]
    bg.add_patch(PathPatch(Path(drop, [Path.MOVETO] + [Path.LINETO] * (len(drop) - 2) + [Path.CLOSEPOLY]), fc='#cfe6f2', ec='#5b8db3', lw=0.7))
    bg.text(cx - 0.06, cy - 0.03, 'H$^+$', fontsize=4.6, color='#1f4e79', ha='center', va='center')
    bg.text(cx + 0.055, cy + 0.06, 'OH$^-$', fontsize=4.2, color='#1f4e79', ha='center', va='center')
    bg.text(cx, cy - 0.25, 'H$_2$O', fontsize=5.4, color='#1f4e79', ha='center', va='center')
    bg.text(0.58, 0.95, 'Water', fontsize=6.6, fontweight='bold', color=TEAL_D, va='center')
    bg.text(0.58, 0.81, 'reservoir', fontsize=6.6, fontweight='bold', color=TEAL_D, va='center')
    bg.text(0.58, 0.60, '$\\mu_{\\mathrm{H}}$(pH, $U$)', fontsize=6.2, color='0.15', va='center')
    bg.text(0.58, 0.47, '(CHE)', fontsize=5.4, color='0.35', va='center')
    bg.text(0.58, 0.27, '$\\mu_{\\mathrm{O}}=\\mu_{\\mathrm{H_2O}}-2\\mu_{\\mathrm{H}}$', fontsize=5.5, color='0.15', va='center')
    # ---- grand potential card (centre)
    card(1.47, 0.55, 0.92, 1.18, VIO_L, VIO_E)
    ai = fig.add_axes([1.53 / 3.4, 1.12 / 2.3, 0.80 / 3.4, 0.55 / 2.3]); ai.axis('off'); ai.imshow(_crop('render/complex_VOH.png'))
    bg.text(1.93, 0.98, '$\\Omega[X^{q}]$', fontsize=7.6, color=VIO_D, ha='center', va='center')
    bg.text(1.93, 0.81, 'min over', fontsize=5.6, color='0.35', ha='center', va='center')
    bg.text(1.93, 0.67, '$q,\\ m_{\\mathrm{H}},\\ m_{\\mathrm{O}}$', fontsize=6.4, color='0.15', ha='center', va='center')
    # ---- connectors
    for (x0, y0, x1, y1, rad, col) in ((1.27, 1.73, 1.47, 1.30, -0.35, BLUE_D), (1.27, 0.63, 1.47, 0.95, 0.35, TEAL_D)):
        bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=6, lw=0.8, color=col))
    bg.add_patch(FancyArrowPatch((2.39, 1.14), (2.55, 1.14), arrowstyle='-|>', mutation_scale=6, lw=0.8, color=VIO_D))
    # ---- stability map (computed, silicon vacancy, water window only)
    sp = A['si']['species']; gap = A['si']['host']['gap']
    S = [Species(d['cfg'], d['q'], d['mH'], d['mO'], d.get('Ef_G', d['Ef_ref'])) for d in sp.values()]
    am = fig.add_axes([2.70 / 3.4, 0.42 / 2.3, 0.62 / 3.4, 1.42 / 2.3])
    mlo, mhi = -1.30, 0.30
    EF = np.linspace(0, gap, 120); MU = np.linspace(mlo, mhi, 260); idx, _ = stability_map(S, EF, MU)
    pres = sorted(set(idx.ravel().tolist())); remap = {i: k for k, i in enumerate(pres)}
    am.pcolormesh(EF, MU, np.vectorize(remap.get)(idx), cmap=ListedColormap([SPECIES_COLOR[S[i].name] for i in pres]),
                  vmin=-0.5, vmax=len(pres) - 0.5, shading='auto', rasterized=True)
    for lo, hi in ((0, mhi), (mlo, -E_O2_H2O)):
        am.axhspan(lo, hi, facecolor='white', alpha=0.55, edgecolor='0.45', hatch='////', lw=0)
    for i in pres:
        m_ = idx == i
        if m_.mean() < 0.03: continue
        yy, xx = np.where(m_); yc = MU[int(np.median(yy))]
        am.text(gap * 0.5, yc, label_of(S[i].name, S[i].q), fontsize=5.6, ha='center', va='center',
                color='white' if yc < 0 else '0.1', fontweight='bold')
    am.set_xlim(0, gap); am.set_ylim(mlo, mhi); am.set_xticks([]); am.set_yticks([])
    am.set_xlabel('$E_F$', fontsize=6.2, labelpad=2); am.set_ylabel('$\\Delta\\mu_{\\mathrm{H}}$', fontsize=6.2, labelpad=2)
    for sp_ in am.spines.values(): sp_.set_linewidth(0.6); sp_.set_color('0.35')
    am.set_title('stable species', fontsize=6.2, pad=3, color='0.2')
    bg.text(3.01, 0.13, 'Si vacancy (computed)', fontsize=4.8, color='0.45', ha='center')
    save(fig, 'fig1_concept')


def toc_jpcc():
    """ACS TOC graphic (3.25 x 1.75 in): rendered silicon-vacancy complexes ordered by the conditions
    under which each is stable (V-O -> V-OH -> VH4 with decreasing pH / cathodic bias), next to a
    miniature of the computed map. Not a copy of any manuscript figure."""
    from hydration_engine import stability_map
    fig = plt.figure(figsize=(3.25, 1.75))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis('off'); bg.set_xlim(0, 3.25); bg.set_ylim(0, 1.75)
    bg.text(0.08, 1.62, 'Silicon vacancy in water', fontsize=7.2, fontweight='bold', color='0.15', va='center')
    items = [('VO', '$V$–O', 'near neutral pH'), ('VOH', '$V$–OH', 'acidic'), ('VH4', '$V$H$_4$', 'cathodic bias')]
    for k, (cfg, lab, cond) in enumerate(items):
        x0 = 0.06 + k * 0.66
        ai = fig.add_axes([x0 / 3.25, 0.50 / 1.75, 0.60 / 3.25, 0.82 / 1.75]); ai.axis('off'); ai.imshow(_crop(f'render/complex_{cfg}.png'))
        bg.text(x0 + 0.30, 0.46, lab, fontsize=6.6, ha='center', va='center', color=SPECIES_COLOR[cfg] if cfg != 'VOH' else '#9a8500', fontweight='bold')
        bg.text(x0 + 0.30, 0.33, cond, fontsize=5.2, ha='center', va='center', color='0.35')
    bg.add_patch(FancyArrowPatch((0.12, 0.17), (1.98, 0.17), arrowstyle='-|>', mutation_scale=7, lw=0.9, color='0.4'))
    bg.text(1.05, 0.06, 'more H: lower pH, cathodic $U$', fontsize=5.0, ha='center', va='center', color='0.35')
    sp = A['si']['species']; gap = A['si']['host']['gap']
    S = [Species(d['cfg'], d['q'], d['mH'], d['mO'], d.get('Ef_G', d['Ef_ref'])) for d in sp.values()]
    am = fig.add_axes([2.28 / 3.25, 0.30 / 1.75, 0.86 / 3.25, 1.18 / 1.75])
    mlo, mhi = -1.23, 0.30
    EF = np.linspace(0, gap, 100); MU = np.linspace(mlo, mhi, 220); idx, _ = stability_map(S, EF, MU)
    pres = sorted(set(idx.ravel().tolist())); remap = {i: k for k, i in enumerate(pres)}
    am.pcolormesh(EF, MU, np.vectorize(remap.get)(idx), cmap=ListedColormap([SPECIES_COLOR[S[i].name] for i in pres]),
                  vmin=-0.5, vmax=len(pres) - 0.5, shading='auto', rasterized=True)
    am.axhspan(0, mhi, facecolor='white', alpha=0.55, edgecolor='0.45', hatch='////', lw=0)
    am.set_xticks([]); am.set_yticks([]); am.set_xlim(0, gap); am.set_ylim(mlo, mhi)
    am.set_xlabel('$E_F$', fontsize=6, labelpad=1); am.set_ylabel('$\\Delta\\mu_{\\mathrm{H}}$', fontsize=6, labelpad=1)
    for sp_ in am.spines.values(): sp_.set_linewidth(0.5)
    am.set_title('stability map', fontsize=5.8, pad=2, color='0.25')
    fig.savefig(f'{OUT}/toc_jpcc.pdf'); fig.savefig(f'{OUT}/toc_jpcc.png', dpi=600)
    plt.close(fig); print('toc_jpcc done')


# =====================================================================
def fig2():
    si = A['si']; H = si['host']; sp = si['species']; gap = H['gap']
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.9), gridspec_kw={'width_ratios': [1.1, 1.25, 1]})
    # (a) bulk benchmark
    ax = axes[0]
    B = A['bulk']; E = EXP['bulk']
    comp = {'Si': [B['si']['a0'], B['si']['B_GPa'], H['gap']],
            '3C-SiC': [B['sic']['a0'], B['sic']['B_GPa'], A['sic']['Si-rich']['host']['gap']]}
    exp = {'Si': [E['Si']['a_Ang'], E['Si']['B_GPa'], E['Si']['Eg_eV']],
           '3C-SiC': [E['3C-SiC']['a_Ang'], E['3C-SiC']['B_GPa'], E['3C-SiC']['Eg_eV']]}
    xr = np.arange(3) * 1.15; w = 0.46
    for k, (name, col) in enumerate([('Si', BLU), ('3C-SiC', VER)]):
        r = [c / e for c, e in zip(comp[name], exp[name])]
        ax.bar(xr + (k - 0.5) * w, r, w * 0.92, color=col, label=name)
        for x, v in zip(xr + (k - 0.5) * w, r):
            ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=7, color='0.25')
    ax.axhline(1, color='0.3', lw=0.7, ls='--')
    ax.set_xticks(xr); ax.set_xticklabels([r'$a_0$', r'$B_0$', r'$E_g$'])
    ax.set_ylabel('PBE / experiment'); ax.set_ylim(0, 1.45)
    ax.legend(frameon=False, ncol=2, loc='upper right', handlelength=1.0, columnspacing=0.8)
    panel(ax, 'a')
    # (b) charged vacancy formation energy
    ax = axes[1]
    charges = [q for q in [1, 0, -1, -2] if f'V_q{q}' in sp]
    EF = np.linspace(0, gap, 300)
    lines = {q: sp[f'V_q{q}']['Ef_ref'] + q * EF for q in charges}
    for q in charges:
        ax.plot(EF, lines[q], ls=QSTYLE[q], color=BLU, lw=1.1, label=label_of('V', q))
    env = np.min(np.array([lines[q] for q in charges]), axis=0)
    ax.plot(EF, env, color=BLU, lw=2.2, alpha=0.9)
    tl = si['transition_levels'].get('V', {})
    for k, v in tl.items():
        if k.startswith('(') and 0 < v < gap and k != '(+1/-1)':
            ax.plot(v, np.min([lines[q][np.argmin(abs(EF - v))] for q in charges]), 'o', ms=4.5, mfc='white', mec='k', mew=0.8, zorder=5)
            kk = k.replace('+0', '0').replace('+1', '+').replace('-1', '$-$').replace('-2', '2$-$')
            ax.annotate(f"$\\varepsilon${kk}\n{v:.2f} eV", (v, np.min([lines[q][np.argmin(abs(EF - v))] for q in charges])),
                        xytext=(6, 6), textcoords='offset points', ha='left', va='bottom', fontsize=fs(6), color='0.25')
    ax.axvspan(0, gap, color='#f2f2f2', zorder=0)
    ax.set_xlabel(r'$E_F - E_{\mathrm{VBM}}$ (eV)'); ax.set_ylabel(r'$E^{f}$ (eV)')
    ax.set_xlim(0, gap); ax.legend(frameon=False, ncol=2, loc='upper right', columnspacing=1.0, handlelength=1.6)
    ax.set_title('Si vacancy, MP + alignment'); panel(ax, 'b')
    # (c) Jahn-Teller geometry: nn pair distances per charge state
    ax = axes[2]
    xs, lab = [], []
    for i, q in enumerate(charges):
        d = sp[f'V_q{q}']['nn_pair_distances']
        if not d: continue
        ax.plot([i] * len(d), d, 'o', ms=4, color=BLU, alpha=0.7, mec='white', mew=0.4)
        xs.append(i); lab.append(label_of('V', q))
    ideal = H['a'] / np.sqrt(2)
    ax.axhline(ideal, color='0.5', lw=0.7, ls='--')
    ax.text(len(charges) - 0.55, ideal + 0.02, 'ideal', fontsize=fs(6), color='0.4', ha='right')
    ax.set_xticks(xs); ax.set_xticklabels(lab)
    ax.set_ylabel('Si–Si pair distance (Å)')
    ax.set_title('Vacancy relaxation'); panel(ax, 'c')
    fig.tight_layout(w_pad=2.0); save(fig, 'fig2_validation')


# =====================================================================
def fig3():
    si = A['si']; sp = si['species']; gap = si['host']['gap']
    vib = A['lvm']; have_vib = all(vib[c] and vib[c].get('stretch_ir') for c in ['VH', 'VH2', 'VH3', 'VH4'])
    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.6)); axes = axes.ravel()
    order = ['V', 'VH', 'VH2', 'VH3', 'VH4']; m = [0, 1, 2, 3, 4]
    # (a) formation energy ladder (neutral)
    ax = axes[0]
    Ef = [sp[f'{o}_q0']['Ef_ref'] for o in order]
    ax.plot(m, Ef, '-o', color=BLU, ms=4.5, mfc='white', lw=1.3)
    for x, y in zip(m, Ef):
        ax.text(x + 0.12, y + 0.06, f'{y:.2f}', ha='left', va='bottom', fontsize=fs(6), color='0.25')
    ax.set_xlabel('hydrogen count $m$'); ax.set_ylabel(r'$E^{f}$ at $\Delta\mu_{\mathrm{H}}=0$ (eV)'); ax.set_xticks(m)
    ax.set_ylim(min(Ef) - 0.3, max(Ef) + 0.45); ax.set_xlim(-0.3, 4.6); panel(ax, 'a', dx=-0.15)
    # (b) sequential binding energy
    ax = axes[1]
    bind = si['binding_neutral']; bv = [bind[o] for o in order[1:]]
    cols = [GRN if v > 0 else VER for v in bv]
    ax.bar([1, 2, 3, 4], bv, width=0.6, color=cols)
    for x, v in zip([1, 2, 3, 4], bv):
        ax.text(x, v + (0.04 if v >= 0 else -0.10), f'{v:+.2f}', ha='center', fontsize=fs(6), color='0.25')
    ax.axhline(0, color='0.4', lw=0.6); ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel('$m$ (VH$_{m-1}$ + ½H$_2$ $\\rightarrow$ VH$_m$)'); ax.set_ylabel(r'$E_{\mathrm{bind}}$ (eV)')
    ax.set_ylim(min(bv + [0]) - 0.25, max(bv) + 0.3); panel(ax, 'b', dx=-0.15)
    # (c) LVM parity plot
    ax = axes[2]
    if have_vib:
        allx, ally = [], []
        mk = {'VH': 'o', 'VH2': 's', 'VH3': '^', 'VH4': 'D'}
        for c in ['VH', 'VH2', 'VH3', 'VH4']:
            calc = sorted(vib[c]['stretch_ir']); ex = sorted(EXP['Si_VHn_LVM_cm'][c])
            # collapse degenerate modes, then pair by rank; otherwise compare means
            uniq = []
            for x in calc:
                if not uniq or abs(x - uniq[-1]) > 2.0:
                    uniq.append(x)
            if len(uniq) == len(ex):
                pairs = list(zip(ex, uniq))
            else:
                pairs = [(np.mean(ex), np.mean(calc))]
            for e, cc in pairs:
                ax.plot(e, cc, mk[c], color=SPECIES_COLOR[c], ms=5, mec='white', mew=0.5, label=name_of(c) if (e, cc) == pairs[0] else None)
                allx.append(e); ally.append(cc)
        lo, hi = min(allx + ally) - 60, max(allx + ally) + 60
        ax.plot([lo, hi], [lo, hi], color='0.5', lw=0.7, ls='--')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_xlabel(r'experiment (cm$^{-1}$)'); ax.set_ylabel(r'PBE harmonic (cm$^{-1}$)')
        ax.legend(frameon=False, loc='lower right', handletextpad=0.3, borderpad=0.2, labelspacing=0.25)
        ax.set_title('Si–H stretch LVMs')
    else:
        ax.text(0.5, 0.5, 'LVM pending', transform=ax.transAxes, ha='center')
    panel(ax, 'c', dx=-0.15)
    # (d) spin-resolved KS states near the gap at the first k-point of the shifted 2x2x2 mesh
    # ('eigs_gamma' is that k-point, not Gamma).  Oct 2026: defect levels are aligned with the
    # potential-alignment shift and shown against the pristine band edges at the same k.
    ax = axes[3]; cells = ['V', 'VH', 'VH2', 'VH3', 'VH4']
    vbm = si['host']['vbm']
    pk = json.load(open('res64_si_perfect_q0.json')); ek = np.array(pk['eigs_gamma']['0'])
    ev_k = ek[ek < pk['fermi']].max() - vbm; ec_k = ek[ek > pk['fermi']].min() - vbm
    for c, cfg in enumerate(cells):
        d = sp[f'{cfg}_q0']; eg = d.get('eigs_gamma')
        if not eg: continue
        for s, (dx, col) in enumerate([(-0.28, BLU), (0.02, VER)]):
            e = np.array(eg[str(s)]) - vbm - (d.get('dV_align') or 0.0)
            e = e[(e > ev_k - 0.7) & (e < ec_k + 0.3)]
            ax.hlines(e, c + dx, c + dx + 0.26, color=col, lw=1.0, alpha=0.9)
        ax.text(c, ec_k + 0.45, name_of(cfg), ha='center', fontsize=fs(6), fontweight='bold')
    ax.axhspan(ev_k, ec_k, color='#eef4fb', zorder=0)
    ax.axhline(ev_k, color='0.3', lw=0.5); ax.axhline(ec_k, color='0.3', lw=0.5)
    ax.text(len(cells) - 0.35, ec_k, '$E_c(k)$', fontsize=fs(5.5), ha='left', va='center', color='0.3', clip_on=False)
    ax.text(len(cells) - 0.35, ev_k, '$E_v(k)$', fontsize=fs(5.5), ha='left', va='center', color='0.3', clip_on=False)
    ax.set_ylim(ev_k - 0.7, ec_k + 0.65); ax.set_xlim(-0.6, len(cells) - 0.4); ax.set_xticks([])
    ax.set_ylabel(r'$\epsilon_{\mathrm{KS}} - E_{\mathrm{VBM}}$ (eV)')
    ax.legend(handles=[Line2D([0], [0], color=BLU, lw=1.2, label='spin up'), Line2D([0], [0], color=VER, lw=1.2, label='spin down')],
              frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, handlelength=1.2)
    ax.set_title('Gap states'); panel(ax, 'd', dx=-0.15)
    fig.tight_layout(w_pad=2.0, h_pad=1.5); save(fig, 'fig3_hydrogen')


# =====================================================================
def si_species(include_O=True):
    sp = A['si']['species']
    # Oct 2026: vibrational free energies (Ef_G, local H/O modes + gas-phase references) when available
    return [Species(d['cfg'], d['q'], d['mH'], d['mO'], d.get('Ef_G', d['Ef_ref'])) for d in sp.values() if include_O or d['mO'] == 0]


E_O2_H2O = 1.23   # eV per H: O2/H2O equilibrium on the dmu_H scale (CHE, pH-independent here)


def draw_map(ax, species, gap, dmu_range=(-1.4, 1.0), n=400, legend=True, T=300.0, ef_max=None, water_window=False):
    """ef_max: extend the Fermi-level axis beyond the PBE gap (e.g. to the experimental
    gap); the PBE conduction-band edge is then marked by a vertical line.
    water_window (Oct 2026): dim the regions where liquid water is not stable (above the
    H2/H2O line, dmu_H > 0, and below the O2/H2O line, dmu_H < -1.23 eV) and mark both lines."""
    ef_top = ef_max or gap
    EF = np.linspace(0, ef_top, n); DMU = np.linspace(*dmu_range, n)
    idx, om = stability_map(species, EF, DMU)
    present = sorted(set(idx.ravel().tolist()))
    def col(s):
        base = SPECIES_COLOR[s.name]
        # lighten/darken by charge for distinguishability
        rgb = np.array(mpl.colors.to_rgb(base))
        f = {2: 0.45, 1: 0.7, 0: 1.0, -1: 1.25, -2: 1.5}[s.q]
        if f <= 1: rgb = 1 - (1 - rgb) * f          # lighter for positive
        else: rgb = rgb / f                         # darker for negative
        return np.clip(rgb, 0, 1)
    remap = {i: k for k, i in enumerate(present)}
    Z = np.vectorize(remap.get)(idx)
    ax.pcolormesh(EF, DMU, Z, cmap=ListedColormap([col(species[i]) for i in present]),
                  vmin=-0.5, vmax=len(present) - 0.5, shading='auto', rasterized=True)
    # region labels at centroid
    for i in present:
        mask = idx == i
        if mask.sum() < (0.03 if water_window else 0.01) * mask.size: continue   # thin V+ strip: named in the caption
        lcol, veiled = 'white', False
        if water_window:   # label inside the water window when the region reaches it
            inwin = mask & ((DMU > -E_O2_H2O + 0.05) & (DMU < -0.05))[:, None]
            if inwin.sum() > 0.01 * mask.size:
                mask = inwin
            else:
                veiled = True; lcol = '0.1'
        yy, xx = np.where(mask)
        xc = np.clip(EF[int(np.median(xx))], ef_top * 0.08, ef_top * 0.92)
        yc = np.clip(DMU[int(np.median(yy))], dmu_range[0] + 0.08, dmu_range[1] - 0.08)
        if veiled and DMU[int(np.median(yy))] > 0:      # region only above the H2 line: between the two labels
            xc, yc = ef_top * 0.22, 0.47 * dmu_range[1]
        ax.text(xc, yc, label_of(species[i].name, species[i].q),
                ha='center', va='center', fontsize=fs(6.5), color=lcol, fontweight='bold', zorder=5)
    ax.axhline(0, color='0.15', lw=0.7, ls='--', zorder=3)
    if water_window:
        # outside the stability window of liquid water: hatched veil + label
        top, bot = dmu_range[1], dmu_range[0]
        if top > 0:
            ax.axhspan(0, top, facecolor='white', alpha=0.55, edgecolor='0.35', hatch='////', lw=0, zorder=2)
            ax.text(ef_top * 0.02, top - 0.03, 'water reduced (H$_2$ evolves)', fontsize=fs(5.2), color='0.15',
                    ha='left', va='top', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        if bot < -E_O2_H2O:
            ax.axhspan(bot, -E_O2_H2O, facecolor='white', alpha=0.55, edgecolor='0.35', hatch='////', lw=0, zorder=2)
            ax.axhline(-E_O2_H2O, color='0.15', lw=0.7, ls='--', zorder=3)
            ax.text(ef_top * 0.98, -E_O2_H2O + 0.03, 'O$_2$/H$_2$O', fontsize=fs(5.2), color='0.15', ha='right', va='bottom', zorder=4)
        ax.text(ef_top * 0.98, 0.03, 'H$_2$/H$_2$O (pH 0, $U$ = 0)', fontsize=fs(5.2), color='0.15', ha='right', va='bottom', zorder=4,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
    else:
        ax.text(ef_top * 0.98, 0.03, '½H$_2$(g), 1 bar', fontsize=fs(5.5), color='0.15', ha='right')
    # pH reference lines at U = 0 V vs SHE (pH 0 coincides with the H2 line and is labelled there)
    for pH in ((7, 14) if water_window else (0, 7, 14)):
        y = dmu_H_from_pH_U(pH, 0.0, T)
        if dmu_range[0] < y < dmu_range[1]:
            ax.axhline(y, color='0.15', lw=0.5, ls=':', zorder=3)
            ax.text(ef_top * 0.02, y + 0.02, f'pH {pH}, $U$ = 0', fontsize=fs(5.2), color='0.15', ha='left', zorder=4)
    if ef_max and ef_max > gap:
        ax.axvline(gap, color='0.15', lw=0.8, ls='-')
        ax.text(gap + 0.01, dmu_range[1] - 0.08, 'PBE CBM', fontsize=fs(5.5), color='0.15', ha='left', va='top', rotation=90)
    ax.set_xlabel(r'$E_F - E_{\mathrm{VBM}}$ (eV)'); ax.set_ylabel(r'$\Delta\mu_{\mathrm{H}}$ (eV)')
    ax.set_xlim(0, ef_top); ax.set_ylim(*dmu_range)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)


def fig4():
    si = A['si']; gap = si['host']['gap']
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.9), gridspec_kw={'width_ratios': [0.9, 1.1, 1.1]})
    # (a) first-H binding Si vs SiC
    ax = axes[0]
    b = A['sic']['binding_neutral']
    vals = [si['binding_neutral']['VH'], b.get('VC', np.nan), b.get('VSi', np.nan)]
    cols = [BLU, VER, GRN]
    labs = ['Si\n$V$\n(Si–H)', 'SiC\n$V_{\\mathrm{C}}$\n(Si–H–Si)', 'SiC\n$V_{\\mathrm{Si}}$\n(C–H)']
    ax.bar(range(3), vals, color=cols, width=0.62)
    ax.axhline(0, color='0.4', lw=0.7)
    for i, v in enumerate(vals):
        if not np.isnan(v): ax.text(i, v + (0.05 if v >= 0 else -0.16), f'{v:+.2f}'.replace('-', '−'), ha='center', fontsize=fs(6.5), color='0.25')
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=fs(6.5), linespacing=1.1); ax.set_ylabel(r'$E_{\mathrm{bind}}$(H) vs ½H$_2$ (eV)')
    ax.set_ylim(min(np.nanmin(vals), 0) - 0.4, np.nanmax(vals) + 0.4); panel(ax, 'a')
    # (b) hydrogen-only map, (c) with O/OH
    draw_map(axes[1], si_species(include_O=False), gap, dmu_range=(-1.45, 0.45), water_window=True); axes[1].set_title('H only'); panel(axes[1], 'b')
    draw_map(axes[2], si_species(include_O=True), gap, dmu_range=(-1.45, 0.45), water_window=True); axes[2].set_title('H, O, OH (water)'); panel(axes[2], 'c')
    fig.tight_layout(w_pad=1.4); save(fig, 'fig4_model')


def fig4_extgap():
    """Variant of Fig. 4 with E_F extended to the experimental gap of Si (1.17 eV)."""
    si = A['si']; gap = si['host']['gap']; egap = EXP['bulk']['Si']['Eg_eV']
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.9))
    draw_map(axes[0], si_species(include_O=False), gap, ef_max=egap); axes[0].set_title('H only'); panel(axes[0], 'a')
    draw_map(axes[1], si_species(include_O=True), gap, ef_max=egap); axes[1].set_title('H, O, OH (water)'); panel(axes[1], 'b')
    fig.tight_layout(w_pad=1.4); save(fig, 'fig4_model_extgap')


def figS1():
    """SiC: (a) carbon vacancy with its charge states and V_C-H, (b) silicon vacancy and V_Si-H.
    Both are drawn at the Si-rich limit; because every species on a panel removes the same
    host atom, the Si-rich/C-rich choice shifts all of them equally and moves no boundary."""
    sp = A['sic']['Si-rich']['species']; gap = A['sic']['Si-rich']['host']['gap']
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.9))
    for ax, cfgs, title, lab in ((axes[0], ('VC', 'VCH'), '3C-SiC carbon vacancy', 'a'),
                                 (axes[1], ('VSi', 'VSiH'), '3C-SiC silicon vacancy', 'b')):
        species = [Species(d['cfg'], d['q'], d['mH'], d['mO'], d['Ef_ref']) for d in sp.values() if d['cfg'] in cfgs]
        draw_map(ax, species, gap, dmu_range=(-2.2, 1.0)); ax.set_title(title); panel(ax, lab)
    fig.tight_layout(w_pad=1.4); save(fig, 'figS1_sic_map')


def toc():
    si = A['si']; gap = si['host']['gap']
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    draw_map(ax, si_species(include_O=True), gap, dmu_range=(-1.45, 0.45), n=250, water_window=True)
    ax.set_title('Silicon vacancy in contact with water', fontsize=fs(7.5))
    fig.tight_layout(); save(fig, 'toc_graphic')


if __name__ == '__main__':
    for f in (fig1, fig2_structures, fig2, fig3, fig4, fig4_extgap, figS1, toc):
        try:
            f()
        except Exception as e:
            print(f'{f.__name__} skipped: {e!r}')
