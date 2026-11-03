# Structure renders for Fig. 1 (panels a, b)

Ray-traced with OVITO (Tachyon renderer) from the relaxed structures in the project root
(`final64_si_<cfg>_q0.xyz`, `res64_si_<cfg>_q0.json`). Cite: A. Stukowski, Modelling Simul. Mater. Sci. Eng. 18, 015012 (2010).

    python -m venv ~/ovito_venv && ~/ovito_venv/bin/pip install ovito ase    # OVITO 3.17, ase 3.29 used (~750 MB)
    ~/ovito_venv/bin/python render/render_complexes.py    # -> render/complex_<cfg>.png  (Fig. 1b)
    ~/ovito_venv/bin/python render/render_slab.py         # -> render/slab.png, slab_mask.png (Fig. 1a)
    /opt/anaconda3/envs/gpaw-run/bin/python make_figs_jpcc.py   # composes fig1_concept from the PNGs

The water layer in `slab.png` is illustrative (random, non-overlapping H2O placed above the cell); it is not a computed interface.
`slab_mask.png` is a flat-colored render with the same camera, used only to locate the vacancy for the arrows.
The previous, purely schematic version of Fig. 1 is kept as `fig1_schematic()` in make_figs_jpcc.py.
