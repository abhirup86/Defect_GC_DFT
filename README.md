# Charge state and hydration of vacancies in Si and SiC in contact with water

Code and data for

> A. Patra, *Charge state and hydration of vacancies in silicon and silicon carbide in contact with water: A grand-canonical first-principles stability map*, submitted to Computational Materials Science (2026).

The paper minimizes the grand potential of a vacancy complex over its charge state and its hydrogen/oxygen content, with the computational hydrogen electrode tying the chemical potentials to pH and electrode potential. All DFT was done with GPAW (PBE, plane waves, spin-polarized) in 64-atom cells.

This repository contains every DFT result the paper uses, the relaxed structures, the raw GPAW output, and the scripts that run the calculations and turn them into the numbers, tables and figures.

## Quick start: regenerate the paper's numbers and figures (no DFT)

```bash
conda env create -f environment.yml && conda activate gpaw-run
bash run_analysis.sh
```

This takes about a minute and writes to `work/paper/`:

- `numbers.tex`: every number quoted in the manuscript, as LaTeX macros
- `si_table_*.tex`: the Supplementary Material tables
- `figs/`: Figs. 1–5, Fig. S1 and the graphical abstract (PDF + PNG)

`numbers.tex` and the SI tables come out byte-identical to the submitted manuscript. Figures use the Aptos font if it is found (set `APTOS_DIR`) and the matplotlib default otherwise.

## Layout

```
code/
  run64.py          DFT driver: relaxation of a complex in a 64-atom cell (neutral and charged, variants)
  prerelax64.py     MACE-MP-0 pre-relaxation of starting geometries
  bench_bulk.py     bulk equations of state (Si, 3C-SiC)
  sp700.py          700 eV single points (cutoff correction for neutral formation energies)
  molbox.py         H2 / H2O reference energies vs box size and cutoff
  run_vib64.py      local vibrational modes (partial Hessian) of the Si vacancy complexes
  mol_vib.py        H2 / H2O vibrational frequencies and thermochemistry
  cutoff_check.py   prints the 340 vs 700 eV comparison
  assemble64.py     aggregates all results: formation energies, corrections, levels, vibrational free energies
  hydration_engine.py  grand-potential minimization, transition levels, CHE pH/U conversion
  make_figs_jpcc.py all figures
  render/           OVITO scripts (and their PNG output) for the structure renders in Figs. 1-2
  launch64.sh, launch_oct2026.sh, reproduce.sh   job launchers for the full campaign
data/
  results_json/     one JSON per calculation (energy, forces, magnetic moment, site potentials, eigenvalues),
                    vibrational results, bulk EOS, experimental reference values (exp_data.json),
                    and the aggregated assembled64.json / species_Si_64.csv
  structures/       relaxed structures, final64_<host>_<complex>_q<charge>[_variant].xyz
  gpaw_logs/        raw GPAW text output for every calculation
```

File names: `res64_<host>_<complex>_q<charge>[_variant].json` for the 64-atom relaxations at 340 eV; `res700_*` for the 700 eV single points; `resbox_*` for the molecular references; `vibres64_*` for the local modes. Variants: `_sym` symmetric start, `_hs` high-spin seed, `_ideal` ideal Si–H start, `_fsm` fixed spin moment, `_sharp` 0.01 eV smearing. `assemble64.py` keeps the lowest-energy converged variant and tabulates the rest.

## Computational protocol

- GPAW 26.7.0 / ASE 3.29, PBE, plane waves 340 eV, Fermi–Dirac smearing 0.05 eV, spin-polarized throughout.
- 64-atom cubic cells (2×2×2 conventional) at the PBE lattice constant; shifted 2×2×2 Monkhorst–Pack mesh; BFGS to 0.03 eV/Å.
- Symmetry broken on purpose: 0.05 Å rattle, a 0.15 Å pairing kick of two dangling-bond atoms, and seeded magnetic moments. Charged states start from the relaxed neutral cell.
- Neutral formation energies are corrected to 700 eV with single points. The O PAW set is not converged at 340 eV; H-only cells change by ≤ 0.04 eV.
- Band edges from the pristine cell on a Γ-centred 4×4×4 mesh.
- Charged cells: Makov–Payne monopole correction plus potential alignment (far-atom average of the site-potential difference).
- Vibrational free energies of bound H and O from local partial Hessians; H2 and H2O with standard 298 K enthalpy and entropy.

## Full reproduction (DFT)

The DFT scripts run in a flat working directory with all scripts and results side by side:

```bash
mkdir run && cp code/*.py code/*.sh run/ && cd run
export GPAW_PYTHON=$(which python) MACE_PYTHON=/path/to/mace/env/python
bash reproduce.sh      # resumable: skips any step whose output exists
```

Jobs run serially (one core each, `OMP_NUM_THREADS=1`) and in parallel across configurations. On an 8-core Apple M1 Pro with 6 concurrent jobs each cell took 3–28 h. The main campaign (33 cells plus 4 sets of local modes) took about 4 days; `launch_oct2026.sh` runs the later vibrational and variant calculations.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`data/LICENSE.md`).
