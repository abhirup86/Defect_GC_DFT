"""
hydration_engine.py
===================
Grand-canonical hydration model for charged point defects (see hydration_model.md).

Consumes a table of DFT-derived *reference formation energies* for defect species
and produces:
  (1) the 2-D defect "Pourbaix" stability map  Omega*(E_F, dmu_H),
  (2) hydration-dependent charge-transition levels  eps(q/q'),
  (3) H/OH binding energies per charge state.

------------------------------------------------------------------------------
THERMODYNAMICS (consistent with hydration_model.md, section 3)
------------------------------------------------------------------------------
Grand potential of species X (vacancy + m_H hydrogens + m_O oxygens) in charge q:

    Omega(E_F, dmu_H) = Ef_ref                       # DFT formation energy at the
                                                     #   reservoir reference point
                        - m_H * dmu_H                # H reservoir (CHE)
                        + 2 * m_O * dmu_H            # O slaved to water: mu_O = mu_H2O - 2 mu_H
                        + q * E_F                    # electron reservoir (E_F above VBM)

Here `Ef_ref` is the fully-corrected formation energy the user computes from DFT
at the chosen reference (mu_H = 1/2 E[H2], mu_O = mu_H2O - E[H2], E_F = VBM):

    Ef_ref = E_tot[X^q] - E_tot[bulk_Nref]
             + sum_host n_removed * mu_host          # +1*mu_Si for a Si monovacancy
             - m_H * (1/2 E[H2]) - m_O * mu_O^ref
             + q * E_VBM + E_corr(q) + F_vib(T)

`dmu_H` (eV, the y-axis) is the shift of the H chemical potential above 1/2 E[H2];
convert to physical (pH, U, T) with `dmu_H_from_pH_U()`.  Raising dmu_H  = more
proton-rich (lower pH / cathodic U / higher RH); it simultaneously lowers mu_O
(H-rich => O-poor), which is why the O term enters with +2*m_O.
------------------------------------------------------------------------------
"""
from __future__ import annotations
from dataclasses import dataclass
import csv
import numpy as np

KB = 8.617333262e-5  # eV/K


@dataclass
class Species:
    name: str          # e.g. "VH2"
    q: int             # charge state
    mH: int            # number of H in the complex
    mO: int            # number of O in the complex
    Ef_ref: float      # DFT reference formation energy (eV), fully corrected

    def omega(self, EF, dmuH):
        """Grand potential at Fermi level EF (above VBM) and H potential shift dmuH."""
        return self.Ef_ref - self.mH * dmuH + 2.0 * self.mO * dmuH + self.q * EF


# ---------------------------------------------------------------------------
# Reservoir helpers
# ---------------------------------------------------------------------------
def dmu_H_from_pH_U(pH, U, T=300.0):
    """Computational hydrogen electrode: dmu_H = -eU - 2.303 kB T pH  (eV).

    U in volts vs SHE; pH dimensionless; T in K.  (Thermal 1/2 dG_H2 term, if
    desired, folds into the caller's reference and is omitted here.)
    """
    return -U - np.log(10.0) * KB * T * pH


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_species(csv_path):
    """Load species from a CSV with header: name,q,mH,mO,Ef_ref  (see template)."""
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row.get("name") or row["name"].lstrip().startswith("#"):
                continue
            out.append(Species(row["name"].strip(), int(row["q"]),
                               int(row["mH"]), int(row["mO"]), float(row["Ef_ref"])))
    return out


# ---------------------------------------------------------------------------
# Core analyses
# ---------------------------------------------------------------------------
def stability_map(species, EF_grid, dmuH_grid):
    """Return (index_grid, omega_grid).

    index_grid[i,j] = position in `species` of the minimum-Omega species at
    (EF_grid[j], dmuH_grid[i]).  omega_grid holds that minimum value.
    """
    EF, MU = np.meshgrid(EF_grid, dmuH_grid)              # shape (nmu, nef)
    stack = np.stack([s.omega(EF, MU) for s in species])  # (nspecies, nmu, nef)
    idx = np.argmin(stack, axis=0)
    return idx, np.min(stack, axis=0)


def transition_levels(species):
    """Charge-transition levels eps(q/q') (eV above VBM) for each fixed decoration.

    Returns dict keyed by (name, mH, mO) -> list of ((q, q'), eps).
    eps is reservoir-independent because the -mH*dmuH + 2mO*dmuH terms cancel
    between two charge states of the *same* decoration.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for s in species:
        groups[(s.name, s.mH, s.mO)].append(s)
    result = {}
    for key, grp in groups.items():
        grp = sorted(grp, key=lambda s: s.q, reverse=True)  # +... -> -...
        levels = []
        for a, b in zip(grp[:-1], grp[1:]):                 # consecutive charges
            # Omega(a)=Omega(b): Ef_ref(a)+q_a EF = Ef_ref(b)+q_b EF
            eps = (b.Ef_ref - a.Ef_ref) / (a.q - b.q)
            levels.append(((a.q, b.q), eps))
        if levels:
            result[key] = levels
    return result


def binding_energies(species):
    """H binding energy for VH_m <- VH_{m-1} + 1/2 H2, per charge state (eV).

    Positive = H is bound (exothermic) at the reservoir reference (dmu_H = 0).
    Returns dict: (name_base, q) -> list of (m, E_bind).
    """
    from collections import defaultdict
    # group by (base vacancy label, q, mO) and vary mH
    # here we treat the leading letters up to first 'H' as the base for VH_n series
    def base(s):
        return s.name.split("H")[0] if "H" in s.name else s.name
    groups = defaultdict(dict)
    for s in species:
        if s.mO == 0:                       # pure hydrogenation series only
            groups[(base(s), s.q)][s.mH] = s.Ef_ref
    out = {}
    for key, bym in groups.items():
        ms = sorted(bym)
        ebs = []
        for m in ms:
            if (m - 1) in bym:
                # Omega(VH_m) - Omega(VH_{m-1}) at dmuH=0 = Ef_ref(m) - Ef_ref(m-1)
                # E_bind = -(that) = reservoir-referenced binding vs 1/2 H2
                ebs.append((m, bym[m - 1] - bym[m]))
        if ebs:
            out[key] = ebs
    return out
