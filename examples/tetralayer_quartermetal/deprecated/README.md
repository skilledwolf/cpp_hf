# Deprecated quartermetal scripts

These scripts were exploratory diagnostics or earlier production drivers
written before the current conventions stabilized. They predate one or
more of the following corrections, so their saved figures should not be
trusted:

* **C3 projector fix** — earlier C3 implementations gave PM and PM+C3
  energies 28+ meV apart on cells where they should be degenerate.
  The current `_quartermetal_common.make_c3_project_fn_numpy` averages
  only complete in-patch C3 orbits.
* **PM_C3 self-consistent CN reference subtraction** — earlier scripts
  used `u0` (non-interacting CN at U=0) or `sameU` (non-interacting CN
  at running U) for the kernel `reference_density`. The production
  recipe now uses `qm.pm_c3_cn_reference_density(...)`, which is the
  PM_C3-projected self-consistent HF density at charge neutrality at
  the running D.
* **ε consistency between Hartree and Fock** — older runs used different
  effective dielectric constants for the Hartree and exchange terms
  (~5 vs 13.8). The production scripts use `dualgate_eps2` (or other
  matched-ε kinds) for both.
* **Physical Fock for observables** — the kernel computes the Fock
  matrix in the reference-subtracted convention; physical Δ_tb and DOS
  require adding `Σ_x[refP] + diag(HH @ diag(refP))`. Earlier scripts
  reported the bare reference-subtracted Δ_tb / DOS, which gave
  anti-screened layer potentials and discretization-dependent DOS.
* **Tetrahedron DOS via bztetra.twod** — earlier scripts used Lorentzian
  or Fermi smearing for DOS at E_F, which is sensitive to the alignment
  of k-points with the Fermi surface. The production grids use
  `bztetra.twod.density_of_states_weights` with the canonical
  `area_prefactor = |det(B)| / (2π)²` normalization.

## Replacement map

| Deprecated | Current replacement |
|---|---|
| `quartermetal_grid.py` | `quartermetal_grid_pm_c3.py`, `quartermetal_grid_svp_c3.py` |
| `quartermetal_qm_phase_scan.py` | `quartermetal_grid_pm_c3.py` + `quartermetal_grid_svp_c3.py` |
| `quartermetal_convergence_trace.py`, `convergence_trace2.py` | `quartermetal_D_scan.py` |
| `quartermetal_long_iter_check.py`, `eps2_ref_trace.py` | `quartermetal_D_scan.py` |
| `quartermetal_nk_scan.py`, `nk96_bands.py`, `T_nk_scan.py` | `quartermetal_nk_convergence.py` |
| `quartermetal_4configs_bands.py`, `single_cell.py`, `n05_D08_compare.py`, `trace_D10.py`, `inspect.py`, `kmax_check.py` | `quartermetal_D_scan.py` |
| `quartermetal_D_scan_sameU.py` | duplicate of `quartermetal_D_scan.py` |
| `quartermetal_hartree_validation.py`, `hartree_vs_fock.py` | covered by `quartermetal_grid_pm_c3.py` Δ_tb diagnostics |

If you need to revive any of these, port them onto the current
`_quartermetal_common.pm_c3_cn_reference_density` reference and the
physical Fock reconstruction (Σ_x[refP] + Hartree[refP]).
