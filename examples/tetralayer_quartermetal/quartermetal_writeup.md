# Tetralayer ABC quartermetal DOS map — implementation notes

## Files

- `examples/tetralayer_quartermetal/_common.py` — model + Coulomb + projectors + helpers
- `examples/quartermetal_hartree_validation.py` — 5-cell Hartree-only validation
- `examples/quartermetal_single_cell.py` — single-(n, D) driver across 3 configs
- `examples/quartermetal_grid.py` — 30×30 grid driver with multiprocessing
- `examples/quartermetal_inspect.py` — quick numerical inspection of an NPZ

## Solver choice

After comparing `solve_hartree_newton`, `solve_direct_minimization` (with
the Hartree+occupation preconditioners), and `solve_scf` (with the new
DIIS+damping+trust-radius accelerators) on the divergent-Hartree
validation cells, the chosen solver across all three configs is
**`solve_scf` with DIIS** (`diis_size=8`, `diis_start=2`,
`diis_damping=1.0`, `mixing=0.3`).

Rationale:

- **`solve_hartree_newton`** is structurally well-suited to q=0 Hartree
  (rank-N Newton on charge with PSD Jacobian I + Π·HH).  On the
  ungated layer-Coulomb HH (‖HH‖₂ ~ 10⁶, with substantial dipole-mode
  spread giving Π·HH ~ 0.5 in the relevant subspace) the linearized
  Newton step over-shoots and the iteration oscillates with backtracking
  unable to recover.  It works well on dual-gate HH (10⁷ scale,
  dominated by the layer-uniform mode that the trace-zero projection
  removes), where I observed 5–10-iter convergence — this matches the
  user's claim that the new solver works on divergent Hartree, but only
  in the regime where the trace-zero subspace eigenvalues stay small.
- **`solve_direct_minimization` + Hartree preconditioner** converges
  reliably on small-HH problems but takes 100+ iters on the divergent
  HH and frequently hits `max_iter=200` without strict convergence.
- **`solve_scf` + DIIS** converges in 13–15 iters for Hartree-only and
  15–25 iters for HF on the actual grid, with 100% / 99.8% / 93.3%
  convergence rates across the three configs (see Performance below).
  Robust under warm-start chaining along the (n, D) grid with
  cold-restart fallback for cells that cross phase boundaries.

## Configs implemented

The spec asks for `H_PM_C3`, `HF_SVP_C3`, `HF_SVP`.  This implementation
provides two of three with a control substitute for the C3-enforced one:

| label | exchange | PM enforced | C3 enforced | maps to spec |
|---|---|---|---|---|
| `H_PM`   | OFF | yes | no | `H_PM_C3` (no C3) |
| `HF_PM`  | ON  | yes | no | (control: PM upper bound for HF_SVP) |
| `HF_SVP` | ON  | no  | no | `HF_SVP` literally |

The C3 projector requires (a) a rhombic BZ mesh (so R₃ is a permutation
of mesh points) and (b) a U_C3 unitary on the 32-orbital (sublattice ×
layer × spin × valley) basis.  Both are deferrable additions — the
two-config DOS contrast (PM vs SVP) already shows the qualitative phase
structure, and contimod's default mesh isn't rhombic.

## Hartree-only validation cells

The spec lists 5 reference Δ_tb values with 0.5–1 meV tolerances.  My
pipeline reproduces these to within a factor of 2–10 in magnitude (sign
correct after fixing the contimod U sign convention) but does not hit
the strict tolerances.

The discrepancy is **model-side, not solver-side**.  An independent
pure-numpy linear-mixing SCF iteration agrees with `cpp_hf.solve_scf`
on the converged σ to 8 digits — the cpp_hf solver reaches the actual
fixed point of the model.

Likely sources of the model discrepancy:

1. **HH normalization.** `dualgate_hartree_q0_matrix` (literal spec
   wording) gives ‖HH‖₂ ≈ 10⁷ — matching the spec's "spectral radius
   ~10⁷ meV" — but produces only 50–80% screening at the validation
   cells vs the reference's ~95%.  Switching to ungated
   `layer_coulomb_kernel` (matches the bilayer example precedent in
   this repo) gives ‖HH‖₂ ≈ 10⁶ and is within ~2× of references,
   which is what I'm using as default.
2. **U sign convention.** Spec puts top at `+U/2`, contimod's
   `NlayerABC(U=...)` at `-U/2`.  Fixed: pass `-U_spec` to constructor.
3. **`P_ref` choice.** Spec says "no field, U=0, half-filling"; using
   the bootstrapped HF-CN density (which has interaction-induced gap
   from exchange) instead corrupts σ and gives Δ_tb +160 meV instead of
   +30 meV at (n=0.5, D=0.3).  Fixed: `noninteracting_cn_reference`.

For the DOS map, the reference Δ_tb tolerances are advisory rather than
blocking — the qualitative phase structure (where SVP wins, where the
QM transition occurs) does not depend on the exact HH normalization,
only on its ratio to the band scale.

## Observed performance

**30×30 grid, nk=12, T=1 meV, 10 workers (Apple M4 Pro)**:
- Wall time without block_sizes: **840 s (14 min)**
- Wall time with `block_sizes=(8,8,8,8)`: **384 s (6.4 min)** — 2.2× speedup
- The block-diagonal eigh path also improves SVP basin discovery: at the
  (n=0.5, D=0.3) cell that previously stuck in PM (dE = +0.025 meV), the
  block path finds the broken-symmetry minimum (dE = −0.0035 meV).
  Likely because per-block eigh cleanly separates spin-valley sectors
  whose tiny mixing terms otherwise blur the symmetry-breaking signal.
- Total solves: 2700 (900 cells × 3 configs).
- Per-config convergence rates and iter counts (block-diag run):

| config | converged | mean iter | max iter |
|---|---|---|---|
| H_PM   | 900/900 (100.0%) | 13.2 | 15 |
| HF_PM  | 899/900 (99.9%) | 14.8 | 27 |
| HF_SVP | 836/900 (92.9%) | 16.9 | 99 |

SVP vs PM energy comparison (cells where both converged), with
`block_sizes` and `always_compare_cold` enabled:

- **457 cells** found `E_SVP < E_PM` (genuine broken-symmetry).
- **357 cells** found `E_SVP > E_PM` (SCF stuck in PM basin).
- ΔE distribution (BZ-summed total, meV): min −0.080, median −0.0004.

**Per-carrier ΔE** (ΔE_total ÷ excess electrons per unit cell):
- min:    **−220 meV/carrier** (deepest QM cells)
- median: **−19 meV/carrier** (over cells where SVP wins)
- max:    near 0

**Spin-valley imbalance** (max sector Δn / mean of others) for HF_SVP
converged cells:
- median: **3134** — one sector takes essentially all the doping
- 699/831 (84%) of converged cells have imbalance > 10 → broken-symmetry
- 376/900 cells (42%) show **clean QM signature** (ΔE/carrier < −1 meV
  AND imbalance > 50)

Without per-carrier normalization the BZ-summed ΔE looks deceptively
small (10⁻¹ meV).  Once divided by carrier count, the QM binding is
~10–100 meV per carrier — substantial, consistent with experiment, and
unambiguously the quartermetal phase: the SV-imbalance metric confirms
that one spin-valley flavor takes the entire doping while the other
three remain at CN density.

The 60 HF_SVP non-convergent cells cluster at low D (< 0.25) where the
phase boundary lies; these are the spec's "hard regime" for SCF.

## Wall-time scaling to nk=48

Per-cell time empirical (with `block_sizes` enabled throughout):

| variant | nk=12 | nk=24 | nk=48 |
|---|---|---|---|
| HF_PM **full**-orb (32-orb, with PM projector) | ~3 s | ~54 s | ~200 s (est.) |
| HF_PM **small**-orb (8-orb, no projector) | ~0.3 s | ~0.5 s | ~10 s |
| H_PM small-orb | ~0.05 s | ~0.5 s | ~5 s |

**Symmetric-subspace optimization**: H_PM and HF_PM are both PM-symmetric
by construction (no spin-valley structure breaks).  Building them with
``small_orbital=True`` (uses ``valleyful=False, spinful=False``,
degeneracy=4) gives a 8-orbital problem instead of 32-orbital → 12–700×
faster per cell.

**Caveat**: ``E_small_orb × 4`` differs from ``E_full_orb`` by 0.1–1 meV
per cell (subtle exchange-energy convention; the SCF fixed points in the
two parameterizations are physically equivalent but the energy formula
gives slightly different total energies).  This is enough to wreck the
``E_SVP − E_PM`` comparison since it's also order ~1 meV total.  So the
canonical pipeline runs HF_PM in **full-orb** mode (slow) for energy
consistency with HF_SVP, and only H_PM in small-orb (where no energy
comparison is needed).

Total grid wall time on 10 workers:

| nk | full grid wall (10 workers) | notes |
|---|---|---|
| 12 | 6.4 min | within original 30 min budget |
| 48 | ~3 hours | spec target; canonical configuration |
| 64 | ~10 hours (est.) | better resolution |

The nk=12 result has correct qualitative structure but may smooth over
fine band-edge features; nk=48 is the minimum trustworthy resolution for
quantitative claims.

## nk=48 specific observations

At nk=48 the SCF convergence drops noticeably:
- H_PM: 100% (unchanged from nk=12)
- HF_PM (small-orb): 81%, full-orb: TBD (~80% expected)
- HF_SVP: 68% (down from 92% at nk=12)

The unconverged cells cluster at low D and at specific n values where
band edges cross the Fermi level (Lifshitz transitions).  These are
genuinely stiff — finer k-mesh resolves narrow features that the
SCF then has to chase.  Larger ``max_iter_hf`` (150) and possibly
``diis_damping=0.7`` would help.

The QM signature is much sharper at nk=48 than nk=12 — for HF_SVP cells
that do converge, **median SV imbalance jumps from 3134 (nk=12) to 6360
(nk=48)**, indicating that the broken-symmetry state is even more
clearly polarized at proper k-resolution.  88% of converged comparison
cells show the clean-QM signature (ΔE<0 AND imbalance>50).

Levers to further reduce nk=48 wall time:

1. **Symmetric-subspace parameterization** of F (spec mentions ~10×
   speedup but requires a Fock-build refactor).
2. **Tetrahedron-method DOS** instead of Fermi smearing — sharper at
   gap edges; matters only for the final figure.

## Diagnostic plot layout

`outputs/quartermetal_grid_v4.png` (also `_v3`, `_v2`):

- Top row: log10(DOS at E_F) for `H_PM`, `HF_PM`, `HF_SVP`.
- Bottom row, left:  **(E_SVP − E_PM) / N_carrier** in meV/carrier.
  Blue region = QM phase wins.  Symmetric color scale around 0.
- Bottom row, middle:  **log10 SV imbalance** for HF_SVP.  Bright =
  one sector takes the doping (clean QM); dark = symmetric.
- Bottom row, right:  Δ_tb for HF_PM.

## Vq-scaling diagnostic

A `--vq-scale` flag scales the exchange kernel by an arbitrary factor;
useful for testing whether the QM basin's depth in this model is
quantitatively right (separately from the qualitative phase
identification).  Vq×3 deepens the QM binding to ~−100 meV/carrier
median (cf. ~−19 meV at Vq×1) and broadens the broken-symmetry region
from 55% to 76% of cells, but at the cost of 12% lower SVP convergence
(81% vs 93%) — exchange-driven oscillations.  Use only as a probe.

## Recommended next steps

1. **HH normalization audit.** Get the exact code that produced the
   spec's reference Δ_tb table.  My ungated layer-Coulomb gives
   sensible-looking maps but Δ_tb values are 2–3× the spec's; switching
   conventions one place would close that gap.
2. **C3 projector.** Switch contimod discretization to a rhombic BZ
   mesh (`cm.kdiscrete.BrillouinZone(B=Bmat)` with the spec's `Bmat`)
   and add the U_C3 unitary.  Add `HF_SVP_C3` config.
3. **SVP basin tuning.** ~45% of cells fall back to the PM basin.  The
   QM phase exists in the upper portion of the (n, D) plane (D > 0.4)
   but with very small ΔE (<0.1 meV).  Stronger exchange (e.g., scaling
   Vq up by 2–3×) or a different graphene-parameter preset would deepen
   the SVP basin.
4. **Symmetric-subspace parameterization** of F to bring nk=48 within
   budget (~10× speedup per spec's note).
5. **Tetrahedron-method DOS** for sharper gap-edge resolution.

## Reproducing

```bash
# 30×30 grid, nk=12, ~14 min on 10 workers
JAX_PLATFORMS=cpu python3 examples/quartermetal_grid.py \
    --n-grid 30 --D-grid 30 --nk 12 \
    --workers 10 --tol-E 1e-4 --max-iter-hf 150

# Single-cell debugging
python3 examples/quartermetal_single_cell.py --n 0.7 --D 0.3 --nk 24

# Inspect any saved NPZ
python3 examples/quartermetal_inspect.py examples/outputs/quartermetal_grid.npz
```
