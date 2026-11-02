# Changelog

All notable changes to **hartreefock** (formerly the separate `cpp_hf` and
`jax_hf` packages) are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## [3.0.0] — unreleased

### Breaking changes
- `solve_continuation` and `ContinuationResult` are gone:
  `hf.solve(kernel, P0, n, config=, coarse=coarse_kernel)` solves the coarse
  kernel from `P0` with the same config and particle number, resamples its
  density and solves the fine kernel; the coarse result is
  `result.coarse`.
- A contact term `(g, O_i, O_j)` that is not its own Hermitian partner
  gets `(g, O_j†, O_i†)` added by the kernel: the Hubbard term is
  `contact_terms=[(U, up, down)]`. Callers that passed both partners must
  drop one, or the coupling doubles.
- `hf.solve` results hold NumPy arrays and Python scalars (`float` energy
  and mu, `int` n_iter, `bool` converged), so callers no longer wrap them in
  `float()` or `np.asarray()`. Traced calls still return JAX values.
- An embedded reference (`embed_reference=True`) reports the standard HF
  energy, chemical potential and Fock matrix from `hf.solve`, the
  continuation, deflation and stability tools and `fock_operators`.
  `center_hartree` and `embedded_hartree_shift` are gone: the centring was a
  gauge shift of `h` that only moved the chemical potential.
- One entry point, one config, one result: `hf.solve(kernel, P0, n,
  config=hf.SolverConfig(method=...))` with `method` one of `"rdiis"` (the
  default), `"anderson"`, `"oda"`, `"linear"` (Roothaan iteration) or
  `"cg"`, `"newton"` (direct minimization). `solve_scf`, `SCFConfig` and
  `SCFResult` are gone; `SolverConfig(optimizer=)` and
  `SCFConfig(acceleration=)` are `method=`. `SolveResult` carries `Q` and
  `p` from direct minimization only; the Roothaan `message` string is gone
  (the residuals are in `history`). `hf.solve` without a config now runs
  RDIIS, not CG.
- One kernel and a mesh type. `hf.Mesh(cell, nk, offset, periodic)` (with
  `Mesh.square` and `Mesh.bz`) holds the k-points, their weights, the
  transfer grid, the −k map and resampling, and `HartreeFockKernel(mesh, h,
  V=None, *, T, plane_waves=None, ...)` replaces both `HartreeFockKernel(
  weights, h, Vq, ...)` and `SuperlatticeHartreeFockKernel`. `V` is a
  callable of |q| (evaluated on `mesh.transfer_distance_grid`) or a table,
  scalar or orbital-resolved; `V=None` is a contact-only model. A moiré
  basis is `plane_waves=g_frac` on a periodic mesh. The keywords
  `reference_density`, `embed_reference_density` and `center_hartree_density`
  are `reference`, `embed_reference` and `center_hartree`; `hartree=` takes
  `True` or an explicit q=0 table. Non-uniform weight arrays are gone: a
  mesh's weight is its cell area over nk².
- `SCFConfig(acceleration="diis")`, Pulay DIIS on the commutator `[F, P]`,
  is removed and `"rdiis"` is the default. The commutator vanishes for any
  collinear state, so commutator DIIS never engaged there and ran as plain
  linear mixing (22 iterations against RDIIS's 5 in the solvers guide).
- `cpp_hf` and `jax_hf` are merged into one package, `hartreefock`, with one
  kernel (`HartreeFockKernel`, `SuperlatticeHartreeFockKernel`) and one set
  of configs and results. `import cpp_hf` and `import jax_hf` are gone.
- One engine: the solvers are jitted JAX, and JAX is a required dependency.
  The C++ solver loops of `cpp_hf` (SCF, direct minimization, trust-region
  Newton) are removed; the extension keeps only the OpenMP kernels JAX calls
  through `jax.ffi` on the CPU (batched eigh and eigvalsh, batched complex
  matmul, the superlattice Fock build), which bring a JAX SCF iteration
  within 1.1x of the removed C++ loop. The `backend=` argument,
  `available_backends` and `get_backend` are gone, and so are the options
  only the C++ loops had: `optimizer="auto"`, `preconditioner="hartree"` and
  `SCFConfig.block_sizes`. `hf.fock_operators(kernel)` evaluates the Fock
  matrix, energy and response.
- Importing hartreefock turns on `jax_enable_x64`; a solve runs in the dtype
  of the kernel's arrays (complex128 from NumPy defaults).
- `SolveResult.history` arrays are cut to `n_iter` entries, as
  `SCFResult.history` already were.
- Result fields follow the former `jax_hf` names: `SCFResult.density`, `.fock`,
  `.mu`, `.n_iter` (was `density_matrix`, `fock_matrix`, `chemical_potential`,
  `iterations` in `cpp_hf`). Configs are keyword-only.
- The package is MIT licensed (binary wheels bundling FFTW remain GPL as a
  whole).

### Added
- `hf.solve(kernel, None, n)` starts from `kernel.noninteracting_density(n)`,
  the Fermi-filled bands of `h` at `T`; `field=` (a `(D, D)` matrix or one
  per k-point) biases the start, e.g. toward spin polarization.
- `kernel.mean_density(P)`, the k-summed density ρ̄ (its diagonal is the
  orbital occupations), and `kernel.energies(P)`, the kinetic, Hartree and
  exchange parts of the standard energy, which sum to `hf.solve`'s `energy`.
- `solve_minimum` and `stability_analysis`, and the `rdiis` and `anderson`
  SCF from `cpp_hf` 1.2.0, on the merged API.
- The documentation is rebuilt as executed pages: a three-step tutorial on
  the electron gas (exchange, Bloch ferromagnetism, the Stoner instability)
  and guides to the kernel, the solvers, superlattices, and precision and
  speed.
  Each page checks its result against a closed form, and the build fails
  when a check does. The solver and result docstrings have NumPy sections.

### Fixed
- `SuperlatticeHartreeFockKernel` with `center_embedded_hartree=True` now
  reports the uncentred `embedded_energy_offset` and exposes
  `embedded_hartree_shift`, so the standard HF energy is recovered at any
  particle count.

## cpp_hf [1.2.0] — 2026-10-31

### Added
- `SCFConfig(acceleration="rdiis")`: safeguarded DIIS fitted to the full
  finite-temperature density residual `f_mu(F[P]) - P` with a pivoted-QR solve,
  extrapolating Fock matrices and backtracking against the actual free energy.
  Recommended fixed-point solver for finite-temperature workloads.
- `SCFConfig(acceleration="anderson")`: the same residual and safeguard,
  extrapolating density updates.
- `SCFConfig.preconditioner` (`"none"` or `"hartree"`) and `SCFConfig.safeguard`
  for the two new methods. Both require `level_shift=0`.
- `SolverConfig(optimizer="auto")`: RDIIS with 20% of the iteration budget
  reserved for trust-region Newton recovery.
- `solve_minimum` / `MinimumResult`: RDIIS plus a local stability check and
  bounded feasible saddle escapes. `stability_analysis` / `StabilityResult`:
  the standalone matrix-free curvature test.
- `history["free_energy"]` and `history["fock_builds"]` on `solve` and
  `solve_scf` results (Fock builds include rejected trials).
- `HartreeFockKernel.exchange_reciprocity_defect` and
  `embedded_hartree_shift` (also on `SuperlatticeHartreeFockKernel`).
- The wheel smoke test now runs a native solve; a source-build test workflow
  runs on every push and PR.

### Correctness (these change numerical results)
- Exchange uses the self-adjoint part of the sampled interaction `VR`, so the
  Fock matrix is the exact derivative of the reported HF energy. On even or
  oblique grids where `V(q)` is not reciprocal under periodic inversion,
  exchange results change.
- Nonuniform quadrature weights now multiply the convolved density instead of
  the interaction kernel. Uniform-weight results are unchanged.
- Embedded-reference kernels include contact interactions in the embedded Fock
  and energy offset; the centered reference energy/μ conversion uses the
  weighted particle count.
- Chemical potential: occupation-aware brackets for dilute and near-full
  fillings, representable-update stopping, and `±inf` at exactly empty or full
  filling. `LOGIT_CLIP` tightened from 1e-6 to 1e-14 (matches jax_hf), which
  changes cold-temperature occupation gradients.
- Corrected trust-region Newton: joint orbital/occupation steps, one quadrature
  metric, the unitary connection in the covariant Hessian, full Gaussian
  deflation curvature, and symmetry projection of the gradient and response.
- With `tol_grad > 0`, CG and Newton also require the full density fixed-point
  residual before reporting convergence, and CG uses the free energy for its
  decisions. `optimizer="cg"` remains the default, but default solves may stop
  at different iterates than in 1.1.x.
- FFTW plans are created with `FFTW_UNALIGNED` (execution buffers need not match
  the planning buffer's alignment). Plan creation and destruction are
  serialized, and FFTs inside OpenMP superlattice channels run single-threaded.

### Breaking changes
- macOS wheels (arm64 and x86_64) now require macOS 15: GitHub retired the
  macOS 14 runner, and the bundled Homebrew `libomp` targets macOS 15. On
  older macOS, pip falls back to the sdist, which needs FFTW.
- `SCFConfig` fields were reordered (`project_fn` and `block_sizes` now follow
  `trust_radius`). Positional construction breaks; keyword construction is
  unaffected.
- `SolverConfig.mu_maxiter` default 25 → 50.
- `SCFConfig`, `SolverConfig` and kernel constructors validate their inputs and
  raise `ValueError` for non-finite or negative weights/temperatures, malformed
  shapes and out-of-range options that were previously accepted.
