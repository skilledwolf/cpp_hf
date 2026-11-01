"""Exchange-driven spin polarization of a dilute two-dimensional electron gas."""

import hartreefock as hf
import numpy as np

# A square k-mesh; the weights make sum_k w(k) f(k) an integral over d²k/(2π)².
nk, kmax = 48, 1.2
k = np.linspace(-kmax, kmax, nk, endpoint=False) + kmax / nk
kx, ky = np.meshgrid(k, k, indexing="ij")
weights = np.full((nk, nk), (2 * kmax / nk) ** 2 / (2 * np.pi) ** 2)

# Two bands, spin up and spin down, with dispersion k²/2 (units with m = ħ = 1).
eps = 0.5 * (kx**2 + ky**2)
h = eps[..., None, None] * np.eye(2)

# The interaction lives on the grid of momentum transfers q = k - k': the same
# spacing, with q = 0 at index nk // 2. Here a dual-gate form 2π tanh(q d) / q.
dk = k[1] - k[0]
q1 = dk * (np.arange(nk) - nk // 2)
qx, qy = np.meshgrid(q1, q1, indexing="ij")
q = np.hypot(qx, qy)
d, strength = 10.0, 5.0
V = strength * 2 * np.pi * np.where(q > 0, np.tanh(q * d) / np.where(q > 0, q, 1.0), d)
coulomb_q = V[..., None, None]

kernel = hf.HartreeFockKernel(weights, h, coulomb_q, T=0.005)
n_electrons = 0.02  # electrons per unit area

# Two starting densities: spin-balanced, and fully polarized into spin up.
fill = (eps < 0.5).astype(float)
P_para = 0.5 * fill[..., None, None] * np.eye(2)
P_ferro = fill[..., None, None] * np.diag([1.0, 0.0])

# Solve from each, then ask whether the solution is a local minimum of the free energy.
config = hf.SCFConfig(acceleration="rdiis")
for name, P0 in (("balanced", P_para), ("polarized", P_ferro)):
    result = hf.solve_scf(kernel, P0, n_electrons, config=config)
    up, down = np.einsum("ij,ijaa->a", weights, result.density).real
    check = hf.stability_analysis(kernel, result.density, n_electrons)
    print(
        f"{name:>9}: E = {result.energy:.6f}, spin up/down = {up:.4f}/{down:.4f}, "
        f"stable = {check.stable} (lowest curvature {check.lowest_eigenvalue:+.1e})"
    )
