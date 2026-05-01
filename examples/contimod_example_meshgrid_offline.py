# Offline regression runner inspired by `contimod_example_meshgrid.py`.
# Loads a saved meshgrid Hamiltonian + Coulomb kernel from disk and
# compares the cpp_hf solver output against stored reference results.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make the src/ layout importable when running from the repo checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import cpp_hf  # noqa: E402
from cpp_hf import SCFConfig  # noqa: E402


def _default_case_path() -> Path:
    return REPO_ROOT / "tests" / "data" / "meshgrid_regression_case_v1.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=Path,
        default=_default_case_path(),
        help="Path to a saved regression case .npz file",
    )
    args = parser.parse_args()

    data = np.load(args.case)

    kernel = cpp_hf.HartreeFockKernel(
        weights=data["weights"],
        hamiltonian=data["hamiltonian"],
        coulomb_q=data["coulomb_q"],
        T=float(data["T"]),
        include_hartree=True,
        include_exchange=True,
        reference_density=data["reference_density"],
        hartree_matrix=data["hartree_matrix"],
    )

    result = cpp_hf.solve_scf(
        kernel,
        np.asarray(data["P0"]),
        float(data["electrondensity0"]),
        config=SCFConfig(
            max_iter=int(data["max_iter"]),
            density_tol=float(data["comm_tol"]),
            comm_tol=float(data["comm_tol"]),
            mixing=0.5,
        ),
    )
    P_fin = result.density_matrix
    F_fin = result.fock_matrix
    E_fin = result.energy
    mu_fin = result.chemical_potential
    k_fin = result.iterations

    P_expected = data["P_expected"]
    F_expected = data["F_expected"]
    E_expected = float(data["E_expected"])
    mu_expected = float(data["mu_expected"])
    k_expected = int(data["k_expected"])

    p_err = float(np.max(np.abs(np.array(P_fin) - P_expected)))
    f_err = float(np.max(np.abs(np.array(F_fin) - F_expected)))
    e_err = float(abs(float(E_fin) - E_expected))
    mu_err = float(abs(float(mu_fin) - mu_expected))

    print("k_fin:", int(k_fin), "(expected:", k_expected, ")")
    print("E_fin:", float(E_fin), "dE:", e_err)
    print("mu_fin:", float(mu_fin), "dmu:", mu_err)
    print("max |dP|:", p_err)
    print("max |dF|:", f_err)
    print("last dC:", float(result.history["commutator_residual"][int(k_fin) - 1]))

    # Tight checks; meant as a regression sanity check.
    np.testing.assert_allclose(np.array(P_fin), P_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(F_fin), F_expected, rtol=1e-6, atol=1e-6)
    if e_err > 1e-6:
        raise AssertionError(f"E mismatch: got {float(E_fin)} expected {E_expected}")
    if mu_err > 1e-6:
        raise AssertionError(f"mu mismatch: got {float(mu_fin)} expected {mu_expected}")
    if int(k_fin) != k_expected:
        print(
            f"iteration count differs from the original reference: "
            f"got {int(k_fin)}, expected {k_expected}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
