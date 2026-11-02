"""Direct minimization entry point: dispatches to Riemannian CG
(:mod:`.riemannian`) or trust-region Newton (:mod:`.newton`)."""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from ..config import SolverConfig, SolveResult, drop_unrequested
from .fock import build_fock, fock_builder
from .newton import _solve_newton_jitted
from .riemannian import _solve_jitted
from .utils import (
    _has_tracer,
    device_args,
    validate_electron_count,
    validate_temperature,
)


def minimize(
    kernel,
    P0,
    n_electrons: float,
    *,
    config: SolverConfig | None = None,
) -> SolveResult:
    """Direct-minimization Hartree-Fock solver.

    With ``config.method="cg"``: preconditioned Riemannian CG on
    Stiefel x capped simplex with true-free-energy backtracking (one
    current-state Fock build plus trial builds).  With
    ``config.method="newton"``: trust-region Newton (Steihaug truncated-CG
    on the joint (Q, p) response Hessian), which converges on the gradient norm
    and typically needs fewer outer iterations on stiff problems.  See
    :class:`SolverConfig` for tuning knobs.

    Parameters
    ----------
    kernel
        :class:`HartreeFockKernel` holding problem arrays (pre-computed FFT
        of the interaction kernel, Hamiltonian, reference density, etc).
    P0
        Initial density matrix, shape ``(nk1, nk2, nb, nb)``.  If the seed
        is arbitrary (e.g. zeros), the solver auto-diagonalises ``F[P0]``
        internally before starting the CG loop.
    n_electrons
        Target electron count (sum of weights × occupations).
    config
        Solver controls; ``method`` is ``"cg"`` or ``"newton"``.

    Returns
    -------
    SolveResult
        Named tuple with ``Q, p, mu, density, fock, energy, n_iter,
        converged, history``.  ``density`` = ``Q @ diag(p) @ Q†`` is the
        self-consistent density matrix represented by the returned optimized
        ``Q`` and ``p``.

    Notes
    -----
    The inner loop uses Cayley retractions for orbitals and capped-simplex
    occupation steps.  The returned ``Q`` and ``p`` represent the optimized
    density directly. There is no unconditional final re-occupation; a single
    canonical fixed-point polish is used only when it reduces an already-small
    residual and the polished state is rechecked against its rebuilt Fock
    matrix.

    JIT cache strategy: only *structurally static* config fields trigger
    recompilation — loop-bound integers (``max_iter``, ``bt_max``,
    ``cg_restart``, ``mu_maxiter``), the optional ``block_sizes`` tuple,
    and ``project_fn`` (callable identity).  Float tolerances and step
    sizes (``tol_E``, ``tol_grad``, ``max_step``, ``bt_shrink``,
    ``denom_scale``) stay dynamic and can be tuned between calls without
    triggering a recompile.
    """
    if config is None:
        config = SolverConfig(method="cg")
    P0_arr = jnp.asarray(P0)
    if P0_arr.shape != kernel.h.shape:
        raise ValueError(
            f"P0 must have shape {tuple(kernel.h.shape)}, got {tuple(P0_arr.shape)}."
        )
    validate_electron_count(
        kernel.w2d,
        kernel.h.shape[-1],
        n_electrons,
        context="n_electrons",
    )
    validate_temperature(kernel.T, context="kernel.T")
    if config.block_sizes is not None and sum(config.block_sizes) != kernel.h.shape[-1]:
        raise ValueError(
            f"block_sizes must sum to {kernel.h.shape[-1]}, got {sum(config.block_sizes)}."
        )

    builder = fock_builder(kernel)
    custom_fock = None if builder is build_fock else builder

    optimizer = config.method
    deflation_on = (
        config.deflation_targets is not None and float(config.deflation_sigma) > 0.0
    )
    if deflation_on and optimizer != "newton":
        raise ValueError(
            f"deflation is only supported with method='newton' (got method={config.method!r})."
        )

    if optimizer == "newton":
        # Trust-region Newton is a second-order method: its Steihaug Hessian-
        # vector products go through the FFT Fock build, whose float32 round-off
        # (~1e-6) swamps the small Hessian eigenvalues on near-degenerate
        # problems and yields ascent directions; warn rather than silently
        # return converged=False.
        if jnp.finfo(jnp.asarray(kernel.h).real.dtype).bits < 64:
            warnings.warn(
                "method='newton' needs float64 precision to converge "
                "reliably; the kernel is float32. Build it from float64 "
                "arrays; method='cg' works in float32.",
                stacklevel=2,
            )
        # Trust-region Newton converges on the gradient norm; default tol_grad
        # to 1e-6 when the caller leaves it at 0.
        tol_grad = float(config.tol_grad) if config.tol_grad > 0.0 else 1e-6

        # Deflation targets -> (n_solutions, nk1, nk2, nb, nb), matching the
        # kernel dtype.  An empty (0, ...) stack with has_deflation=False is the
        # no-deflation path (the bias machinery is then compiled out entirely).
        h_dtype = jnp.asarray(kernel.h).dtype
        nk1, nk2, nb = kernel.h.shape[0], kernel.h.shape[1], kernel.h.shape[-1]
        if deflation_on:
            defl_arr = jnp.asarray(config.deflation_targets, dtype=h_dtype)
            if defl_arr.ndim == 4:
                defl_arr = defl_arr[None, ...]
            if defl_arr.ndim != 5 or defl_arr.shape[1:] != (nk1, nk2, nb, nb):
                raise ValueError(
                    "deflation_targets must have shape "
                    "(n_solutions, nk1, nk2, nb, nb) matching the kernel; got "
                    f"{tuple(jnp.asarray(config.deflation_targets).shape)}."
                )
        else:
            defl_arr = jnp.zeros((0, nk1, nk2, nb, nb), dtype=h_dtype)

        return _finish(_solve_newton_jitted(
            P0_arr,
            float(n_electrons),
            tol_grad,
            float(config.tr_delta0),
            float(config.denom_scale),
            **device_args(kernel),
            deflation_targets=defl_arr,
            deflation_sigma=float(config.deflation_sigma),
            deflation_length=float(config.deflation_length),
            has_deflation=bool(deflation_on),
            max_iter=int(config.max_iter),
            tr_cg_max=int(config.tr_cg_max),
            mu_maxiter=int(config.mu_maxiter),
            block_sizes=config.block_sizes,
            project_fn=config.project_fn,
            fock_build_fn=custom_fock,
        ), config)

    return _finish(_solve_jitted(
        P0_arr,
        float(n_electrons),
        float(config.tol_E),
        float(config.tol_grad),
        float(config.max_step),
        float(config.bt_shrink),
        float(config.denom_scale),
        **device_args(kernel),
        max_iter=int(config.max_iter),
        bt_max=int(config.bt_max),
        cg_restart=int(config.cg_restart),
        mu_maxiter=int(config.mu_maxiter),
        plateau_window=int(config.plateau_window),
        block_sizes=config.block_sizes,
        project_fn=config.project_fn,
        fock_build_fn=custom_fock,
    ), config)


def _finish(result, config):
    """Trim the fixed-length history to the iterations run and drop unrequested arrays."""
    if not _has_tracer(result.n_iter):
        n_iter = int(result.n_iter)
        result = result._replace(
            history={key: value[:n_iter] for key, value in result.history.items()}
        )
    return drop_unrequested(result, config)
