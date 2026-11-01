// Direct minimization solver: preconditioned Riemannian CG on Stiefel × simplex.
//
// Occupations that have saturated to 0 or 1 at working precision are treated
// as active bound constraints: their logit is clipped, so their entropy
// gradient is not a gradient, and letting it into the particle-conservation
// projection turns the occupation step into an ascent direction (see
// dm_internal::occupation_gradient).
#pragma once

#include "cpp_hf/fock.hpp"
#include "cpp_hf/kernel.hpp"
#include "cpp_hf/linalg.hpp"
#include "cpp_hf/parallel.hpp"
#include "cpp_hf/utils.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cpp_hf {

struct SolverConfig {
    std::size_t max_iter = 200;
    f32 tol_E = 1.0e-7;
    // Gradient stopping criterion.  When tol_grad > 0 it is the SOLE stop for
    // both optimizers.  The logged norm includes both the orbital gradient and
    // the full Fermi-Dirac density fixed-point residual, so diagonal F and P
    // cannot hide non-stationary occupations.  tol_grad == 0 (default) uses a
    // windowed free-energy criterion plus a derived stationarity tolerance in
    // the CG path; Newton falls back to an internal 1e-6 tolerance.
    f32 tol_grad = 0.0;
    // Free-energy convergence is judged over a sliding window: stop when
    // Omega=E-T*S has improved by less than tol_E over the last
    // plateau_window iterations and the stationarity residual is small.
    // plateau_window=0 restores the per-iteration objective test.
    std::size_t plateau_window = 5;
    f32 denom_scale = 1.0e-3;
    f32 max_step = 0.6;
    std::size_t cg_restart = 10;
    f32 bt_shrink = 0.5;
    std::size_t bt_max = 8;
    int mu_maxiter = 25;
    // optimizer: 0 = preconditioned Riemannian CG (default, solve_dm),
    // 1 = trust-region Newton (response Hessian, solve_rtr in solver_rtr.hpp).
    int optimizer = 0;
    bool preserve_initial_density = false; // internal hybrid continuation
    // Trust-region Newton controls (optimizer == 1): a Steihaug truncated-CG
    // solves the joint (Q,p) response-Hessian subproblem; convergence is on full
    // orbital-and-occupation stationarity (tol_grad, falling back to 1e-6).
    f32 tr_delta0 = 0.5;        // initial trust radius
    std::size_t tr_cg_max = 20; // max Steihaug inner CG iterations per outer step
    // Deflation (optimizer == 1 only): add a repulsive Gaussian bias
    //   Phi(P) = deflation_sigma * sum_i exp(-d_i^2 / (2*deflation_length^2)),
    //   d_i^2 = sum_k w_k ||P_k - P*_{i,k}||_F^2  (weighted Frobenius),
    // to the free energy, pushing the Newton solve out of already-found basins
    // so it can discover distinct HF solutions.  deflation_targets is
    // n_deflation stacked (nk,nb,nb) density matrices, flat row-major.  Inactive
    // (a strict no-op) when n_deflation==0 or deflation_sigma==0.
    std::vector<c64> deflation_targets;     // (n_deflation * nk * nb * nb)
    std::size_t n_deflation = 0;
    f32 deflation_sigma = 0.0;              // bias height (0 = deflation off)
    f32 deflation_length = 1.0;             // bias width (weighted Frobenius)
    std::vector<std::size_t> block_sizes;  // empty = no block structure
    bool return_Q = true;
    bool return_density = true;
    bool return_fock = true;
};

struct DMResult {
    std::vector<c64> Q;        // (nk, nb, nb)
    std::vector<f32> p;        // (nk, nb)
    std::vector<c64> density;  // (nk, nb, nb)
    std::vector<c64> fock;     // (nk, nb, nb)
    f32 mu = 0.0;
    f32 energy = 0.0;
    std::size_t n_iter = 0;
    bool converged = false;
    // Optimizer-internal per-iteration logs (one entry per outer iteration):
    // hist_E stores the physical internal HF energy; hist_free stores the
    // optimized thermodynamic objective Omega=E-T*S (plus the deflation bias
    // in the Newton path); hist_grad stores the maximum of the orbital-gradient
    // norm and the full Fermi-Dirac density fixed-point residual.
    std::vector<f32> hist_E;
    std::vector<f32> hist_free;
    std::vector<f32> hist_grad;
};

namespace dm_internal {

inline bool block_sizes_enabled(const std::vector<std::size_t>& sizes,
                                std::size_t nb) {
    if (sizes.empty()) return false;
    std::size_t total = 0;
    for (std::size_t s : sizes) {
        if (s == 0) throw std::invalid_argument("block_sizes entries must be positive.");
        total += s;
    }
    if (total != nb) throw std::invalid_argument("block_sizes must sum to nb.");
    return sizes.size() > 1;
}

inline bool block_structure_ok(const c64* M, std::size_t nk, std::size_t nb,
                               const std::vector<std::size_t>& sizes) {
    const f32 scale = max_abs(M, nk * nb * nb);
    const f32 off = max_offblock_sizes(M, nk, nb, sizes);
    const f32 tol = 1.0e-10 + 1.0e-10 * scale;
    return off <= tol;
}

inline void eigh_checked(const c64* M, f32* eigs, c64* Q,
                         std::size_t nk1, std::size_t nk2, std::size_t nb,
                         const std::vector<std::size_t>& blocks) {
    if (!blocks.empty() && block_structure_ok(M, nk1*nk2, nb, blocks))
        eigh_block_sizes_batched(M, eigs, Q, nk1, nk2, nb, blocks, false);
    else
        eigh_batched(M, eigs, Q, nk1, nk2, nb);
}

// Preserve the legacy CG floating-point product evaluation order. Its strict
// energy-only line search is sensitive to changes near the rounding floor.
inline void cg_density_from_Qp(const c64* Q, const f32* p, c64* P,
                            std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        Eigen::Map<const Eigen::Array<f32, Eigen::Dynamic, 1>> pk(p + k * nb, nb);
        MatXcf Qp = Qk;
        for (std::size_t j = 0; j < nb; ++j) Qp.col(j) *= pk[j];
        MatXcf P_k = Qp * Qk.adjoint();
        std::memcpy(P + k * nb2, P_k.data(), nb2 * sizeof(c64));
    });
}

inline void cg_fock_in_orbital_basis(const c64* Q, const c64* F, c64* Ft,
                                   std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        ConstMapMatXcf Fk(F + k * nb2, nb, nb);
        MatXcf Ftk = Qk.adjoint() * Fk * Qk;
        std::memcpy(Ft + k * nb2, Ftk.data(), nb2 * sizeof(c64));
    });
}

// G_ij = 0.5 * (A_ij - conj(A_ji)),  A = (p_j - p_i) * Ft_ij,  with off-diagonal mask.

inline bool block_problem_invariant(const HFKernel& K, const ProjectFn* project_fn,
                                    const std::vector<std::size_t>& sizes) {
    (void)project_fn; // generated densities/Focks are checked before block operations
    const std::size_t nk = K.nk();
    const std::size_t nb = K.nb;
    if (!block_structure_ok(K.h, nk, nb, sizes)) return false;
    if (K.has_refP && !block_structure_ok(K.refP, nk, nb, sizes)) return false;
    for (std::size_t t = 0; t < K.n_contact; ++t) {
        if (K.contact_g[t] == 0.0) continue;
        if (!block_structure_ok(K.contact_Oi + t * nb * nb, 1, nb, sizes))
            return false;
        if (!block_structure_ok(K.contact_Oj + t * nb * nb, 1, nb, sizes))
            return false;
    }
    return true;
}

inline void density_from_Qp(const c64* Q, const f32* p, c64* P,
                            std::size_t nk, std::size_t nb,
                            const std::vector<std::size_t>& blocks = {}) {
    const std::size_t nb2 = nb * nb;
    const bool blocked = !blocks.empty() && block_structure_ok(Q, nk, nb, blocks);
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        MapMatXcf out(P + k * nb2, nb, nb);
        if (blocked) {
            out.setZero();
            std::size_t off = 0;
            for (auto size : blocks) {
                MatXcf q = Qk.block(off, off, size, size), qp = q;
                for (std::size_t j = 0; j < size; ++j) qp.col(j) *= p[k*nb+off+j];
                out.block(off, off, size, size) = qp*q.adjoint();
                off += size;
            }
        } else {
            MatXcf qp = Qk;
            for (std::size_t j = 0; j < nb; ++j) qp.col(j) *= p[k*nb+j];
            out = qp*Qk.adjoint();
        }
    });
}

// Project a density already constructed from Q/p. A degenerate orbital gauge
// or a preconditioned update can leave the symmetry algebra even when the
// Fock gradient belongs to it. Keep the entropy and gradient in the chart of
// the actual projected density; avoid an eigensolve for roundoff-only changes.
// scratch must hold nk1*nk2*nb*nb entries and must not alias P or Q.
inline bool project_density_eigensystem(
        c64* P, c64* Q, f32* p, c64* scratch, const ProjectFn* project_fn,
        std::size_t nk1, std::size_t nk2, std::size_t nb,
        const std::vector<std::size_t>& blocks = {}) {
    const std::size_t nk = nk1 * nk2, n = nk * nb * nb;
    if (!(project_fn && *project_fn)) {
        hermitize_inplace(P, nk, nb);
        return false;
    }
    std::memcpy(scratch, P, n * sizeof(c64));
    (*project_fn)(P, nk1, nk2, nb);
    hermitize_inplace(P, nk, nb);
    f32 change = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        change = std::max(change, std::abs(P[i] - scratch[i]));
    const f32 roundoff = 64.0 * std::numeric_limits<f32>::epsilon()
                       * max_abs(scratch, n);
    if (change <= roundoff) return false;
    eigh_checked(P, p, Q, nk1, nk2, nb, blocks);
    return true;
}

inline void fock_in_orbital_basis(const c64* Q, const c64* F, c64* Ft,
                                   std::size_t nk, std::size_t nb,
                                   const std::vector<std::size_t>& blocks = {}) {
    const std::size_t nb2 = nb * nb;
    const bool blocked = !blocks.empty() && block_structure_ok(Q, nk, nb, blocks)
                      && block_structure_ok(F, nk, nb, blocks);
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb), Fk(F + k * nb2, nb, nb);
        // Ft can alias F; materialize the result before writing.
        MatXcf result = MatXcf::Zero(nb, nb);
        if (blocked) {
            std::size_t off = 0;
            for (auto size : blocks) {
                auto q = Qk.block(off, off, size, size);
                result.block(off, off, size, size) = q.adjoint()*Fk.block(off, off, size, size)*q;
                off += size;
            }
        } else result = Qk.adjoint()*Fk*Qk;
        std::memcpy(Ft+k*nb2, result.data(), nb2*sizeof(c64));
    });
}

// G_ij = 0.5 * (A_ij - conj(A_ji)),  A = (p_j - p_i) * Ft_ij,  with off-diagonal mask.
inline void compute_orbital_gradient(const c64* Ft, const f32* p, c64* G,
                                     std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        const c64* Ftk = Ft + k * nb2;
        const f32* pk = p + k * nb;
        c64* Gk = G + k * nb2;
        for (std::size_t i = 0; i < nb; ++i) {
            for (std::size_t j = 0; j < nb; ++j) {
                if (i == j) { Gk[i * nb + j] = c64(0.0, 0.0); continue; }
                const f32 dp_ij = pk[j] - pk[i];
                const f32 dp_ji = pk[i] - pk[j];
                const c64 a_ij = dp_ij * Ftk[i * nb + j];
                const c64 a_ji = dp_ji * Ftk[j * nb + i];
                Gk[i * nb + j] = 0.5 * (a_ij - std::conj(a_ji));
            }
        }
    });
}

// Weighted Frobenius inner product Σ_k w_k Σ_ij Re(conj(X)_ij * Y_ij)
inline f32 ip_matrix(const c64* X, const c64* Y, const f32* w_norm,
                     std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const c64* xk = X + k * nb2;
        const c64* yk = Y + k * nb2;
        double per_k = 0.0;
        for (std::size_t i = 0; i < nb2; ++i) {
            per_k += static_cast<double>((std::conj(xk[i]) * yk[i]).real());
        }
        total += static_cast<double>(w_norm[k]) * per_k;
    }
    return static_cast<f32>(total);
}

inline f32 ip_vec(const f32* x, const f32* y, const f32* w_norm,
                  std::size_t nk, std::size_t nb) {
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        const f32* xk = x + k * nb;
        const f32* yk = y + k * nb;
        double per_k = 0.0;
        for (std::size_t b = 0; b < nb; ++b)
            per_k += static_cast<double>(xk[b]) * static_cast<double>(yk[b]);
        total += static_cast<double>(w_norm[k]) * per_k;
    }
    return static_cast<f32>(total);
}

inline f32 norm_matrix(const c64* X, const f32* w_norm,
                       std::size_t nk, std::size_t nb) {
    const f32 sq = ip_matrix(X, X, w_norm, nk, nb);
    return std::sqrt(std::max(0.0, sq));
}

inline f32 norm_matrix_difference(const c64* X, const c64* Y,
                                  const f32* w_norm,
                                  std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    double total = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        double per_k = 0.0;
        for (std::size_t i = 0; i < nb2; ++i)
            per_k += std::norm(X[k * nb2 + i] - Y[k * nb2 + i]);
        total += static_cast<double>(w_norm[k]) * per_k;
    }
    return static_cast<f32>(std::sqrt(std::max(total, 0.0)));
}

// Remove the uniform component that changes the weighted particle count.
inline void project_occupation_tangent(f32* x, const f32* weights,
                                       std::size_t nk, std::size_t nb) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        for (std::size_t b = 0; b < nb; ++b)
            num += static_cast<double>(weights[k]) * x[k * nb + b];
        den += static_cast<double>(weights[k]) * static_cast<double>(nb);
    }
    const f32 shift = den > 0.0 ? static_cast<f32>(num / den) : 0.0;
    for (std::size_t i = 0; i < nk * nb; ++i) x[i] -= shift;
}

// Bands whose occupation is resolvable away from the 0/1 bounds.  Occupations
// inside the logit clipping window are saturated: their entropy gradient is
// numerically undefined and they cannot move at the working precision, so the
// optimizers treat them as active bound constraints — held fixed, given zero
// Fermi-Dirac curvature, and excluded from the particle-conservation
// projection.
inline bool occupation_active(f32 p) {
    return p > LOGIT_CLIP && p < 1.0 - LOGIT_CLIP;
}

inline void active_occupations(const f32* p, unsigned char* active,
                               std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) active[i] = occupation_active(p[i]) ? 1 : 0;
}

// Particle-conserving projection restricted to the active bands: a uniform
// shift over the active bands only; saturated bands are held fixed (zero
// variation).  Equals project_occupation_tangent when every band is active,
// which is why that unmasked form is kept: it is the reference case this
// projection is defined against, and the tests check the two agree.
inline void project_occupation_tangent_masked(f32* x, const f32* weights,
                                              const unsigned char* active,
                                              std::size_t nk, std::size_t nb) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t i = k * nb + b;
            if (!active[i]) continue;
            num += static_cast<double>(weights[k]) * x[i];
            den += static_cast<double>(weights[k]);
        }
    }
    const f32 shift = den > 0.0 ? static_cast<f32>(num / den) : 0.0;
    for (std::size_t i = 0; i < nk * nb; ++i)
        x[i] = active[i] ? x[i] - shift : 0.0;
}

// Curvature-preconditioned particle-conserving projection
//   out = curv · (v − λ),   λ = Σ_k w_k Σ_i curv_i v_i / Σ_k w_k Σ_i curv_i,
// so that Σ_k w_k Σ_i out_i = 0: the projection of v onto the fixed-particle
// tangent space under the metric 1/curv.  Bands with curv = 0 (saturated
// occupations) receive no variation and do not influence λ, so an
// unresolvable entropy gradient on a saturated band cannot leak into the
// partially occupied ones.
inline void project_occupation_curvature(const f32* v, const f32* weights,
                                         const f32* curv, f32* out,
                                         std::size_t nk, std::size_t nb) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t k = 0; k < nk; ++k) {
        for (std::size_t b = 0; b < nb; ++b) {
            const std::size_t i = k * nb + b;
            const double wc = static_cast<double>(weights[k]) * curv[i];
            num += wc * v[i];
            den += wc;
        }
    }
    const f32 lam = den > 0.0 ? static_cast<f32>(num / den) : 0.0;
    for (std::size_t i = 0; i < nk * nb; ++i) out[i] = curv[i] * (v[i] - lam);
}

// Projected occupation gradient of Ω = E − T·S and its preconditioned form.
//
// The raw gradient with respect to p_i is eps_i + T·logit(p_i).  At a
// Fermi-Dirac state it equals the chemical potential for every band, but a
// band whose occupation has saturated to 0 or 1 at working precision has a
// clipped logit, so its raw gradient is off by up to |eps_i − μ|.  On
// low-temperature problems most bands are saturated; if that error enters the
// uniform particle-conservation shift it is spread onto the partially filled
// bands and turns the occupation step into an ascent direction.  Saturated
// bands are therefore masked out of the Euclidean projection, get zero
// curvature, and the particle constraint is enforced over the active bands.
//
// Writes g (the masked particle-conserving gradient), h = curv · (g_raw − λ)
// (its Fermi-Dirac-curvature preconditioned form with Σ_k w_k Σ_i h_i = 0 —
// a descent direction, since <g_raw, h> = Σ w curv (g_raw − λ)² ≥ 0),
// curv = p(1−p)/T on active bands and 0 elsewhere, and the active mask.
inline void occupation_gradient(const f32* eps, const f32* p, f32 T,
                                const f32* weights, std::size_t nk, std::size_t nb,
                                f32* g, f32* h, f32* curv, unsigned char* active) {
    const std::size_t n = nk * nb;
    for (std::size_t i = 0; i < n; ++i) {
        const bool act = occupation_active(p[i]);
        active[i] = act ? 1 : 0;
        const f32 pv = std::clamp(p[i], LOGIT_CLIP, 1.0 - LOGIT_CLIP);
        g[i] = eps[i] + T * (std::log(pv) - std::log1p(-pv));  // g_raw
        curv[i] = act ? p[i] * (1.0 - p[i]) / T : 0.0;
    }
    project_occupation_curvature(g, weights, curv, h, nk, nb);  // from g_raw
    project_occupation_tangent_masked(g, weights, active, nk, nb);
}

// Euclidean projection onto 0 <= p <= 1 at a fixed weighted particle count.
// In the optimizer's weighted metric the Lagrange multiplier is a uniform
// shift, found robustly by scalar bisection.
inline void project_capped_occupations(f32* p, const f32* weights,
                                       std::size_t nk, std::size_t nb,
                                       f32 n_target, int maxiter = 50) {
    double capacity = 0.0;
    for (std::size_t k = 0; k < nk; ++k)
        capacity += static_cast<double>(weights[k]) * static_cast<double>(nb);
    if (n_target <= 0.0) {
        std::fill(p, p + nk * nb, 0.0);
        return;
    }
    if (static_cast<double>(n_target) >= capacity) {
        std::fill(p, p + nk * nb, 1.0);
        return;
    }

    f32 pmin = p[0], pmax = p[0];
    for (std::size_t i = 1; i < nk * nb; ++i) {
        pmin = std::min(pmin, p[i]);
        pmax = std::max(pmax, p[i]);
    }
    f32 lo = -pmax - 1.0;
    f32 hi = 1.0 - pmin + 1.0;
    for (int it = 0; it < maxiter; ++it) {
        const f32 mid = 0.5 * (lo + hi);
        double count = 0.0;
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t b = 0; b < nb; ++b)
                count += static_cast<double>(weights[k])
                       * std::clamp(static_cast<double>(p[k * nb + b] + mid),
                                    0.0, 1.0);
        if (count < static_cast<double>(n_target)) lo = mid;
        else                                        hi = mid;
    }
    const f32 shift = 0.5 * (lo + hi);
    for (std::size_t i = 0; i < nk * nb; ++i)
        p[i] = std::clamp(p[i] + shift, 0.0, 1.0);
}

// Cayley spectral setup: eigh(i * d) where d is skew-Hermitian.  Returns
// (V_per_k, lam_per_k).  Hermitises iA exactly first to suppress roundoff.
//
// IMPORTANT: assigning Eigen's eigenvector matrix to a row-major
// ``MapMatXcf`` (rather than memcpy'ing its raw buffer) is what guarantees
// the columns end up where we expect — Eigen's internal storage may not
// match the requested ``MatrixType`` row-majorness.
inline void cayley_spectral_setup(const c64* d, c64* V, f32* lam,
                                   std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        std::vector<c64> iA(nb2);
        const c64* dk = d + k * nb2;
        for (std::size_t i = 0; i < nb2; ++i) iA[i] = c64(0.0, 1.0) * dk[i];
        MapMatXcf iAm(iA.data(), nb, nb);
        MatXcf iAh = 0.5 * (iAm + iAm.adjoint());
        Eigen::SelfAdjointEigenSolver<MatXcf> es(iAh);
        const auto& w = es.eigenvalues();
        const auto& Vk = es.eigenvectors();
        for (std::size_t i = 0; i < nb; ++i) lam[k * nb + i] = w[i];
        MapMatXcf VOut(V + k * nb2, nb, nb);
        VOut = Vk;
    });
}

inline void cayley_spectral_setup_block_sizes(const c64* d, c64* V, f32* lam,
                                               std::size_t nk, std::size_t nb,
                                               const std::vector<std::size_t>& sizes) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        std::vector<c64> iA(nb2);
        const c64* dk = d + k * nb2;
        for (std::size_t i = 0; i < nb2; ++i) iA[i] = c64(0.0, 1.0) * dk[i];
        MapMatXcf iAm(iA.data(), nb, nb);
        MapMatXcf VOut(V + k * nb2, nb, nb);
        VOut.setZero();

        std::size_t start = 0;
        for (std::size_t s : sizes) {
            MatXcf sub = iAm.block(start, start, s, s);
            sub = 0.5 * (sub + sub.adjoint());
            Eigen::SelfAdjointEigenSolver<MatXcf> es(sub);
            const auto& w = es.eigenvalues();
            const auto& Vk = es.eigenvectors();
            for (std::size_t i = 0; i < s; ++i) lam[k * nb + start + i] = w[i];
            VOut.block(start, start, s, s) = Vk;
            start += s;
        }
    });
}

// c_t = (1 + i τλ/2) / (1 − i τλ/2),  |c_t| = 1 by construction.
inline c64 cayley_factor(f32 lam, f32 tau) {
    const f32 arg = 0.5 * tau * lam;
    const c64 iexp(0.0, arg);
    return (c64(1.0, 0.0) + iexp) / (c64(1.0, 0.0) - iexp);
}

// U(τ) = V · diag(c(τ, λ)) · V†.  Output U_out is per-k (nb, nb).
inline void cayley_unitary_from_spectrum(const c64* V, const f32* lam, f32 tau,
                                          c64* U_out,
                                          std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Vk(V + k * nb2, nb, nb);
        MatXcf VcM = Vk;
        for (std::size_t j = 0; j < nb; ++j) {
            const c64 c = cayley_factor(lam[k * nb + j], tau);
            VcM.col(j) *= c;
        }
        MatXcf U = VcM * Vk.adjoint();
        std::memcpy(U_out + k * nb2, U.data(), nb2 * sizeof(c64));
    });
}

// diag(U(τ)† Ft U(τ)) = real diag of  V · M(τ) · V†, where
//   M(τ)[i, j] = c̄(τ, λ_i) · Ft_eig[i, j] · c(τ, λ_j)
// Output diag_out has shape (nk, nb).
inline void diag_UFU_from_spectrum(const c64* V, const c64* Ft_eig,
                                    const f32* lam, f32 tau, f32* diag_out,
                                    std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        std::vector<c64> c_vec(nb);
        MatXcf M(nb, nb);
        ConstMapMatXcf Vk(V + k * nb2, nb, nb);
        ConstMapMatXcf Fe(Ft_eig + k * nb2, nb, nb);
        for (std::size_t i = 0; i < nb; ++i) c_vec[i] = cayley_factor(lam[k * nb + i], tau);

        for (std::size_t i = 0; i < nb; ++i) {
            const c64 cb_i = std::conj(c_vec[i]);
            for (std::size_t j = 0; j < nb; ++j) {
                M(i, j) = cb_i * Fe(i, j) * c_vec[j];
            }
        }
        MatXcf A = Vk * M;
        // diag_out[k, i] = Re Σ_j A[i, j] * conj(V[i, j])
        for (std::size_t i = 0; i < nb; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < nb; ++j)
                s += static_cast<double>((A(i, j) * std::conj(Vk(i, j))).real());
            diag_out[k * nb + i] = static_cast<f32>(s);
        }
    });
}

inline void compute_Ft_eig(const c64* Ft, const c64* V_d, c64* Ft_eig,
                            std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Vk(V_d + k * nb2, nb, nb);
        ConstMapMatXcf Ftk(Ft + k * nb2, nb, nb);
        MatXcf out = Vk.adjoint() * Ftk * Vk;
        std::memcpy(Ft_eig + k * nb2, out.data(), nb2 * sizeof(c64));
    });
}

// Q_new = Q * U  (per k)
inline void Q_times_U(const c64* Q, const c64* U, c64* Q_new,
                       std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        ConstMapMatXcf Uk(U + k * nb2, nb, nb);
        MatXcf out = Qk * Uk;
        std::memcpy(Q_new + k * nb2, out.data(), nb2 * sizeof(c64));
    });
}

inline void Q_times_cayley_spectrum_inplace(c64* Q, const c64* V,
                                             const f32* lam, f32 tau,
                                             std::size_t nk, std::size_t nb) {
    const std::size_t nb2 = nb * nb;
    parallel_for(nk, [&](std::size_t k) {
        ConstMapMatXcf Qk(Q + k * nb2, nb, nb);
        ConstMapMatXcf Vk(V + k * nb2, nb, nb);
        MatXcf VcM = Vk;
        for (std::size_t j = 0; j < nb; ++j) {
            const c64 c = cayley_factor(lam[k * nb + j], tau);
            VcM.col(j) *= c;
        }
        MatXcf U = VcM * Vk.adjoint();
        MatXcf out = Qk * U;
        std::memcpy(Q + k * nb2, out.data(), nb2 * sizeof(c64));
    });
}

}  // namespace dm_internal

inline DMResult solve_dm(const HFKernel& K, const c64* P0, f32 n_e,
                          const SolverConfig& cfg,
                          const ProjectFn* project_fn = nullptr) {
    using namespace dm_internal;

    const std::size_t nk1 = K.nk1;
    const std::size_t nk2 = K.nk2;
    const std::size_t nb = K.nb;
    const std::size_t nk = nk1 * nk2;
    const std::size_t nb2 = nb * nb;
    const std::size_t n_tot = nk * nb2;
    const bool block_cfg_enabled = block_sizes_enabled(cfg.block_sizes, nb);
    bool use_block_ops = block_cfg_enabled
        && (!(project_fn && *project_fn) && block_problem_invariant(K, project_fn, cfg.block_sizes));

    // Normalised weights and target electron count
    std::vector<f32> w_norm(nk);
    const f32 ws = std::max(K.weight_sum, TINY_REAL);
    for (std::size_t i = 0; i < nk; ++i) w_norm[i] = K.w2d[i] / ws;
    const f32 n_target_norm = n_e / ws;

    const f32 T_r = std::max(K.T, 1.0e-12);

    // Workspaces
    std::vector<c64> Q(n_tot);
    std::vector<f32> p(nk * nb);
    std::vector<c64> P_cur(n_tot);
    std::vector<c64> Sigma(n_tot), F(n_tot);
    std::vector<f32> hartree_diag(nb, 0.0);
    std::vector<f32> eps(nk * nb);
    std::vector<c64> G_Q(n_tot);
    std::vector<f32> g_p(nk * nb), h_p(nk * nb), curv_p(nk * nb);
    std::vector<unsigned char> active_p(nk * nb);
    std::vector<c64> G_Q_prev(n_tot, c64(0.0, 0.0));
    std::vector<f32> g_p_prev(nk * nb, 0.0);
    std::vector<c64> d_Q(n_tot), d_Q_prev(n_tot, c64(0.0, 0.0));
    std::vector<f32> d_p(nk * nb), d_p_prev(nk * nb, 0.0);
    std::vector<c64> V_d(n_tot);
    std::vector<f32> lam_d(nk * nb);
    std::vector<f32> eps_work(nk * nb);
    std::vector<c64> Q_fd(n_tot), P_fd(n_tot), V_fd(n_tot);
    std::vector<f32> p_fd(nk * nb), eps_fd(nk * nb);
    std::vector<c64> Q_trial(n_tot), P_trial(n_tot);
    std::vector<f32> p_trial(nk * nb);
    std::vector<c64> Sigma_trial(n_tot), F_trial(n_tot);
    std::vector<f32> hartree_trial(nb, 0.0);

    DMResult out;
    out.hist_E.assign(cfg.max_iter, 0.0);
    out.hist_free.assign(cfg.max_iter, 0.0);
    out.hist_grad.assign(cfg.max_iter, 0.0);

    auto project_in_place = [&](c64* M) {
        if (project_fn && *project_fn) {
            (*project_fn)(M, nk1, nk2, nb);
        }
    };

    // ---- Initialize (Q, p) from Fock at P0 ----
    {
        std::memcpy(P_cur.data(), P0, n_tot * sizeof(c64));
        project_in_place(P_cur.data());
        hermitize_inplace(P_cur.data(), nk, nb);
        // Project the complete gradient, including the one-body Hamiltonian.
        // For a symmetry matrix algebra its Fermi density belongs to the same
        // algebra, so Q/p describe the actual constrained density and entropy.
        build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(),
                           hartree_diag.data(), project_fn);
        if (use_block_ops && !block_structure_ok(F.data(), nk, nb, cfg.block_sizes))
            use_block_ops = false;
        if (use_block_ops)
            eigh_block_sizes_batched(F.data(), eps.data(), Q.data(), nk1, nk2, nb,
                                     cfg.block_sizes, false);
        else
            eigh_batched(F.data(), eps.data(), Q.data(), nk1, nk2, nb);
        const f32 mu0 = solve_mu_inloop(eps.data(), nk * nb, w_norm.data(),
                                         nk, nb, n_target_norm, 0.0, T_r,
                                         cfg.mu_maxiter);
        for (std::size_t i = 0; i < nk * nb; ++i)
            p[i] = expit_scalar((mu0 - eps[i]) / T_r);
        out.mu = mu0;
    }

    f32 grad_norm = std::numeric_limits<f32>::infinity();
    f32 Omega_prev = std::numeric_limits<f32>::infinity();
    f32 E = 0.0;
    f32 Omega = 0.0;
    f32 mu = out.mu;
    std::size_t k = 0;
    bool converged = false;
    std::size_t rejected_in_a_row = 0;  // consecutive iterations that changed nothing

    while (k < cfg.max_iter) {
        // 1. Build Fock at projected hermitised density
        cg_density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb);
        const bool chart_changed = project_density_eigensystem(
            P_cur.data(), Q.data(), p.data(), Sigma.data(), project_fn,
            nk1, nk2, nb, cfg.block_sizes);

        build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(),
                           hartree_diag.data(), project_fn);
        E = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(),
                                        hartree_diag.data());
        Omega = free_energy_value(E, p.data(), K.w2d, nk, nb, T_r);

        // Full Fermi-Dirac fixed-point density of the constrained Fock.  This is
        // the occupation-safe stationarity test: diagonal F and P may have a
        // zero orbital gradient while still being far from self-consistency.
        //
        // LAZY: the FD build costs a full batched eigh + mu solve + density
        // build (+ a projection callback) per iteration, but it is consumed
        // only by (a) the stationarity stop, which cannot pass while the cheap
        // orbital gradient alone already exceeds the tolerance (grad_norm is a
        // max of the two), and (b) the damped-mixing line-search fallback.
        // Defer it until one of those needs it; F and mu are unchanged in
        // between, so the deferred result is identical to the eager one.  The
        // blocked path stays eager to keep the per-iteration
        // ``block_structure_ok`` downgrade at its original test point.
        bool fd_ready = false;
        auto ensure_fd = [&]() {
            if (fd_ready) return;
            if (use_block_ops && !block_structure_ok(F.data(), nk, nb, cfg.block_sizes))
                use_block_ops = false;
            if (use_block_ops)
                eigh_block_sizes_batched(F.data(), eps_fd.data(), Q_fd.data(),
                                         nk1, nk2, nb, cfg.block_sizes, false);
            else
                eigh_batched(F.data(), eps_fd.data(), Q_fd.data(), nk1, nk2, nb);
            const f32 mu_fd = solve_mu_inloop(
                eps_fd.data(), nk * nb, w_norm.data(), nk, nb,
                n_target_norm, mu, T_r, cfg.mu_maxiter);
            for (std::size_t i = 0; i < nk * nb; ++i)
                p_fd[i] = expit_scalar((mu_fd - eps_fd[i]) / T_r);
            cg_density_from_Qp(Q_fd.data(), p_fd.data(), P_fd.data(), nk, nb);
            project_in_place(P_fd.data());
            hermitize_inplace(P_fd.data(), nk, nb);
            fd_ready = true;
        };
        if (use_block_ops) ensure_fd();

        // Orbital-basis Fock goes into F_trial (dead until the line search
        // rebuilds it) so F keeps the constrained Fock for the deferred FD build.
        cg_fock_in_orbital_basis(Q.data(), F.data(), F_trial.data(), nk, nb);
        for (std::size_t kk = 0; kk < nk; ++kk) {
            const c64* Ftk = F_trial.data() + kk * nb2;
            for (std::size_t b = 0; b < nb; ++b)
                eps[kk * nb + b] = Ftk[b * nb + b].real();
        }

        // 2. Gradient
        compute_orbital_gradient(F_trial.data(), p.data(), G_Q.data(), nk, nb);
        // Occupation gradient on the fixed-particle tangent space with
        // saturated bands masked out, and its Fermi-Dirac-curvature
        // preconditioned form h_p (see occupation_gradient).
        occupation_gradient(eps.data(), p.data(), T_r, w_norm.data(), nk, nb,
                            g_p.data(), h_p.data(), curv_p.data(),
                            active_p.data());
        const f32 orbital_norm = norm_matrix(G_Q.data(), w_norm.data(), nk, nb);

        // Stop-test bookkeeping (unchanged decisions; the FD residual is
        // folded in only when the orbital norm alone would let the stop fire).
        const f32 stat_tol = std::max(
            std::sqrt(std::max(cfg.tol_E, 0.0)),
            std::sqrt(std::numeric_limits<f32>::epsilon()));
        bool objective_plateau = false;
        if (cfg.tol_grad <= 0.0) {
            const f32 noise = 100.0 * std::numeric_limits<f32>::epsilon()
                            * std::max(std::abs(Omega), 1.0);
            bool enough_history = false;
            f32 improvement = std::numeric_limits<f32>::infinity();
            if (cfg.plateau_window > 0) {
                enough_history = k >= cfg.plateau_window;
                if (enough_history)
                    improvement =
                        out.hist_free[k - cfg.plateau_window] - Omega;
            } else {
                enough_history = k > 0;
                if (enough_history) improvement = Omega_prev - Omega;
            }
            objective_plateau =
                enough_history && improvement >= -noise && improvement <= cfg.tol_E;
        }
        const bool need_fd_for_stop =
            (cfg.tol_grad > 0.0)
                ? (orbital_norm <= cfg.tol_grad)
                : (objective_plateau && orbital_norm <= stat_tol);
        grad_norm = orbital_norm;
        if (fd_ready || need_fd_for_stop) {
            ensure_fd();
            const f32 fixed_point_norm = norm_matrix_difference(
                P_cur.data(), P_fd.data(), w_norm.data(), nk, nb);
            grad_norm = std::max(orbital_norm, fixed_point_norm);
        }

        // Histories track the actual minimized objective and full stationarity
        // (the orbital-norm lower bound on iterations where the FD build was
        // skipped — the stop tests below are unaffected).
        out.hist_E[k] = E;
        out.hist_free[k] = Omega;
        out.hist_grad[k] = grad_norm;

        // Gradient stop (tol_grad > 0) is tested at the exact returned state.
        if (cfg.tol_grad > 0.0 && grad_norm <= cfg.tol_grad) {
            ++k;
            converged = true;
            break;
        }
        if (cfg.tol_grad <= 0.0 && objective_plateau && grad_norm <= stat_tol) {
            ++k;
            converged = true;
            break;
        }

        // 3. Precondition
        double eps_sq_sum = 0.0;
        for (std::size_t i = 0; i < nk * nb; ++i)
            eps_sq_sum += static_cast<double>(eps[i]) * static_cast<double>(eps[i]);
        const f32 eps_scale = static_cast<f32>(
            std::sqrt(eps_sq_sum / std::max<std::size_t>(nk * nb, 1) + static_cast<double>(TINY_REAL)));
        const f32 lam_pc = std::max(T_r, cfg.denom_scale * eps_scale);

        for (std::size_t kk = 0; kk < nk; ++kk) {
            const c64* Gk = G_Q.data() + kk * nb2;
            c64* Hk = d_Q.data() + kk * nb2;
            const f32* eb = eps.data() + kk * nb;
            for (std::size_t i = 0; i < nb; ++i) {
                for (std::size_t j = 0; j < nb; ++j) {
                    const f32 gap = eb[i] - eb[j];
                    const f32 base = std::sqrt(gap * gap + lam_pc * lam_pc);
                    Hk[i * nb + j] = Gk[i * nb + j] / base;
                }
            }
        }
        // 4. CG direction (Polak-Ribière+)
        const f32 pr_num = (
            ip_matrix(G_Q.data(), d_Q.data(), w_norm.data(), nk, nb)
            - ip_matrix(G_Q_prev.data(), d_Q.data(), w_norm.data(), nk, nb)
            + ip_vec(g_p.data(), h_p.data(), w_norm.data(), nk, nb)
            - ip_vec(g_p_prev.data(), h_p.data(), w_norm.data(), nk, nb));
        const f32 pr_den = (
            ip_matrix(G_Q_prev.data(), G_Q_prev.data(), w_norm.data(), nk, nb)
            + ip_vec(g_p_prev.data(), g_p_prev.data(), w_norm.data(), nk, nb));
        f32 beta = (pr_den > TINY_REAL) ? std::max(0.0, pr_num / pr_den) : 0.0;
        beta = std::min(beta, 5.0);
        if (k == 0 || chart_changed
            || (cfg.cg_restart > 0 && k % cfg.cg_restart == 0)) beta = 0.0;

        for (std::size_t i = 0; i < n_tot; ++i)
            d_Q[i] = d_Q[i] + beta * d_Q_prev[i];
        for (std::size_t i = 0; i < nk * nb; ++i)
            d_p[i] = h_p[i] + beta * d_p_prev[i];

        // 5. Backtracking on the true nonlinear HF free energy.  The same
        // joint (Q,p) candidate tested here is applied below.
        if (use_block_ops)
            cayley_spectral_setup_block_sizes(d_Q.data(), V_d.data(), lam_d.data(),
                                              nk, nb, cfg.block_sizes);
        else
            cayley_spectral_setup(d_Q.data(), V_d.data(), lam_d.data(), nk, nb);

        const f32 d_Q_norm = norm_matrix(d_Q.data(), w_norm.data(), nk, nb);
        f32 tau0 = std::min(1.0, cfg.max_step / std::max(d_Q_norm, TINY_REAL));

        f32 tau_final = tau0;
        bool step_accepted = false;
        bool used_fallback = false;
        bool trial_chart_changed = false;
        for (std::size_t bt = 0; bt < cfg.bt_max; ++bt) {
            Q_trial = Q;
            Q_times_cayley_spectrum_inplace(
                Q_trial.data(), V_d.data(), lam_d.data(), tau_final, nk, nb);
            for (std::size_t i = 0; i < nk * nb; ++i)
                p_trial[i] = p[i] - tau_final * d_p[i];
            project_capped_occupations(
                p_trial.data(), w_norm.data(), nk, nb, n_target_norm);
            cg_density_from_Qp(
                Q_trial.data(), p_trial.data(), P_trial.data(), nk, nb);
            trial_chart_changed = project_density_eigensystem(
                P_trial.data(), Q_trial.data(), p_trial.data(), Sigma_trial.data(),
                project_fn, nk1, nk2, nb, cfg.block_sizes);
            build_fock_compact(
                K, P_trial.data(), Sigma_trial.data(), F_trial.data(),
                hartree_trial.data(), project_fn);
            const f32 E_trial = hf_energy_with_hartree_diag(
                K, P_trial.data(), Sigma_trial.data(), hartree_trial.data());
            const f32 Omega_trial = free_energy_value(
                E_trial, p_trial.data(), K.w2d, nk, nb, T_r);
            if (Omega_trial < Omega) {
                step_accepted = true;
                break;
            }
            tau_final *= cfg.bt_shrink;
        }

        // If the local parameterization stalls (notably for nearly equal
        // occupations), try damped mixing toward the full FD fixed point.
        if (!step_accepted) {
            ensure_fd();
            f32 lam_fb = 1.0;
            for (std::size_t bt = 0; bt < cfg.bt_max; ++bt) {
                for (std::size_t i = 0; i < n_tot; ++i)
                    P_trial[i] = P_cur[i] + c64(lam_fb, 0.0)
                                           * (P_fd[i] - P_cur[i]);
                project_in_place(P_trial.data());
                hermitize_inplace(P_trial.data(), nk, nb);
                eigh_batched(P_trial.data(), p_trial.data(), Q_trial.data(),
                             nk1, nk2, nb);
                project_capped_occupations(
                    p_trial.data(), w_norm.data(), nk, nb, n_target_norm);
                cg_density_from_Qp(
                    Q_trial.data(), p_trial.data(), P_trial.data(), nk, nb);
                project_density_eigensystem(
                    P_trial.data(), Q_trial.data(), p_trial.data(), Sigma_trial.data(),
                    project_fn, nk1, nk2, nb, cfg.block_sizes);
                build_fock_compact(
                    K, P_trial.data(), Sigma_trial.data(), F_trial.data(),
                    hartree_trial.data(), project_fn);
                const f32 E_trial = hf_energy_with_hartree_diag(
                    K, P_trial.data(), Sigma_trial.data(), hartree_trial.data());
                const f32 Omega_trial = free_energy_value(
                    E_trial, p_trial.data(), K.w2d, nk, nb, T_r);
                if (Omega_trial < Omega) {
                    step_accepted = true;
                    used_fallback = true;
                    break;
                }
                lam_fb *= cfg.bt_shrink;
            }
        }

        // 6. Apply exactly the accepted trial; no unchecked tiny step and no
        // post-line-search Fermi reoccupation.
        if (step_accepted) {
            Q = Q_trial;
            p = p_trial;
            cg_fock_in_orbital_basis(
                Q.data(), F_trial.data(), Sigma_trial.data(), nk, nb);
            for (std::size_t kk = 0; kk < nk; ++kk)
                for (std::size_t b = 0; b < nb; ++b)
                    eps_work[kk * nb + b] =
                        Sigma_trial[kk * nb2 + b * nb + b].real();
            mu = solve_mu_inloop(
                eps_work.data(), nk * nb, w_norm.data(), nk, nb,
                n_target_norm, mu, T_r, cfg.mu_maxiter);
        }

        // 7. A fixed-point fallback or a new projected eigensystem changes the
        // orbital gauge, so its next step cannot reuse the old CG directions.
        if (step_accepted && !used_fallback && !trial_chart_changed) {
            d_Q_prev = d_Q;
            d_p_prev = d_p;
            G_Q_prev = G_Q;
            g_p_prev = g_p;
        } else {
            std::fill(d_Q_prev.begin(), d_Q_prev.end(), c64(0.0, 0.0));
            std::fill(d_p_prev.begin(), d_p_prev.end(), 0.0);
            std::fill(G_Q_prev.begin(), G_Q_prev.end(), c64(0.0, 0.0));
            std::fill(g_p_prev.begin(), g_p_prev.end(), 0.0);
        }
        Omega_prev = Omega;
        ++k;

        // Neither a line-search candidate nor a fixed-point candidate
        // descended, so the state is unchanged.  The CG history was reset
        // above, so the next iteration retries from steepest descent; if that
        // fails as well, every further iteration is an exact repeat (same F,
        // gradient, direction and candidates) and would only burn
        // 1 + 3*bt_max Fock builds each until max_iter.  The objective can no
        // longer change, so the free-energy plateau is reached; stationarity
        // decides the flag exactly as the windowed test would once the window
        // had elapsed.
        rejected_in_a_row = step_accepted ? 0 : rejected_in_a_row + 1;
        if (rejected_in_a_row >= 2) {
            ensure_fd();  // already built for the fixed-point candidates
            grad_norm = std::max(orbital_norm, norm_matrix_difference(
                P_cur.data(), P_fd.data(), w_norm.data(), nk, nb));
            out.hist_grad[k - 1] = grad_norm;
            converged = (cfg.tol_grad > 0.0)
                ? (grad_norm <= cfg.tol_grad)
                : (grad_norm <= stat_tol);
            break;
        }
    }

    // ---- Finalize without changing the optimized state ----
    cg_density_from_Qp(Q.data(), p.data(), P_cur.data(), nk, nb);
    project_density_eigensystem(
        P_cur.data(), Q.data(), p.data(), Sigma.data(), project_fn,
        nk1, nk2, nb, cfg.block_sizes);
    build_fock_compact(K, P_cur.data(), Sigma.data(), F.data(),
                       hartree_diag.data(), project_fn);
    if (use_block_ops && !block_structure_ok(F.data(), nk, nb, cfg.block_sizes))
        use_block_ops = false;
    if (use_block_ops)
        eigh_block_sizes_batched(F.data(), eps_fd.data(), Q_fd.data(),
                                 nk1, nk2, nb, cfg.block_sizes, false);
    else
        eigh_batched(F.data(), eps_fd.data(), Q_fd.data(), nk1, nk2, nb);
    const f32 mu_check = solve_mu_inloop(
        eps_fd.data(), nk * nb, w_norm.data(), nk, nb,
        n_target_norm, mu, T_r, cfg.mu_maxiter);
    for (std::size_t i = 0; i < nk * nb; ++i)
        p_fd[i] = expit_scalar((mu_check - eps_fd[i]) / T_r);
    cg_density_from_Qp(Q_fd.data(), p_fd.data(), P_fd.data(), nk, nb);
    project_in_place(P_fd.data());
    hermitize_inplace(P_fd.data(), nk, nb);
    const f32 final_res = norm_matrix_difference(
        P_cur.data(), P_fd.data(), w_norm.data(), nk, nb);
    const f32 stationarity_tol = std::max(
        cfg.tol_grad > 0.0 ? cfg.tol_grad
                           : std::sqrt(std::max(cfg.tol_E, 0.0)),
        cfg.tol_grad > 0.0
            ? 100.0 * std::numeric_limits<f32>::epsilon()
            : std::sqrt(std::numeric_limits<f32>::epsilon()));
    converged = converged && final_res <= stationarity_tol;

    const f32 E_fin = hf_energy_with_hartree_diag(K, P_cur.data(), Sigma.data(),
                                                  hartree_diag.data());
    cg_fock_in_orbital_basis(Q.data(), F.data(), Sigma.data(), nk, nb);
    for (std::size_t kk = 0; kk < nk; ++kk)
        for (std::size_t b = 0; b < nb; ++b)
            eps_work[kk * nb + b] =
                Sigma[kk * nb2 + b * nb + b].real();
    mu = solve_mu_inloop(
        eps_work.data(), nk * nb, w_norm.data(), nk, nb,
        n_target_norm, mu, T_r, cfg.mu_maxiter);

    if (cfg.return_Q) out.Q = std::move(Q);
    out.p = std::move(p);
    if (cfg.return_density) out.density = std::move(P_cur);
    if (cfg.return_fock) out.fock = std::move(F);
    out.mu = mu;
    out.energy = E_fin;
    out.n_iter = k;
    out.converged = converged;
    out.hist_E.resize(k);
    out.hist_free.resize(k);
    out.hist_grad.resize(k);
    return out;
}


}  // namespace cpp_hf
