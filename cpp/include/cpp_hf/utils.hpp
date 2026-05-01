// Physics utilities: Fermi-Dirac, chemical potential bisection/Newton,
// k-grid resampling.
#pragma once

#include "cpp_hf/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpp_hf {

// Stable sigmoid; mirrors jax.scipy.special.expit and the numpy port's expit.
inline f32 expit_scalar(f32 x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    } else {
        const f32 e = std::exp(x);
        return e / (1.0 + e);
    }
}

inline f32 fermidirac_scalar(f32 x, f32 T) {
    return expit_scalar(-x / (T + 1.0e-12));
}

inline void validate_electron_count(const f32* weights, std::size_t nk,
                                     std::size_t nb, f32 n_e,
                                     const std::string& context = "n_electrons") {
    double total = 0.0;
    for (std::size_t i = 0; i < nk; ++i) total += weights[i];
    const double max_e = total * static_cast<double>(nb);
    const double tol = std::max(1e-8, 1e-6 * std::max(1.0, std::abs(max_e)));
    const double target = static_cast<double>(n_e);
    if (!std::isfinite(target))
        throw std::invalid_argument(context + " must be finite.");
    if (target < -tol || target > max_e + tol)
        throw std::invalid_argument(
            context + "=" + std::to_string(target) +
            " is outside the physically reachable range [0, " +
            std::to_string(max_e) + "].");
}

// Bisection μ-finder.  Returns μ such that ∑_k w_k Σ_j f(ε_kj − μ) = n_e.
inline f32 find_mu_bisection(const f32* eps, std::size_t n_eps,
                              const f32* weights, std::size_t nk, std::size_t nb,
                              f32 n_e, f32 T, int maxiter = 0) {
    if (maxiter <= 0) maxiter = 30;  // float32 default matches jax_hf
    const f32 Tj = std::max(T, 1.0e-12);
    f32 emin = eps[0], emax = eps[0];
    for (std::size_t i = 1; i < n_eps; ++i) {
        if (eps[i] < emin) emin = eps[i];
        if (eps[i] > emax) emax = eps[i];
    }
    const f32 span = emax - emin + 10.0 * std::max(Tj, 1.0e-6);
    f32 lo = emin - span;
    f32 hi = emax + span;
    for (int it = 0; it < maxiter; ++it) {
        const f32 mid = 0.5 * (lo + hi);
        double count = 0.0;
        for (std::size_t k = 0; k < nk; ++k) {
            const f32 wk = weights[k];
            const f32* eb = eps + k * nb;
            for (std::size_t b = 0; b < nb; ++b)
                count += static_cast<double>(wk) * fermidirac_scalar(eb[b] - mid, Tj);
        }
        if (count > n_e) hi = mid;
        else             lo = mid;
    }
    return 0.5 * (lo + hi);
}

// In-loop Newton-bracket μ-finder, mirrors jax_hf.solver._solve_mu (uses
// normalised weights and the previous μ as a warm start).
inline f32 solve_mu_inloop(const f32* eps, std::size_t n_eps,
                           const f32* w_norm, std::size_t nk, std::size_t nb,
                           f32 n_target_norm, f32 mu0, f32 T, int maxiter = 25) {
    const f32 Tj = std::max(T, 1.0e-12);
    f32 emin = eps[0], emax = eps[0];
    for (std::size_t i = 1; i < n_eps; ++i) {
        if (eps[i] < emin) emin = eps[i];
        if (eps[i] > emax) emax = eps[i];
    }
    f32 lo = emin - 50.0 * Tj;
    f32 hi = emax + 50.0 * Tj;
    f32 mu = std::clamp(mu0, lo, hi);

    for (int it = 0; it < maxiter; ++it) {
        double N = 0.0, Z = 0.0;
        for (std::size_t k = 0; k < nk; ++k) {
            const f32 wk = w_norm[k];
            const f32* eb = eps + k * nb;
            for (std::size_t b = 0; b < nb; ++b) {
                const f32 x = (mu - eb[b]) / Tj;
                const f32 p = expit_scalar(x);
                N += static_cast<double>(wk * p);
                Z += static_cast<double>(wk * p * (1.0 - p) / Tj);
            }
        }
        const f32 g = static_cast<f32>(N) - n_target_norm;
        if (g < 0.0) lo = mu;
        if (g > 0.0) hi = mu;
        const f32 Z_safe = std::max(static_cast<f32>(Z), 1.0e-18);
        f32 mu_new = mu - g / Z_safe;
        const f32 mu_bis = 0.5 * (lo + hi);
        if (mu_new <= lo || mu_new >= hi) mu_new = mu_bis;
        mu_new = std::clamp(mu_new, lo, hi);
        if (!std::isfinite(mu_new)) mu_new = mu_bis;
        mu = mu_new;
    }
    return mu;
}

// Periodic linear resampling along a single axis of a 2D-axis array.  See
// utils.resample_kgrid for the convention (centered grid, k=0 at index n//2).
inline void resample_axis_periodic_linear(const c64* x, c64* y,
                                          std::size_t n_old, std::size_t n_new,
                                          std::size_t inner) {
    if (n_old == n_new) {
        std::memcpy(y, x, n_old * inner * sizeof(c64));
        return;
    }
    const double ratio = static_cast<double>(n_old) / static_cast<double>(n_new);
    const std::size_t s_new = n_new / 2;
    const std::size_t s_old = n_old / 2;
    for (std::size_t j = 0; j < n_new; ++j) {
        double u = (static_cast<double>(j) - static_cast<double>(s_new)) * ratio
                 + static_cast<double>(s_old);
        const double floor_u = std::floor(u);
        std::int64_t i0 = static_cast<std::int64_t>(floor_u);
        i0 = ((i0 % static_cast<std::int64_t>(n_old)) + static_cast<std::int64_t>(n_old))
            % static_cast<std::int64_t>(n_old);
        const std::size_t i1 = (static_cast<std::size_t>(i0) + 1) % n_old;
        const double frac = u - floor_u;
        const c64 a = c64(1.0 - frac);
        const c64 b = c64(frac);
        const c64* x0 = x + static_cast<std::size_t>(i0) * inner;
        const c64* x1 = x + i1 * inner;
        c64* yj = y + j * inner;
        for (std::size_t i = 0; i < inner; ++i) yj[i] = a * x0[i] + b * x1[i];
    }
}

// resample_kgrid for arrays shaped (nk_old, nk_old, ...inner) to
// (nk_new, nk_new, ...inner).  Performs a two-pass axis-by-axis linear
// resampling along axes 0 and 1.
inline std::vector<c64> resample_kgrid_2d(const c64* x,
                                          std::size_t nk_old, std::size_t nk_new,
                                          std::size_t inner) {
    if (nk_old == nk_new) {
        return std::vector<c64>(x, x + nk_old * nk_old * inner);
    }
    // First pass: axis 0 — output shape (nk_new, nk_old, inner)
    std::vector<c64> tmp(nk_new * nk_old * inner);
    {
        // Treat axis 0 as the resample axis with row stride = nk_old * inner.
        const std::size_t row_in = nk_old * inner;
        const std::size_t row_out = nk_old * inner;  // keeps nk_old along axis 1
        // For each "column" along axis 1 + inner, pull from axis 0.
        // Easiest: resample axis 0 with inner = nk_old * inner.
        resample_axis_periodic_linear(x, tmp.data(), nk_old, nk_new, row_in);
        (void)row_out;
    }
    // Second pass: axis 1 of the (nk_new, nk_old, inner) intermediate.
    std::vector<c64> out(nk_new * nk_new * inner);
    for (std::size_t i = 0; i < nk_new; ++i) {
        const c64* row = tmp.data() + i * nk_old * inner;
        c64* row_out = out.data() + i * nk_new * inner;
        resample_axis_periodic_linear(row, row_out, nk_old, nk_new, inner);
    }
    return out;
}

}  // namespace cpp_hf
