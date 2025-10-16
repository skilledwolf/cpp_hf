#pragma once
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace cpp_hf_mixing {

using cxd = std::complex<double>;

// ----------------------------- Utilities -----------------------------
inline double real_inner(const std::vector<cxd>& a, const std::vector<cxd>& b) {
    double s = 0.0; const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        // Re( conj(a[i]) * b[i] )
        s += std::real(a[i]) * std::real(b[i]) + std::imag(a[i]) * std::imag(b[i]);
    }
    return s;
}
inline double norm2_real(const std::vector<cxd>& a) {
    return std::sqrt(real_inner(a, a));
}

// ----------------------------- CDIIS (DIIS) -----------------------------
struct DiisState {
    std::vector<std::vector<cxd>> residuals; // ring buffer (flattened)
    std::vector<std::vector<cxd>> ps_flat;   // ring buffer (flattened P)
    size_t count = 0;
    size_t max_vecs;

    explicit DiisState(size_t m) : max_vecs(m) {
        residuals.reserve(m); ps_flat.reserve(m);
    }

    // Matches JAX defaults
    std::vector<cxd> update_cdiis(const std::vector<cxd>& P_vec,   // candidate mix (current iterate)
                                  const std::vector<cxd>& resid,   // current residual (flattened)
                                  const std::vector<cxd>& P_cur,   // current P (unmixed)
                                  double coeff_cap    = 5.0,
                                  double eps_reg      = 1e-9,
                                  double blend_keep   = 0.7,
                                  double blend_new    = 0.3,
                                  double angle_thresh = 0.99) {
        const double tiny = 1e-12;

        // ---- 1) Parallel-residual overwrite logic (JAX parity) ----
        bool do_overwrite = false;
        size_t idx = 0;
        if (count > 0 && max_vecs > 0 && !residuals.empty()) {
            const size_t last_idx = (count - 1) % max_vecs;
            const auto& r_last = residuals[last_idx];
            const double denom = norm2_real(r_last) * norm2_real(resid) + tiny;
            const double cosang = (denom > 0.0) ? (real_inner(r_last, resid) / denom) : 0.0;
            do_overwrite = (cosang > angle_thresh);
            idx = do_overwrite ? last_idx : (count % max_vecs);
        } else {
            idx = 0;
        }

        if (residuals.size() < max_vecs && !do_overwrite && count < max_vecs) {
            residuals.push_back(resid);
            ps_flat.push_back(P_vec);
        } else {
            if (max_vecs == 0) return P_vec;
            if (residuals.size() < max_vecs) { residuals.resize(max_vecs); ps_flat.resize(max_vecs); }
            residuals[idx] = resid;
            ps_flat[idx]   = P_vec;
        }
        if (!do_overwrite) ++count;

        const size_t m = std::min(count, max_vecs);
        if (m < 2) {
            // Gentle fallback: 0.7*P_cur + 0.3*P_vec
            std::vector<cxd> out(P_vec.size());
            for (size_t t = 0; t < out.size(); ++t)
                out[t] = P_cur[t] * blend_keep + P_vec[t] * blend_new;
            return out;
        }

        // ---- 2) B matrix over the last m entries ----
        Eigen::MatrixXd B((Eigen::Index)m, (Eigen::Index)m);
        for (size_t i = 0; i < m; ++i) {
            const size_t ii = (count - m + i) % max_vecs;
            for (size_t j = 0; j < m; ++j) {
                const size_t jj = (count - m + j) % max_vecs;
                B((Eigen::Index)i, (Eigen::Index)j) = real_inner(residuals[ii], residuals[jj]);
            }
        }

        // ---- 3) Augmented Pulay system, full-diagonal regularization ----
        const size_t pad = m + 1;
        Eigen::MatrixXd BA = Eigen::MatrixXd::Zero((Eigen::Index)pad, (Eigen::Index)pad);
        BA.topLeftCorner((Eigen::Index)m, (Eigen::Index)m) = B;
        BA.block((Eigen::Index)m, 0, 1, (Eigen::Index)m).setConstant(-1.0);
        BA.block(0, (Eigen::Index)m, (Eigen::Index)m, 1).setConstant(-1.0);
        BA.diagonal().array() += eps_reg;  // JAX: B_full += eps * I on full diag

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero((Eigen::Index)pad);
        rhs((Eigen::Index)pad - 1) = -1.0;

        // ---- 4) SVD pseudo-inverse solve (JAX parity) ----
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(BA, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& U = svd.matrixU();
        const auto& V = svd.matrixV();
        Eigen::VectorXd s = svd.singularValues();
        Eigen::VectorXd s_inv = s.unaryExpr([](double x) { return (x > 1e-8) ? (1.0 / x) : 0.0; });
        Eigen::VectorXd sol = V * s_inv.asDiagonal() * (U.transpose() * rhs);

        // Coeffs (drop Lagrange multiplier), normalize by sum
        std::vector<double> coeff(m, 0.0);
        double sumc = 0.0;
        for (size_t i = 0; i < m; ++i) { coeff[i] = sol((Eigen::Index)i); sumc += coeff[i]; }
        if (std::abs(sumc) > 0.0) for (auto& c : coeff) c /= sumc;

        // Cap / stability
        double cmax = 0.0;
        for (double c : coeff) cmax = std::max(cmax, std::abs(c));
        const bool unstable = (!std::isfinite(cmax)) || (cmax > coeff_cap);

        // ---- 5) Build mixed P ----
        std::vector<cxd> mix(P_vec.size(), cxd(0.0, 0.0));
        for (size_t i = 0; i < m; ++i) {
            const size_t ii = (count - m + i) % max_vecs;
            const double ci = coeff[i];
            const auto& src = ps_flat[ii];
            for (size_t t = 0; t < mix.size(); ++t) mix[t] += src[t] * ci;
        }

        if (unstable) {
            std::vector<cxd> out(mix.size());
            for (size_t t = 0; t < mix.size(); ++t)
                out[t] = P_cur[t] * blend_keep + mix[t] * blend_new;
            return out;
        }
        return mix;
    }
};

// ----------------------------- EDIIS -----------------------------
struct EdiisState {
    std::vector<std::vector<cxd>> ps_flat, fs_flat;
    std::vector<double> energy; // real energies
    size_t count = 0, max_vecs;

    explicit EdiisState(size_t m) : max_vecs(m) {
        ps_flat.reserve(m); fs_flat.reserve(m); energy.reserve(m);
    }

    static std::vector<double> project_simplex(std::vector<double> v) {
        // Euclidean projection onto the probability simplex
        std::vector<double> u = v;
        std::sort(u.begin(), u.end(), std::greater<double>());
        double css = 0.0; size_t rho = 0;
        for (size_t i = 0; i < u.size(); ++i) {
            css += u[i];
            const double t = (css - 1.0) / (i + 1);
            if (u[i] - t > 0.0) rho = i;
        }
        const double tau = (std::accumulate(u.begin(), u.begin() + rho + 1, 0.0) - 1.0) / (rho + 1);
        for (auto& x : v) x = std::max(0.0, x - tau);
        return v;
    }

    // Returns (P_mix, used_ediis_only)
    std::pair<std::vector<cxd>, bool>
    update(const std::vector<cxd>& P,
           const std::vector<cxd>& F,
           double E,
           size_t max_iter_qp = 40) {
        if (max_vecs < 2) return {P, true};

        if (count < max_vecs) { ps_flat.push_back(P); fs_flat.push_back(F); energy.push_back(E); }
        else {
            size_t idx = count % max_vecs;
            ps_flat[idx] = P; fs_flat[idx] = F; energy[idx] = E;
        }
        ++count;
        const size_t m = std::min(count, max_vecs);
        if (m < 2) return {P, true};

        // Build g and M over the last m entries
        std::vector<double> g(m, 0.0);
        std::vector<std::vector<double>> M(m, std::vector<double>(m, 0.0));
        double max_abs_M = 0.0;

        for (size_t i = 0; i < m; ++i) {
            const size_t ii = (count - m + i) % max_vecs;
            const double tr = real_inner(ps_flat[ii], fs_flat[ii]);
            g[i] = energy[ii] - 0.5 * tr;
            for (size_t j = 0; j < m; ++j) {
                const size_t jj = (count - m + j) % max_vecs;
                M[i][j] = real_inner(ps_flat[ii], fs_flat[jj]);
                max_abs_M = std::max(max_abs_M, std::abs(M[i][j]));
            }
        }

        // PG iterations with γ = 1/(max|M| + 1)
        const double gamma = 1.0 / (max_abs_M + 1.0);
        std::vector<double> c(m, 1.0 / static_cast<double>(m));

        for (size_t it = 0; it < max_iter_qp; ++it) {
            std::vector<double> grad(m, 0.0);
            for (size_t i = 0; i < m; ++i) {
                double s = g[i];
                for (size_t j = 0; j < m; ++j) s += M[i][j] * c[j];
                grad[i] = s;
            }
            for (size_t i = 0; i < m; ++i) c[i] -= gamma * grad[i];
            c = project_simplex(c);
        }

        // Build mixed P
        std::vector<cxd> out(P.size(), cxd(0.0, 0.0));
        for (size_t i = 0; i < m; ++i) {
            const size_t ii = (count - m + i) % max_vecs;
            const double ci = c[i];
            const auto& src = ps_flat[ii];
            for (size_t t = 0; t < out.size(); ++t) out[t] += src[t] * ci;
        }
        return {out, false};
    }
};

// ----------------------------- Orbital Preconditioner -----------------------------
// comm, F: flattened n x n (column- or row-major must be consistent across calls)
inline std::vector<cxd>
orbital_preconditioner(const std::vector<cxd>& comm,
                       const std::vector<cxd>& F,
                       int n,
                       double delta = 1e-3) {
    Eigen::Map<const Eigen::MatrixXcd> Fm(F.data(), n, n);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Fm);
    const Eigen::VectorXd eps = es.eigenvalues();
    const Eigen::MatrixXcd C  = es.eigenvectors();          // columns are eigenvectors

    Eigen::Map<const Eigen::MatrixXcd> R(comm.data(), n, n);
    Eigen::MatrixXcd Rmo = C.adjoint() * R * C;             // to MO
    // denom_{pq} = eps_p - eps_q + delta
    Eigen::MatrixXcd Rtilde(n, n);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            const double denom = (eps(p) - eps(q)) + delta;
            Rtilde(p, q) = Rmo(p, q) / denom;
        }
    }
    Eigen::MatrixXcd R_pc = C * Rtilde * C.adjoint();       // back to AO

    std::vector<cxd> out(n * n);
    Eigen::Map<Eigen::MatrixXcd>(out.data(), n, n) = R_pc;
    return out;
}

// ----------------------------- Broyden (L-BFGS-like) -----------------------------
struct BroydenState {
    std::vector<std::vector<cxd>> s_hist; // differences of P (flattened)
    std::vector<std::vector<cxd>> y_hist; // differences of residual (or precond. residual)
    size_t count = 0;
    size_t max_vecs;
    explicit BroydenState(size_t m, size_t n_flat, cxd zero = cxd(0,0)) : max_vecs(m) {
        s_hist.assign(m, std::vector<cxd>(n_flat, zero));
        y_hist.assign(m, std::vector<cxd>(n_flat, zero));
    }
};

inline std::pair<BroydenState, std::vector<cxd>>
broyden_update(BroydenState st,
               const std::vector<cxd>& P,
               const std::vector<cxd>& resid_flat,
               double alpha = 1.3) {
    const size_t n = P.size();
    const size_t k = st.count;
    if (k == 0) { return {st, P}; }

    const size_t idx_prev = (k - 1) % st.max_vecs;
    // s_k := P_k - P_{k-1} (we store P_{k-1} in s_hist[idx_prev] to economize a buffer pass)
    // y_k := r_k - r_{k-1}
    std::vector<cxd> s_k(n), y_k(n);
    for (size_t i = 0; i < n; ++i) {
        s_k[i] = P[i] - st.s_hist[idx_prev][i];
        y_k[i] = resid_flat[i] - st.y_hist[idx_prev][i];
    }
    const size_t idx = k % st.max_vecs;
    st.s_hist[idx] = s_k;
    st.y_hist[idx] = y_k;
    st.count = k + 1;

    // Two-loop recursion
    std::vector<cxd> q = resid_flat;
    std::vector<cxd> alpha_arr(st.max_vecs, cxd(0,0));
    for (size_t i = 0; i < st.max_vecs; ++i) {
        const size_t j = (st.max_vecs - 1 - i);
        const auto& si = st.s_hist[j];
        const auto& yi = st.y_hist[j];
        double rho = 1.0 / (real_inner(yi, si) + 1e-12);
        cxd a_i = rho * cxd(real_inner(si, q), 0.0);
        alpha_arr[j] = a_i;
        // q -= a_i * yi
        for (size_t t = 0; t < n; ++t) q[t] -= a_i * yi[t];
    }
    // scaling
    double sy = real_inner(s_k, y_k);
    double yy = real_inner(y_k, y_k) + 1e-12;
    double gamma = sy / yy;
    std::vector<cxd> r(n);
    for (size_t t = 0; t < n; ++t) r[t] = cxd(gamma, 0.0) * q[t];

    for (size_t i = 0; i < st.max_vecs; ++i) {
        const auto& si = st.s_hist[i];
        const auto& yi = st.y_hist[i];
        double rho = 1.0 / (real_inner(yi, si) + 1e-12);
        cxd beta = cxd(rho * real_inner(yi, r), 0.0);
        cxd coeff = alpha_arr[i] - beta;
        for (size_t t = 0; t < n; ++t) r[t] += coeff * si[t];
    }

    std::vector<cxd> P_out(n);
    for (size_t i = 0; i < n; ++i) P_out[i] = P[i] + cxd(alpha, 0.0) * r[i];
    return {st, P_out};
}

// ----------------------------- Mixer (EDIIS -> Broyden) -----------------------------
struct MixerState {
    EdiisState ediis;
    BroydenState broyden;
    bool use_ediis;
    MixerState(size_t max_vecs, size_t n_flat)
        : ediis(max_vecs),
          broyden(max_vecs, n_flat),
          use_ediis(true) {}
};

// ‖comm‖∞ threshold to switch from EDIIS to Broyden (JAX default ~ 0.3)
inline std::pair<MixerState, std::vector<cxd>>
mixer_update(MixerState st,
             const std::vector<cxd>& P,           // current P (n_flat)
             const std::vector<cxd>& F,           // current F (n_flat)
             double E,                            // current energy (real)
             const std::vector<cxd>& comm,        // current commutator (n*n flattened)
             int n_mat,                           // matrix size for preconditioner
             double ediis_to_cdiis_tol = 3e-1) {
    // sup-norm of comm
    double comm_norm_inf = 0.0;
    for (const auto& z : comm) comm_norm_inf = std::max(comm_norm_inf, std::abs(z));
    const bool use_ediis_now = st.use_ediis ? true : (comm_norm_inf > ediis_to_cdiis_tol);

    if (use_ediis_now) {
        auto [newP, _ediis_only] = st.ediis.update(P, F, E);
        // still in EDIIS; keep Broyden state unchanged, keep flag until below tol once
        bool new_flag = (comm_norm_inf > ediis_to_cdiis_tol);
        st.use_ediis = new_flag;
        return {st, newP};
    }

    // Leaving EDIIS -> Broyden: apply orbital preconditioner and transitional blend on first step
    const std::vector<cxd> comm_pc = orbital_preconditioner(comm, F, n_mat, 1e-3);

    // Reinit Broyden when switching from EDIIS (JAX parity)
    if (st.use_ediis) {
        st.broyden = BroydenState(st.broyden.max_vecs, P.size());
    }

    // Store previous P/resid in broyden state for difference (we reuse s_hist/y_hist memory)
    if (st.broyden.count == 0) {
        // seed the "previous" with zeros to let first step be mostly a scaled precond direction
        st.broyden.s_hist[(st.broyden.count) % st.broyden.max_vecs] = P;
        st.broyden.y_hist[(st.broyden.count) % st.broyden.max_vecs] = comm_pc;
        st.broyden.count++;
    }
    auto [bst, P_raw] = broyden_update(st.broyden, P, comm_pc, /*alpha=*/1.3);
    st.broyden = bst;

    // Transitional blend ONLY when we just left EDIIS this call
    std::vector<cxd> P_out(P.size());
    if (st.use_ediis) {
        for (size_t i = 0; i < P.size(); ++i)
            P_out[i] = cxd(0.3, 0.0) * P_raw[i] + cxd(0.7, 0.0) * P[i];
    } else {
        P_out = std::move(P_raw);
    }

    st.use_ediis = false;
    return {st, P_out};
}

} // namespace cpp_hf_mixing
