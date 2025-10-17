// mixers.hpp - DIIS/EDIIS/Broyden mixers and orbital preconditioner
#pragma once

#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace hf {

using cxd  = std::complex<double>;
using MatC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Re(tr(A^* B)) for flattened complex arrays
inline double real_inner(const std::vector<cxd>& a, const std::vector<cxd>& b) {
    Eigen::Map<const Eigen::VectorXcd> va(a.data(), (Eigen::Index)a.size());
    Eigen::Map<const Eigen::VectorXcd> vb(b.data(), (Eigen::Index)b.size());
    return va.dot(vb).real();
}

// Weighted Frobenius-Re inner product across k-blocks: sum_k w_k * Re Tr(A_k^† B_k)
inline double weighted_inner_blocks(const std::vector<cxd>& A,
                                    const std::vector<cxd>& B,
                                    const std::vector<double>& w,
                                    size_t nk1, size_t nk2, size_t d) {
    double s = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) reduction(+:s) schedule(static)
#endif
    for (long long k1=0;k1<(long long)nk1;++k1)
      for (long long k2=0;k2<(long long)nk2;++k2) {
        const size_t base = (static_cast<size_t>(k1)*nk2 + static_cast<size_t>(k2)) * d * d;
        const double wk = w[(size_t)k1*nk2 + (size_t)k2];
        Eigen::Map<const MatC> Ak(&A[base], d, d);
        Eigen::Map<const MatC> Bk(&B[base], d, d);
        s += wk * ((Ak.adjoint() * Bk).trace().real());
      }
    return s;
}

// ---------------- DIIS / EDIIS ----------------
struct DiisState {
    std::vector<std::vector<cxd>> residuals; // ring buffer
    std::vector<std::vector<cxd>> ps_flat;   // ring buffer
    size_t count = 0;
    size_t max_vecs;

    explicit DiisState(size_t m): max_vecs(m) { residuals.reserve(m); ps_flat.reserve(m); }
    static std::vector<cxd> flatten(const std::vector<cxd>& a) { return a; }

    std::vector<cxd> update_cdiis(const std::vector<cxd>& p_vec,   // store P_new
                                  const std::vector<cxd>& comm,    // residual for p_vec
                                  const std::vector<cxd>& p_cur,   // current P (for fallback blend)
                                  double coeff_cap, double eps_reg,
                                  double blend_keep, double blend_new) {
        if (max_vecs < 2) { // fallback: simple blend
            std::vector<cxd> out(p_vec.size());
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)out.size();++t) out[(size_t)t] = p_cur[(size_t)t]*blend_keep + p_vec[(size_t)t]*blend_new;
            return out;
        }

        if (count < max_vecs) { residuals.push_back(comm); ps_flat.push_back(p_vec); }
        else { size_t idx = count % max_vecs; residuals[idx] = comm; ps_flat[idx] = p_vec; }
        ++count; const size_t m = std::min(count, max_vecs);
        if (m < 2) return p_vec;

        const size_t n_aug = m + 1;
        Eigen::MatrixXd BA = Eigen::MatrixXd::Zero(n_aug, n_aug);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                const auto& ri = residuals[(count - m + i) % max_vecs];
                const auto& rj = residuals[(count - m + j) % max_vecs];
                BA((Eigen::Index)i,(Eigen::Index)j) = real_inner(ri, rj);
            }
            BA((Eigen::Index)i,(Eigen::Index)m) = -1.0;
            BA((Eigen::Index)m,(Eigen::Index)i) = -1.0;
            BA((Eigen::Index)i,(Eigen::Index)i) += eps_reg;
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_aug); rhs(n_aug-1) = -1.0;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(BA);
        if (ldlt.info() != Eigen::Success) {
            std::vector<cxd> out(p_vec.size());
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)out.size();++t) out[(size_t)t] = p_cur[(size_t)t]*blend_keep + p_vec[(size_t)t]*blend_new;
            return out;
        }
        auto sol = ldlt.solve(rhs);
        if (ldlt.info() != Eigen::Success) {
            std::vector<cxd> out(p_vec.size());
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)out.size();++t) out[(size_t)t] = p_cur[(size_t)t]*blend_keep + p_vec[(size_t)t]*blend_new;
            return out;
        }
        std::vector<double> coeff(m);
        double ssum = 0.0;
        for (size_t i=0;i<m;++i){ coeff[i] = sol((Eigen::Index)i); ssum += coeff[i]; }
        if (std::abs(ssum) > 0.0) for (auto& c : coeff) c /= ssum;
        const double cmax = std::abs(*std::max_element(coeff.begin(), coeff.end(),
                              [](double a, double b){return std::abs(a)<std::abs(b);} ));
        const bool unstable = !std::isfinite(cmax) || cmax > coeff_cap;

        std::vector<cxd> mix(p_vec.size(), cxd(0.0,0.0));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0; t<(long long)mix.size(); ++t) {
            cxd acc(0.0,0.0);
            for (size_t i=0;i<m;++i) {
                size_t idx = (count - m + i) % max_vecs;
                acc += ps_flat[idx][(size_t)t] * coeff[i];
            }
            mix[(size_t)t] = acc;
        }
        if (unstable) {
            std::vector<cxd> out(mix.size());
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)mix.size();++t) out[(size_t)t] = p_cur[(size_t)t]*blend_keep + mix[(size_t)t]*blend_new;
            return out;
        }
        return mix;
    }
};

struct EdiisState {
    std::vector<std::vector<cxd>> ps_flat, fs_flat; std::vector<double> energy;
    size_t count=0, max_vecs;
    explicit EdiisState(size_t m): max_vecs(m) { ps_flat.reserve(m); fs_flat.reserve(m); energy.reserve(m); }
    static std::vector<cxd> flatten(const std::vector<cxd>& a) { return a; }

    static std::vector<double> project_simplex(std::vector<double> v) {
        // Duchi et al. projection onto simplex {x>=0, sum=1}
        auto u = v; std::sort(u.begin(), u.end(), std::greater<>());
        double css = 0.0; size_t rho = 0;
        for (size_t i=0;i<u.size();++i){ css += u[i]; const double t = (css - 1.0) / (i+1); if (u[i] - t > 0.0) rho = i; }
        const double tau = (std::accumulate(u.begin(), u.begin()+rho+1, 0.0) - 1.0) / (rho + 1);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long i=0;i<(long long)v.size();++i) v[(size_t)i] = std::max(0.0, v[(size_t)i] - tau);
        return v;
    }

    // Weighted EDIIS with adaptive PG and early stop
    std::pair<std::vector<cxd>, bool> update(const std::vector<cxd>& p,
                                             const std::vector<cxd>& f,
                                             double e,
                                             const std::vector<double>& weights,
                                             size_t nk1, size_t nk2, size_t d,
                                             size_t max_iter_qp,
                                             double pg_tol = 1e-9) {
        if (max_vecs < 2) return {p, true};
        if (count < max_vecs) { ps_flat.push_back(p); fs_flat.push_back(f); energy.push_back(e); }
        else { size_t idx = count % max_vecs; ps_flat[idx]=p; fs_flat[idx]=f; energy[idx]=e; }
        ++count; const size_t m = std::min(count, max_vecs); if (m < 2) return {p, true};

        std::vector<size_t> idxs(m);
        for (size_t i=0;i<m;++i) idxs[i] = (count - m + i) % max_vecs;

        // Build g and M with WEIGHTED inner products
        Eigen::VectorXd g(m);
        Eigen::MatrixXd M(m,m);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long ii=0; ii<(long long)m; ++ii){
            const size_t i = (size_t)ii;
            const size_t ii_idx = idxs[i];
            const double tr_w = weighted_inner_blocks(ps_flat[ii_idx], fs_flat[ii_idx], weights, nk1, nk2, d);
            g((Eigen::Index)i) = energy[ii_idx] - 0.5*tr_w;
            for (size_t j=0;j<m;++j) {
                const size_t jj_idx = idxs[j];
                M((Eigen::Index)i,(Eigen::Index)j) = weighted_inner_blocks(ps_flat[ii_idx], fs_flat[jj_idx], weights, nk1, nk2, d);
            }
        }

        // Objective: phi(c) = g^T c + 0.5 c^T M c  on simplex {c >= 0, sum=1}
        auto phi = [&](const Eigen::VectorXd& c){ return g.dot(c) + 0.5 * c.transpose() * M * c; };

        // Start at uniform coefficients
        std::vector<double> cvec(m, 1.0/m);
        Eigen::VectorXd c = Eigen::Map<Eigen::VectorXd>(cvec.data(), (Eigen::Index)m);

        // Initial step size from a crude Lipschitz estimate
        double max_abs_m = 0.0;
        for (size_t i=0;i<m;++i) for (size_t j=0;j<m;++j)
            max_abs_m = std::max(max_abs_m, std::abs(M((Eigen::Index)i,(Eigen::Index)j)));
        double gamma0 = 1.0 / (max_abs_m + 1.0);
        const double armijo = 1e-4;

        for (size_t it=0; it<max_iter_qp; ++it) {
            // Projected gradient norm (early stop)
            Eigen::VectorXd grad = g + M * c;
            // Build a std::vector from (c - grad) safely, then project onto simplex
            Eigen::VectorXd diff_pg = (c - grad).eval();
            std::vector<double> tmp_pg(diff_pg.data(), diff_pg.data() + m);
            auto proj_pg = project_simplex(std::move(tmp_pg));
            Eigen::VectorXd c_pg = Eigen::Map<const Eigen::VectorXd>(proj_pg.data(), (Eigen::Index)m);
            double pg_norm = (c - c_pg).norm();
            if (pg_norm < pg_tol) break;

            // Backtracking line search with projection
            double gamma = gamma0;
            const double phi_c = phi(c);
            bool accepted = false;
            for (int bt=0; bt<12; ++bt) {
                // Trial step and projection
                Eigen::VectorXd step = (c - gamma * grad).eval();
                std::vector<double> tmp_trial(step.data(), step.data() + m);
                auto proj_trial = project_simplex(std::move(tmp_trial));
                Eigen::VectorXd c_trial = Eigen::Map<const Eigen::VectorXd>(proj_trial.data(), (Eigen::Index)m);
                const double phi_trial = phi(c_trial);
                const double decr = armijo * gamma * grad.squaredNorm(); // conservative Armijo surrogate
                if (phi_trial <= phi_c - decr) { c = std::move(c_trial); accepted = true; break; }
                gamma *= 0.5;
            }
            if (!accepted) { // still move (small step) to avoid stalling
                Eigen::VectorXd step = (c - 0.2*gamma0 * grad).eval();
                std::vector<double> tmp_small(step.data(), step.data() + m);
                auto proj_small = project_simplex(std::move(tmp_small));
                c = Eigen::Map<const Eigen::VectorXd>(proj_small.data(), (Eigen::Index)m);
            }
        }

        // Build output density
        std::vector<cxd> out(p.size(), cxd(0,0));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)out.size();++t){
            cxd acc(0,0);
            for (size_t i=0;i<m;++i){ const size_t ii_idx = idxs[i]; acc += ps_flat[ii_idx][(size_t)t]*c((Eigen::Index)i); }
            out[(size_t)t] = acc;
        }
        return {out, false};
    }
};

// ---------------- Broyden mixing ----------------
struct BroydenState {
    std::vector<std::vector<cxd>> s_hist; // previous P differences (flattened)
    std::vector<std::vector<cxd>> y_hist; // previous residual differences (flattened)
    size_t count = 0;
    size_t max_vecs;
    size_t n_flat;

    BroydenState(size_t m, size_t n)
        : s_hist(m, std::vector<cxd>(n, cxd(0,0))),
          y_hist(m, std::vector<cxd>(n, cxd(0,0))),
          max_vecs(m), n_flat(n) {}

    void reset() {
        for (auto& v : s_hist) std::fill(v.begin(), v.end(), cxd(0,0));
        for (auto& v : y_hist) std::fill(v.begin(), v.end(), cxd(0,0));
        count = 0;
    }

    static cxd dot(const std::vector<cxd>& a, const std::vector<cxd>& b) {
        Eigen::Map<const Eigen::VectorXcd> va(a.data(), (Eigen::Index)a.size());
        Eigen::Map<const Eigen::VectorXcd> vb(b.data(), (Eigen::Index)b.size());
        return va.dot(vb); // conj(a)^T b
    }

    std::pair<BroydenState, std::vector<cxd>> update(const std::vector<cxd>& P_flat,
                                                     const std::vector<cxd>& resid_flat,
                                                     double alpha = 1.3) const {
        BroydenState st = *this;
        if (st.max_vecs == 0 || st.n_flat == 0) return {st, P_flat};

        if (st.count == 0) {
            // seed history; return unchanged (caller may apply extra descent step)
            st.s_hist[0] = P_flat;
            st.y_hist[0] = resid_flat;
            st.count = 1;
            return {st, P_flat};
        }

        const size_t k = st.count;
        const size_t idx = k % st.max_vecs;

        std::vector<cxd> s_k(st.n_flat), y_k(st.n_flat);
        const auto& P_prev = st.s_hist[(k-1) % st.max_vecs];
        const auto& R_prev = st.y_hist[(k-1) % st.max_vecs];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long i=0;i<(long long)st.n_flat;++i) {
            s_k[(size_t)i] = P_flat[(size_t)i]     - P_prev[(size_t)i];
            y_k[(size_t)i] = resid_flat[(size_t)i] - R_prev[(size_t)i];
        }
        st.s_hist[idx] = s_k;
        st.y_hist[idx] = y_k;
        st.count = k + 1;

        // L-BFGS two-loop
        std::vector<cxd> q = resid_flat;
        std::vector<cxd> alphas(st.max_vecs, cxd(0,0));

        for (size_t ii=0; ii<st.max_vecs; ++ii) {
            const size_t i = st.max_vecs - 1 - ii;
            const cxd yi_si = dot(st.y_hist[i], st.s_hist[i]);
            const cxd rho   = (std::abs(yi_si) > 1e-18) ? (cxd(1.0,0.0) / yi_si) : cxd(0.0,0.0);
            const cxd a_i   = rho * dot(st.s_hist[i], q);
            alphas[i] = a_i;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)st.n_flat;++t) q[(size_t)t] -= a_i * st.y_hist[i][(size_t)t];
        }

        const cxd yk_yk = dot(st.y_hist[idx], st.y_hist[idx]);
        const cxd sk_yk = dot(st.s_hist[idx], st.y_hist[idx]);
        const cxd gamma = (std::abs(yk_yk) > 1e-18) ? (sk_yk / yk_yk) : cxd(0.0,0.0);

        std::vector<cxd> r(q.size());
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)r.size();++t) r[(size_t)t] = gamma * q[(size_t)t];

        for (size_t i=0;i<st.max_vecs;++i) {
            const cxd yi_si = dot(st.y_hist[i], st.s_hist[i]);
            const cxd rho   = (std::abs(yi_si) > 1e-18) ? (cxd(1.0,0.0) / yi_si) : cxd(0.0,0.0);
            const cxd beta  = rho * dot(st.y_hist[i], r);
            const cxd coeff = alphas[i] - beta;
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)st.n_flat;++t) r[(size_t)t] += st.s_hist[i][(size_t)t] * coeff;
        }

        // Step: P + alpha * r  (direction sign handled by preconditioner choice)
        std::vector<cxd> P_out(st.n_flat);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)st.n_flat;++t) P_out[(size_t)t] = P_flat[(size_t)t] + cxd(alpha,0.0)*r[(size_t)t];
        return {st, P_out};
    }
};

// Orbital preconditioner on commutator (minus sign included)
inline void orbital_preconditioner(const size_t nk1, const size_t nk2, const size_t d,
                                   const std::vector<cxd>& F, const std::vector<cxd>& comm,
                                   std::vector<cxd>& out, double delta=1e-3) {
    out.resize(F.size());
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (long long k1=0;k1<(long long)nk1;++k1)
        for (long long k2=0;k2<(long long)nk2;++k2) {
            const size_t base = (static_cast<size_t>(k1)*nk2 + static_cast<size_t>(k2)) * d * d;
            Eigen::Map<const MatC> Fk(&F[base], d, d);
            Eigen::SelfAdjointEigenSolver<MatC> es(Fk);
            if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed (precond)");
            const MatC C = es.eigenvectors();
            const Eigen::VectorXd eps = es.eigenvalues().real();

            // denom(i,j) = (eps_i - eps_j) + delta
            Eigen::MatrixXd denom = eps.replicate(1, (Eigen::Index)d) - eps.transpose().replicate((Eigen::Index)d, 1);
            denom.array() += delta;

            Eigen::Map<const MatC> Rk(&comm[base], d, d);
            MatC Rmo  = C.adjoint() * Rk * C;
            MatC Rtil = - Rmo.cwiseQuotient(denom.cast<cxd>()); // minus for ΔP ≈ -C/(ε_i-ε_j)

            Eigen::Map<MatC> outk(&out[base], d, d);
            outk.noalias() = C * Rtil * C.adjoint();
        }
}

} // namespace hf

