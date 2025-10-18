// mixers.hpp - DIIS/EDIIS/Broyden mixers and orbital preconditioner
#pragma once

#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <span>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "cpp_hf/utils.hpp"
#include "cpp_hf/prof.hpp"
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
// Loop-free via (nblocks × block) row-major maps and row-wise reduction.
inline double weighted_inner_blocks(const std::vector<cxd>& A,
                                    const std::vector<cxd>& B,
                                    std::span<const double> w,
                                    size_t nk1, size_t nk2, size_t d) {
    const size_t nblocks = nk1 * nk2;
    const size_t block   = d * d;

    using RowMatC = Eigen::Matrix<cxd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<const RowMatC> Am(A.data(), (Eigen::Index)nblocks, (Eigen::Index)block);
    Eigen::Map<const RowMatC> Bm(B.data(), (Eigen::Index)nblocks, (Eigen::Index)block);
    Eigen::Map<const Eigen::VectorXd> W(w.data(), (Eigen::Index)nblocks);

    // Per-block Re⟨A_k,B_k⟩, weight, then global sum.
    return ( (Am.conjugate().cwiseProduct(Bm))   // conj(A) ⊙ B
               .rowwise().sum()                 // → VectorXcd(nblocks)
               .real()                          // → VectorXd
               .cwiseProduct(W) )               // weight
           .sum();
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
            Eigen::Map<      Eigen::ArrayXcd> O(out.data(),   (Eigen::Index)out.size());
            Eigen::Map<const Eigen::ArrayXcd> Pc(p_cur.data(),(Eigen::Index)p_cur.size());
            Eigen::Map<const Eigen::ArrayXcd> Pn(p_vec.data(),(Eigen::Index)p_vec.size());
            O = blend_keep*Pc + blend_new*Pn;
            return out;
        }

        if (count < max_vecs) { residuals.push_back(comm); ps_flat.push_back(p_vec); }
        else { size_t idx = count % max_vecs; residuals[idx] = comm; ps_flat[idx] = p_vec; }
        ++count; const size_t m = std::min(count, max_vecs);
        if (m < 2) return p_vec;

        const size_t n_aug = m + 1;
        Eigen::MatrixXd BA = Eigen::MatrixXd::Zero(n_aug, n_aug);

        // Build upper triangle of the Gram (real ⟨r_i,r_j⟩), mirror for symmetry.
        for (size_t i = 0; i < m; ++i) {
            const auto& ri = residuals[(count - m + i) % max_vecs];
            for (size_t j = 0; j <= i; ++j) {
                const auto& rj = residuals[(count - m + j) % max_vecs];
                const double gij = real_inner(ri, rj);
                BA((Eigen::Index)i,(Eigen::Index)j) = gij;
                if (i != j) BA((Eigen::Index)j,(Eigen::Index)i) = gij;
            }
        }
        // Regularize diagonal, fill constraint row/col with Eigen block ops.
        BA.topLeftCorner((Eigen::Index)m,(Eigen::Index)m).diagonal().array() += eps_reg;
        BA.block(0,(Eigen::Index)m,(Eigen::Index)m,1).setConstant(-1.0);
        BA.block((Eigen::Index)m,0,1,(Eigen::Index)m).setConstant(-1.0);

        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_aug); rhs(n_aug-1) = -1.0;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(BA);
        if (ldlt.info() != Eigen::Success) {
            std::vector<cxd> out(p_vec.size());
            Eigen::Map<      Eigen::ArrayXcd> O(out.data(),   (Eigen::Index)out.size());
            Eigen::Map<const Eigen::ArrayXcd> Pc(p_cur.data(),(Eigen::Index)p_cur.size());
            Eigen::Map<const Eigen::ArrayXcd> Pn(p_vec.data(),(Eigen::Index)p_vec.size());
            O = blend_keep*Pc + blend_new*Pn;
            return out;
        }
        auto sol = ldlt.solve(rhs);
        if (ldlt.info() != Eigen::Success) {
            std::vector<cxd> out(p_vec.size());
            Eigen::Map<      Eigen::ArrayXcd> O(out.data(),   (Eigen::Index)out.size());
            Eigen::Map<const Eigen::ArrayXcd> Pc(p_cur.data(),(Eigen::Index)p_cur.size());
            Eigen::Map<const Eigen::ArrayXcd> Pn(p_vec.data(),(Eigen::Index)p_vec.size());
            O = blend_keep*Pc + blend_new*Pn;
            return out;
        }

        // Coefficients (Eigen ops; no manual loops)
        Eigen::VectorXd coeff = sol.head((Eigen::Index)m);
        const double ssum = coeff.sum();
        if (std::abs(ssum) > 0.0) coeff.array() /= ssum;
        const double cmax = coeff.cwiseAbs().maxCoeff();
        const bool unstable = !std::isfinite(cmax) || cmax > coeff_cap;

        // Vectorized mix: Out = Σ_i coeff[i] * P_i  (loop only over small m)
        const Eigen::Index nflat = (Eigen::Index)p_vec.size();
        Eigen::ArrayXcd Out = Eigen::ArrayXcd::Zero(nflat);
        for (size_t i=0;i<m;++i){
            const size_t idx = (count - m + i) % max_vecs;
            const double ci  = coeff((Eigen::Index)i);
            if (ci == 0.0) continue;
            Eigen::Map<const Eigen::ArrayXcd> Pi(ps_flat[idx].data(), nflat);
            Out += ci * Pi;
        }
        std::vector<cxd> mix((size_t)nflat);
        Eigen::Map<Eigen::ArrayXcd>(mix.data(), nflat) = Out;

        if (unstable) {
            std::vector<cxd> out(mix.size());
            Eigen::Map<      Eigen::ArrayXcd> O(out.data(),   (Eigen::Index)out.size());
            Eigen::Map<const Eigen::ArrayXcd> Pc(p_cur.data(),(Eigen::Index)p_cur.size());
            Eigen::Map<const Eigen::ArrayXcd> Mx(mix.data(),  (Eigen::Index)mix.size());
            O = blend_keep*Pc + blend_new*Mx;
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
        const std::size_t n = v.size();
        if (n == 0) return v;

        // 1) Sort copy in descending order
        std::vector<double> u = v;
        std::sort(u.begin(), u.end(), std::greater<double>());

        // 2) Prefix sums of u
        std::vector<double> css(n);
        std::partial_sum(u.begin(), u.end(), css.begin());

        // 3) Find rho = max { i : u[i] - (css[i] - 1)/(i+1) > 0 }
        std::size_t rho = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double t = (css[i] - 1.0) / static_cast<double>(i + 1);
            if (u[i] - t > 0.0) rho = i;
        }

        // 4) Threshold tau and vectorized projection: max(v - tau, 0)
        const double tau = (css[rho] - 1.0) / static_cast<double>(rho + 1);
        Eigen::Map<Eigen::VectorXd> V(v.data(), static_cast<Eigen::Index>(n));
        V.array() = (V.array() - tau).max(0.0);

        return v;
    }


    // Weighted EDIIS with adaptive PG and early stop
    std::pair<std::vector<cxd>, bool> update(const std::vector<cxd>& p,
                                             const std::vector<cxd>& f,
                                             double e,
                                             std::span<const double> weights,
                                             size_t nk1, size_t nk2, size_t d,
                                             size_t max_iter_qp,
                                             double pg_tol = 1e-9) {
        if (max_vecs < 2) return {p, true};
        if (count < max_vecs) { ps_flat.push_back(p); fs_flat.push_back(f); energy.push_back(e); }
        else { size_t idx = count % max_vecs; ps_flat[idx]=p; fs_flat[idx]=f; energy[idx]=e; }
        ++count; const size_t m = std::min(count, max_vecs); if (m < 2) return {p, true};

        std::vector<size_t> idxs(m);
        for (size_t i=0;i<m;++i) idxs[i] = (count - m + i) % max_vecs;

        // Build g and M with WEIGHTED inner products (loop-free kernel)
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
        const double gamma0 = 1.0 / (M.cwiseAbs().maxCoeff() + 1.0);
        const double armijo = 1e-4;

        for (size_t it=0; it<max_iter_qp; ++it) {
            // Projected gradient norm (early stop)
            Eigen::VectorXd grad = g + M * c;
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
                Eigen::VectorXd step = (c - gamma * grad).eval();
                std::vector<double> tmp_trial(step.data(), step.data() + m);
                auto proj_trial = project_simplex(std::move(tmp_trial));
                Eigen::VectorXd c_trial = Eigen::Map<const Eigen::VectorXd>(proj_trial.data(), (Eigen::Index)m);
                const double phi_trial = phi(c_trial);
                const double decr = armijo * gamma * grad.squaredNorm();
                if (phi_trial <= phi_c - decr) { c = std::move(c_trial); accepted = true; break; }
                gamma *= 0.5;
            }
            if (!accepted) {
                Eigen::VectorXd step = (c - 0.2*gamma0 * grad).eval();
                std::vector<double> tmp_small(step.data(), step.data() + m);
                auto proj_small = project_simplex(std::move(tmp_small));
                c = Eigen::Map<const Eigen::VectorXd>(proj_small.data(), (Eigen::Index)m);
            }
        }

        // Build output density: out = Σ_i c_i * P_i  (vectorized AXPY over history; m is small)
        std::vector<cxd> out(p.size(), cxd(0,0));
        Eigen::Map<Eigen::ArrayXcd> Out(out.data(), (Eigen::Index)out.size());
        Out.setZero();
        for (size_t i=0;i<m;++i){
            const size_t ii_idx = idxs[i];
            const double ci = c((Eigen::Index)i);
            if (ci == 0.0) continue;
            Eigen::Map<const Eigen::ArrayXcd> Pi(ps_flat[ii_idx].data(), (Eigen::Index)out.size());
            Out += ci * Pi;
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

        // s_k = P_flat - P_prev,  y_k = resid_flat - R_prev  (vectorized)
        std::vector<cxd> s_k(st.n_flat), y_k(st.n_flat);
        {
            Eigen::Map<const Eigen::ArrayXcd> Pn(P_flat.data(),                (Eigen::Index)st.n_flat);
            Eigen::Map<const Eigen::ArrayXcd> Po(st.s_hist[(k-1)%st.max_vecs].data(), (Eigen::Index)st.n_flat);
            Eigen::Map<const Eigen::ArrayXcd> Rn(resid_flat.data(),            (Eigen::Index)st.n_flat);
            Eigen::Map<const Eigen::ArrayXcd> Ro(st.y_hist[(k-1)%st.max_vecs].data(), (Eigen::Index)st.n_flat);
            Eigen::Map<Eigen::ArrayXcd>       Sk(s_k.data(), (Eigen::Index)st.n_flat);
            Eigen::Map<Eigen::ArrayXcd>       Yk(y_k.data(), (Eigen::Index)st.n_flat);
            Sk = Pn - Po;
            Yk = Rn - Ro;
        }
        st.s_hist[idx] = s_k;
        st.y_hist[idx] = y_k;
        st.count = k + 1;

        // L-BFGS two-loop (vectorized axpy updates)
        std::vector<cxd> q = resid_flat;
        const size_t hist = st.max_vecs; // keep semantics identical to original
        for (size_t ii=0; ii<hist; ++ii) {
            const size_t i = hist - 1 - ii;
            const cxd yi_si = dot(st.y_hist[i], st.s_hist[i]);
            const cxd rho   = (std::abs(yi_si) > 1e-18) ? cxd(1.0,0.0) / yi_si : cxd(0.0,0.0);
            const cxd a_i   = rho * dot(st.s_hist[i], q);

            Eigen::Map<Eigen::ArrayXcd>       Q(q.data(), (Eigen::Index)st.n_flat);
            Eigen::Map<const Eigen::ArrayXcd> Yi(st.y_hist[i].data(), (Eigen::Index)st.n_flat);
            Q -= a_i * Yi;
        }

        const cxd yk_yk = dot(st.y_hist[idx], st.y_hist[idx]);
        const cxd sk_yk = dot(st.s_hist[idx], st.y_hist[idx]);
        const cxd gamma = (std::abs(yk_yk) > 1e-18) ? (sk_yk / yk_yk) : cxd(0.0,0.0);

        std::vector<cxd> r(q.size());
        {
            Eigen::Map<const Eigen::ArrayXcd> Q(q.data(), (Eigen::Index)st.n_flat);
            Eigen::Map<Eigen::ArrayXcd>       R(r.data(), (Eigen::Index)st.n_flat);
            R = gamma * Q;
        }

        for (size_t i=0;i<hist;++i) {
            const cxd yi_si = dot(st.y_hist[i], st.s_hist[i]);
            const cxd rho   = (std::abs(yi_si) > 1e-18) ? cxd(1.0,0.0) / yi_si : cxd(0.0,0.0);
            const cxd beta  = rho * dot(st.y_hist[i], r);
            const cxd a_i   = (std::abs(yi_si) > 1e-18) ? (cxd(1.0,0.0) / yi_si) * dot(st.s_hist[i], q) : cxd(0.0,0.0);

            Eigen::Map<Eigen::ArrayXcd>       Rv(r.data(), (Eigen::Index)st.n_flat);
            Eigen::Map<const Eigen::ArrayXcd> Si(st.s_hist[i].data(), (Eigen::Index)st.n_flat);
            // Classic two-loop: r += s_i * (alpha_i - beta)
            Rv += Si * (a_i - beta);
        }

        // Step: P_out = P_flat + alpha * r
        std::vector<cxd> P_out(st.n_flat);
        {
            Eigen::Map<const Eigen::ArrayXcd> Pn(P_flat.data(), (Eigen::Index)st.n_flat);
            Eigen::Map<const Eigen::ArrayXcd> Rv(r.data(),      (Eigen::Index)st.n_flat);
            Eigen::Map<Eigen::ArrayXcd>       Po(P_out.data(),  (Eigen::Index)st.n_flat);
            Po = Pn + alpha * Rv;
        }
        return {st, P_out};
    }
};

// Orbital preconditioner on commutator (minus sign included)
inline void orbital_preconditioner(const size_t nk1, const size_t nk2, const size_t d,
                                   const std::vector<cxd>& F, const std::vector<cxd>& comm,
                                   std::vector<cxd>& out, double delta=1e-3) {
    out.resize(F.size());
    ::for_k(nk1, nk2, [&](std::size_t k1, std::size_t k2){
        const size_t base = (k1*nk2 + k2) * d * d;
        Eigen::Map<const MatC> Fk(&F[base], d, d);
        Eigen::SelfAdjointEigenSolver<MatC> es(Fk);
        if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed (precond)");
        const MatC C = es.eigenvectors();
        const Eigen::VectorXd eps = es.eigenvalues().real();

        Eigen::Map<const MatC> Rk(&comm[base], d, d);

        // Work in MO basis: Rmo = C^† R C  (materialize once)
        const MatC Rmo = C.adjoint() * Rk * C;

        // Elementwise divide by (ε_i - ε_j + δ) using broadcasted expression,
        // avoiding an explicit denom matrix.
        const auto er = eps.transpose(); // RowVectorXd
        const auto ec = eps;             // VectorXd
        const auto denom_expr =
            (ec.rowwise().replicate((Eigen::Index)d)
           - er.colwise().replicate((Eigen::Index)d)).array()
           + delta;

        const MatC Rtil = - Rmo.cwiseQuotient(denom_expr.matrix().cast<cxd>());

        Eigen::Map<MatC> outk(&out[base], d, d);
        outk.noalias() = C * Rtil * C.adjoint();
    });
}

// ---------------- Mixing Orchestrator (EDIIS/CDIIS/Broyden) ----------------
enum class MixPhase { EDIIS, CDIIS, BROYDEN };

struct MixingConfig {
    double to_cdiis = 0.0;          // threshold to switch from EDIIS -> CDIIS
    double to_broyden = 0.0;        // threshold to switch from CDIIS -> Broyden
    double cdiis_blend_keep = 0.5;  // CDIIS blend keep weight
    double cdiis_blend_new  = 0.5;  // CDIIS blend new weight
    double mixing_alpha = 0.5;      // Broyden step scaling
    double precond_delta = 5.0e-3;  // Preconditioner denominator shift
};

struct MixingState {
    MixPhase last_phase = MixPhase::EDIIS;
    BroydenState bro_state;
    DiisState cdiis_state;
    EdiisState ediis_state;

    MixingState(std::size_t diis_size, std::size_t n_flat)
        : bro_state(diis_size, n_flat), cdiis_state(diis_size), ediis_state(diis_size) {}
    void reset_broyden_if_switched(MixPhase phase_now) {
        if (phase_now != last_phase) bro_state.reset();
        last_phase = phase_now;
    }
};

inline std::vector<cxd> mix_step(
    const std::vector<cxd>& P_cur,
    const std::vector<cxd>& P_new,
    const std::vector<cxd>& F_new,
    const std::vector<cxd>& comm,
    double e_new,
    double comm_rms,
    MixingState& st,
    const MixingConfig& cfg,
    std::span<const double> weights,
    std::size_t nk1, std::size_t nk2, std::size_t d,
    const std::function<void(const std::vector<cxd>&, const std::vector<cxd>&, std::vector<cxd>&, double)>& precondition_cb)
{
    const MixPhase phase_now = (comm_rms > cfg.to_cdiis) ? MixPhase::EDIIS
                               : (comm_rms > cfg.to_broyden) ? MixPhase::CDIIS
                               : MixPhase::BROYDEN;

    std::vector<cxd> P_mix;

    if (phase_now == MixPhase::EDIIS) {
        {
            HF_PROFILE_SCOPE("ediis_update");
            auto ediis_result = st.ediis_state.update(P_new, F_new, e_new,
                                                      weights, nk1, nk2, d,
                                                      /*max_iter_qp=*/20, /*pg_tol=*/1e-7);
            P_mix = std::move(ediis_result.first);
        }
    }
    else if (phase_now == MixPhase::CDIIS) {
        HF_PROFILE_SCOPE("cdiis_update");
        P_mix = st.cdiis_state.update_cdiis(P_new, comm, P_cur,
                                            /*coeff_cap=*/5.0, /*eps_reg=*/1e-12,
                                            /*blend_keep=*/cfg.cdiis_blend_keep,
                                            /*blend_new=*/cfg.cdiis_blend_new);
    }
    else { // MixPhase::BROYDEN
        std::vector<cxd> comm_pc;
        {
            HF_PROFILE_SCOPE("precondition_commutator");
            precondition_cb(F_new, comm, comm_pc, cfg.precond_delta);
        }

        const std::size_t bro_count_before = st.bro_state.count;
        std::vector<cxd> Praw;
        {
            HF_PROFILE_SCOPE("broyden_update");
            auto upd = st.bro_state.update(P_new, comm_pc, cfg.mixing_alpha);
            st.bro_state = std::move(upd.first);
            Praw = std::move(upd.second);
        }

        if (bro_count_before == 0) {
            const double beta = 0.35;
            P_mix.resize(P_cur.size());
            Eigen::Map<      Eigen::ArrayXcd> Pm(P_mix.data(), (Eigen::Index)P_mix.size());
            Eigen::Map<const Eigen::ArrayXcd> Pc(P_cur.data(), (Eigen::Index)P_cur.size());
            Eigen::Map<const Eigen::ArrayXcd> Rc(comm_pc.data(), (Eigen::Index)comm_pc.size());
            Pm = Pc - beta * Rc;
            const double w_keep = 0.7, w_new = 0.3;
            Pm = w_keep*Pc + w_new*Pm;
        } else {
            P_mix = std::move(Praw);
        }
    }

    st.reset_broyden_if_switched(phase_now);
    return P_mix;
}

} // namespace hf
