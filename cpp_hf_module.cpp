// hf_cpp_module.cpp (multicore-optimized)
// C++17 + pybind11 + Eigen + FFTW (guru, batched 2D) + optional Boost (toms748)
// Layout: (nk1, nk2, d, d) row-major (C-order)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

extern "C" {
#include <fftw3.h>
}

#include <vector>
#include <complex>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>
#include <mutex>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ---- Boost root solver (optional, but recommended) ----
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/math/tools/roots.hpp>

namespace py = pybind11;
using cxd  = std::complex<double>;
using MatC = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vecd = Eigen::VectorXd;

static inline size_t offset(size_t nk2, size_t d, size_t k1, size_t k2, size_t i, size_t j) {
    return ((k1 * nk2 + k2) * d + i) * d + j; // C-order
}

static inline double fermi(double x, double T) {
    const double tt = std::max(1e-12, std::abs(T));
    const double y  = x / tt;
    if (y >=  40.0) return 0.0;
    if (y <= -40.0) return 1.0;
    return 1.0 / (1.0 + std::exp(y));
}

static inline double real_inner(const std::vector<cxd>& a, const std::vector<cxd>& b) {
    double s = 0.0; const size_t n = a.size();
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:s) schedule(static)
#endif
    for (long long i = 0; i < (long long)n; ++i) s += (a[i] * std::conj(b[i])).real();
    return s;
}

// ---------------- FFTW Batched 2D (guru) ----------------
struct FftwBatched2D {
    fftw_plan fwd = nullptr;
    fftw_plan bwd = nullptr;
    size_t nk1{}, nk2{}, d{};
    cxd* plan_buf = nullptr;  // planning buffer (full-sized)
    size_t n_tot{};
    int nthreads{1};

    static void init_threads_once() {
#if defined(FFTW3_THREADS)
        static std::once_flag once;
        std::call_once(once, []{
            fftw_init_threads();
        });
#endif
    }

    static int choose_threads() {
#if defined(_OPENMP)
        return std::max(1, omp_get_max_threads());
#else
        return 1;
#endif
    }

    FftwBatched2D(size_t nk1_, size_t nk2_, size_t d_)
        : nk1(nk1_), nk2(nk2_), d(d_), n_tot(nk1_*nk2_*d_*d_), nthreads(choose_threads()) {
        plan_buf = reinterpret_cast<cxd*>(fftw_malloc(sizeof(cxd) * n_tot));
        if (!plan_buf) throw std::bad_alloc{};

        init_threads_once();
#if defined(FFTW3_THREADS)
        fftw_plan_with_nthreads(nthreads);
#endif

        // Strides for (nk1, nk2, d, d), C-order
        fftw_iodim64 dims[2];
        dims[0].n  = static_cast<long long>(nk2); dims[0].is = static_cast<long long>(d*d);          dims[0].os = dims[0].is;
        dims[1].n  = static_cast<long long>(nk1); dims[1].is = static_cast<long long>(nk2*d*d);     dims[1].os = dims[1].is;
        fftw_iodim64 how[2];
        how[0].n   = static_cast<long long>(d);   how[0].is  = static_cast<long long>(d);           how[0].os  = how[0].is; // i
        how[1].n   = static_cast<long long>(d);   how[1].is  = 1;                                   how[1].os  = 1;         // j

        fwd = fftw_plan_guru64_dft(2, dims, 2, how,
                reinterpret_cast<fftw_complex*>(plan_buf),
                reinterpret_cast<fftw_complex*>(plan_buf),
                FFTW_FORWARD, FFTW_MEASURE);
        if (!fwd) throw std::runtime_error("FFTW plan_guru64_dft forward failed");

        bwd = fftw_plan_guru64_dft(2, dims, 2, how,
                reinterpret_cast<fftw_complex*>(plan_buf),
                reinterpret_cast<fftw_complex*>(plan_buf),
                FFTW_BACKWARD, FFTW_MEASURE);
        if (!bwd) { fftw_destroy_plan(fwd); throw std::runtime_error("FFTW plan_guru64_dft backward failed"); }
    }

    void forward(cxd* buf) const  { fftw_execute_dft(fwd, reinterpret_cast<fftw_complex*>(buf), reinterpret_cast<fftw_complex*>(buf)); }
    void backward(cxd* buf) const { fftw_execute_dft(bwd, reinterpret_cast<fftw_complex*>(buf), reinterpret_cast<fftw_complex*>(buf)); }

    ~FftwBatched2D() { if (fwd) fftw_destroy_plan(fwd); if (bwd) fftw_destroy_plan(bwd); if (plan_buf) fftw_free(plan_buf); }
};

// ---------------- DIIS / EDIIS ----------------
struct DiisState {
    std::vector<std::vector<cxd>> residuals; // ring buffer
    std::vector<std::vector<cxd>> ps_flat;   // ring buffer
    size_t count = 0;
    size_t max_vecs;

    explicit DiisState(size_t m): max_vecs(m) { residuals.reserve(m); ps_flat.reserve(m); }

    static std::vector<cxd> flatten(const std::vector<cxd>& a) { return a; }

    std::vector<cxd> update_cdiis(const std::vector<cxd>& p_vec,   // vector stored (here: P_new)
                                  const std::vector<cxd>& comm,    // residual matching p_vec
                                  const std::vector<cxd>& p_cur,   // current P (for fallback blend)
                                  double coeff_cap, double eps_reg,
                                  double blend_keep, double blend_new) {
        if (max_vecs < 2) { // fallback: simple blend
            std::vector<cxd> out(p_vec.size());
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)out.size();++t) out[t] = p_cur[t]*blend_keep + p_vec[t]*blend_new;
            return out;
        }

        if (count < max_vecs) { residuals.push_back(comm); ps_flat.push_back(p_vec); }
        else { size_t idx = count % max_vecs; residuals[idx] = comm; ps_flat[idx] = p_vec; }
        ++count; const size_t m = std::min(count, max_vecs);
        if (m < 2) return p_vec;

        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(m, m);
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long i = 0; i < (long long)m; ++i)
            for (long long j = 0; j < (long long)m; ++j) {
                const auto& ri = residuals[(count - m + (size_t)i) % max_vecs];
                const auto& rj = residuals[(count - m + (size_t)j) % max_vecs];
                B(i,j) = real_inner(ri, rj);
            }

        const size_t n_aug = m + 1;
        Eigen::MatrixXd BA = Eigen::MatrixXd::Zero(n_aug, n_aug);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) BA(i,j) = B(i,j);
            BA(i,m) = -1.0; BA(m,i) = -1.0; BA(i,i) += eps_reg;
        }
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_aug); rhs(n_aug-1) = -1.0;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(BA);
        if (ldlt.info() != Eigen::Success) {
            std::vector<cxd> out(p_vec.size());
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)out.size();++t) out[t] = p_cur[t]*blend_keep + p_vec[t]*blend_new;
            return out;
        }
        auto sol = ldlt.solve(rhs);
        if (ldlt.info() != Eigen::Success) {
            std::vector<cxd> out(p_vec.size());
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0;t<(long long)out.size();++t) out[t] = p_cur[t]*blend_keep + p_vec[t]*blend_new;
            return out;
        }
        std::vector<double> coeff(m);
        double ssum = 0.0;
        for (size_t i=0;i<m;++i){ coeff[i] = sol((Eigen::Index)i); ssum += coeff[i]; }
        if (std::abs(ssum) > 0.0) for (auto& c : coeff) c /= ssum;
        const double cmax = std::abs(*std::max_element(coeff.begin(), coeff.end(), [](double a, double b){return std::abs(a)<std::abs(b);} ));
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

    std::pair<std::vector<cxd>, bool> update(const std::vector<cxd>& p,
                                             const std::vector<cxd>& f,
                                             double e, size_t max_iter_qp) {
        if (max_vecs < 2) return {p, true};
        if (count < max_vecs) { ps_flat.push_back(p); fs_flat.push_back(f); energy.push_back(e); }
        else { size_t idx = count % max_vecs; ps_flat[idx]=p; fs_flat[idx]=f; energy[idx]=e; }
        ++count; const size_t m = std::min(count, max_vecs); if (m < 2) return {p, true};

        auto inner = [](const std::vector<cxd>& a, const std::vector<cxd>& b){ return real_inner(a,b); };
        std::vector<double> g(m); std::vector<std::vector<double>> M(m, std::vector<double>(m));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long i=0;i<(long long)m;++i){
            size_t ii=(count-m+(size_t)i)%max_vecs;
            const double tr = inner(ps_flat[ii], fs_flat[ii]);
            g[(size_t)i] = energy[ii] - 0.5*tr;
            for(size_t j=0;j<m;++j){
                size_t jj=(count-m+j)%max_vecs; M[(size_t)i][j] = inner(ps_flat[ii], fs_flat[jj]);
            }
        }
        std::vector<double> c(m, 1.0/m);
        double max_abs_m = 0.0;
        for (size_t i=0;i<m;++i) for (size_t j=0;j<m;++j) max_abs_m = std::max(max_abs_m, std::abs(M[i][j]));
        const double gamma = 1.0 / (max_abs_m + 1.0);

        for (size_t it=0; it<max_iter_qp; ++it) {
            std::vector<double> grad(m,0.0);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long i=0;i<(long long)m;++i){ double s=g[(size_t)i]; for (size_t j=0;j<m;++j) s += M[(size_t)i][j]*c[j]; grad[(size_t)i]=s; }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long i=0;i<(long long)m;++i) c[(size_t)i] -= gamma*grad[(size_t)i];
            c = project_simplex(c);
        }
        std::vector<cxd> out(p.size(), cxd(0,0));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)out.size();++t){
            cxd acc(0,0);
            for (size_t i=0;i<m;++i){ size_t ii=(count-m+i)%max_vecs; acc += ps_flat[ii][(size_t)t]*c[i]; }
            out[(size_t)t] = acc;
        }
        return {out, false};
    }
};

// ---------------- Broyden mixing (with orbital preconditioning) ----------------
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
        cxd s(0,0);
        const size_t n = a.size();
        for (size_t i=0;i<n;++i) s += a[i]*b[i];
        return s;
    }

    std::pair<BroydenState, std::vector<cxd>> update(const std::vector<cxd>& P_flat,
                                                     const std::vector<cxd>& resid_flat,
                                                     double alpha = 1.3) const {
        BroydenState st = *this;
        if (st.max_vecs == 0 || st.n_flat == 0) return {st, P_flat};

        if (st.count == 0) {
            // Initialize by storing current P and resid; return unchanged P
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
        for (size_t i=0;i<st.n_flat;++i) {
            s_k[i] = P_flat[i]     - P_prev[i];
            y_k[i] = resid_flat[i] - R_prev[i];
        }
        st.s_hist[idx] = s_k;
        st.y_hist[idx] = y_k;
        st.count = k + 1;

        // L-BFGS two-loop
        std::vector<cxd> q = resid_flat;
        std::vector<cxd> alphas(st.max_vecs, cxd(0,0));

        for (size_t ii=0; ii<st.max_vecs; ++ii) {
            const size_t i = st.max_vecs - 1 - ii; // reverse order
            const cxd yi_si = dot(st.y_hist[i], st.s_hist[i]);
            const cxd rho = (std::abs(yi_si) > 1e-18) ? (cxd(1.0,0.0) / yi_si) : cxd(0.0,0.0);
            const cxd a_i = rho * dot(st.s_hist[i], q);
            alphas[i] = a_i;
            // q = q - a_i * y_i
            for (size_t t=0;t<st.n_flat;++t) q[t] -= a_i * st.y_hist[i][t];
        }

        const cxd yk_yk = dot(y_k, y_k);
        const cxd sk_yk = dot(s_k, y_k);
        const cxd gamma = (std::abs(yk_yk) > 1e-18) ? (sk_yk / yk_yk) : cxd(0.0,0.0);

        std::vector<cxd> r(q.size());
        for (size_t t=0;t<st.n_flat;++t) r[t] = gamma * q[t];

        for (size_t i=0;i<st.max_vecs;++i) {
            const cxd yi_si = dot(st.y_hist[i], st.s_hist[i]);
            const cxd rho = (std::abs(yi_si) > 1e-18) ? (cxd(1.0,0.0) / yi_si) : cxd(0.0,0.0);
            const cxd beta = rho * dot(st.y_hist[i], r);
            const cxd coeff = alphas[i] - beta;
            for (size_t t=0;t<st.n_flat;++t) r[t] += st.s_hist[i][t] * coeff;
        }

        // Step: P + alpha * r
        std::vector<cxd> P_out(st.n_flat);
        for (size_t t=0;t<st.n_flat;++t) P_out[t] = P_flat[t] + cxd(alpha,0.0)*r[t];
        return {st, P_out};
    }
};

// Orbital preconditioner on commutator: diagonalize F at each k, divide by (eps_i-eps_j)+delta
static void orbital_preconditioner(const size_t nk1, const size_t nk2, const size_t d,
                                   const std::vector<cxd>& F, const std::vector<cxd>& comm,
                                   std::vector<cxd>& out, double delta=1e-3) {
    out.resize(F.size());
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (long long k1=0;k1<(long long)nk1;++k1)
        for (long long k2=0;k2<(long long)nk2;++k2) {
            const size_t base = offset(nk2,d,(size_t)k1,(size_t)k2,0,0);
            Eigen::Map<const MatC> Fk(&F[base], d, d);
            Eigen::SelfAdjointEigenSolver<MatC> es(Fk);
            if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed (precond)");
            const MatC C = es.eigenvectors();
            const Eigen::VectorXd eps = es.eigenvalues().real();
            Eigen::Map<const MatC> Rk(&comm[base], d, d);
            MatC Rmo = C.adjoint() * Rk * C;
            MatC Rtil = MatC::Zero(d,d);
            for (size_t i=0;i<d;++i)
                for (size_t j=0;j<d;++j) {
                    const double denom = (eps((Eigen::Index)i) - eps((Eigen::Index)j)) + delta;
                    Rtil((Eigen::Index)i,(Eigen::Index)j) = Rmo((Eigen::Index)i,(Eigen::Index)j) / denom;
                }
            MatC Rpc = C * Rtil * C.adjoint();
            Eigen::Map<MatC> outk(&out[base], d, d);
            outk = Rpc;
        }
}

// ---------------- HF Kernel ----------------
struct HFKernel {
    size_t nk1, nk2, d;
    size_t n_tot;
    std::vector<double> weights; // (nk1*nk2)
    double weight_sum = 0.0;     // sum of weights
    double weight_mean = 0.0;    // scalar weight used in JAX path (dk^2/(2π)^2)
    std::vector<cxd> H;          // (nk1*nk2*d*d)
    std::vector<cxd> Vhat_full;   // (nk1*nk2*d*d)
    std::vector<cxd> Vhat_scalar; // (nk1*nk2)
    bool v_is_scalar = false;
    FftwBatched2D plan;          // batched 2D FFT plan
    double T;
    double n_target = 0.0;

    // Reusable FFT scratch (aligned)
    cxd* scratch_fft = nullptr;

    HFKernel(size_t nk1_, size_t nk2_, size_t d_,
             const py::array_t<double>& W,
             const py::array_t<cxd>& H_in,
             const py::array_t<cxd>& V_in,
             double T_,
             double n_target_)
        : nk1(nk1_), nk2(nk2_), d(d_),
          n_tot(nk1_*nk2_*d_*d_),
          plan(nk1_, nk2_, d_),
          T(T_),
          n_target(n_target_) {

        // Keep Eigen single-threaded to avoid oversubscription; OpenMP handles outer parallelism.
        Eigen::setNbThreads(1);

        // weights
        if (W.ndim()!=2 || (size_t)W.shape(0)!=nk1 || (size_t)W.shape(1)!=nk2) throw std::invalid_argument("weights must be (nk1,nk2)");
        weights.assign(W.data(), W.data()+ (nk1*nk2));
        weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        weight_mean = weight_sum / static_cast<double>(nk1*nk2);

        // H
        if (H_in.ndim()!=4 || (size_t)H_in.shape(0)!=nk1 || (size_t)H_in.shape(1)!=nk2 || (size_t)H_in.shape(2)!=d || (size_t)H_in.shape(3)!=d)
            throw std::invalid_argument("H must be (nk1,nk2,d,d)");
        H.assign(H_in.data(), H_in.data() + n_tot);

        // Allocate scratch for FFT
        scratch_fft = reinterpret_cast<cxd*>(fftw_malloc(sizeof(cxd) * n_tot));
        if (!scratch_fft) throw std::bad_alloc{};

        // Build Vhat = FFT(weight_mean * V) along (0,1) — JAX uses a scalar mesh weight
        if (V_in.ndim()!=4 || (size_t)V_in.shape(0)!=nk1 || (size_t)V_in.shape(1)!=nk2)
            throw std::invalid_argument("V must be (nk1,nk2,dv1,dv2)");
        const size_t dv1 = V_in.shape(2), dv2 = V_in.shape(3);
        if (!((dv1==1 && dv2==1) || (dv1==d && dv2==d)))
            throw std::invalid_argument("V last dims must be (1,1) or (d,d)");

        if (dv1==1 && dv2==1) {
            v_is_scalar = true;
            Vhat_scalar.resize(nk1*nk2);
            cxd* buf = reinterpret_cast<cxd*>(fftw_malloc(sizeof(cxd) * nk1 * nk2));
            if (!buf) throw std::bad_alloc{};
            const cxd* vptr = V_in.data();

#ifdef _OPENMP
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (long long k1=0;k1<(long long)nk1;++k1)
                for (long long k2=0;k2<(long long)nk2;++k2) {
                    buf[(size_t)k1*nk2 + (size_t)k2] = vptr[(size_t)k1*nk2 + (size_t)k2] * weight_mean;
                }

#if defined(FFTW3_THREADS)
            FftwBatched2D::init_threads_once();
            fftw_plan_with_nthreads(plan.nthreads);
#endif
            fftw_plan pf = fftw_plan_dft_2d((int)nk1, (int)nk2,
                                reinterpret_cast<fftw_complex*>(buf),
                                reinterpret_cast<fftw_complex*>(buf),
                                FFTW_FORWARD, FFTW_MEASURE);
            if (!pf) { fftw_free(buf); throw std::runtime_error("FFTW plan_dft_2d forward failed for scalar V"); }
            fftw_execute(pf);
            std::copy(buf, buf + nk1*nk2, Vhat_scalar.begin());
            fftw_destroy_plan(pf);
            fftw_free(buf);
        } else {
            std::vector<cxd> Vfull(n_tot);
            const cxd* vptr = V_in.data();

#ifdef _OPENMP
            #pragma omp parallel for collapse(4) schedule(static)
#endif
            for (long long k1=0;k1<(long long)nk1;++k1)
                for (long long k2=0;k2<(long long)nk2;++k2) {
                    for (long long i=0;i<(long long)d;++i)
                        for (long long j=0;j<(long long)d;++j) {
                            const size_t v_idx = (((size_t)k1*nk2 + (size_t)k2)*d + (size_t)i)*d + (size_t)j;
                            Vfull[offset(nk2,d,(size_t)k1,(size_t)k2,(size_t)i,(size_t)j)] = vptr[v_idx] * weight_mean;
                        }
                }
            Vhat_full = Vfull;
            plan.forward(Vhat_full.data());
        }
    }

    ~HFKernel() { if (scratch_fft) fftw_free(scratch_fft); }

    // In-place FFT-based exchange (JAX convention):
    // out = fftshift( - IFFT( FFT(P) * Vhat ), axes=(0,1) ), with IFFT normalized by 1/(nk1*nk2)
    void self_energy_fft(const std::vector<cxd>& P, std::vector<cxd>& out) const {
        std::copy(P.begin(), P.end(), scratch_fft);
        plan.forward(scratch_fft);

        if (v_is_scalar) {
#ifdef _OPENMP
            // GCC requires strictly nested loops for collapse; use collapse(2) and SIMD on inner loop
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (long long k1=0;k1<(long long)nk1;++k1)
              for (long long k2=0;k2<(long long)nk2;++k2) {
                  const cxd v = Vhat_scalar[(size_t)k1*nk2 + (size_t)k2];
                  const size_t base = (((size_t)k1*nk2 + (size_t)k2) * d) * d;
#ifdef _OPENMP
                  #pragma omp simd
#endif
                  for (long long t=0; t<(long long)(d*d); ++t)
                      scratch_fft[base + (size_t)t] *= v;
              }
        } else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (long long t=0; t<(long long)n_tot; ++t)
                scratch_fft[(size_t)t] *= Vhat_full[(size_t)t];
        }

        plan.backward(scratch_fft);
        const double norm = -1.0 / static_cast<double>(nk1*nk2);
        out.resize(n_tot);

        const size_t shift1 = nk1/2, shift2 = nk2/2;
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1=0;k1<(long long)nk1;++k1) {
            for (long long k2=0;k2<(long long)nk2;++k2) {
                const size_t s1 = ((size_t)k1 + shift1) % nk1;
                const size_t s2 = ((size_t)k2 + shift2) % nk2;
                const size_t src_base = (((size_t)k1*nk2 + (size_t)k2) * d) * d;
                const size_t dst_base = (((size_t)s1*nk2 + (size_t)s2) * d) * d;
#ifdef _OPENMP
                #pragma omp simd
#endif
                for (long long t=0; t<(long long)(d*d); ++t)
                    out[dst_base + (size_t)t] = scratch_fft[src_base + (size_t)t] * norm;
            }
        }
    }

    // Return (P_new, mu). NOTE: does NOT compute energy here.
    std::pair<std::vector<cxd>, double> call(const std::vector<cxd>& P) const {
        auto Ps = P;

        std::vector<cxd> Sigma;
        self_energy_fft(Ps, Sigma);
        auto Fock = H;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)n_tot;++t) Fock[(size_t)t] += Sigma[(size_t)t];

        std::vector<std::vector<double>> bands(nk1*nk2);
        std::vector<MatC> evecs(nk1*nk2);
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1=0;k1<(long long)nk1;++k1)
            for (long long k2=0;k2<(long long)nk2;++k2) {
                const size_t base = offset(nk2,d,(size_t)k1,(size_t)k2,0,0);
                Eigen::Map<MatC> Fk(&Fock[base], d, d);
                Eigen::SelfAdjointEigenSolver<MatC> es(Fk);
                if (es.info()!=Eigen::Success) throw std::runtime_error("EVD failed");
                bands[(size_t)k1*nk2+(size_t)k2] = std::vector<double>(es.eigenvalues().data(), es.eigenvalues().data()+d);
                evecs[(size_t)k1*nk2+(size_t)k2] = es.eigenvectors();
            }

        const double mu = find_mu(bands, n_target);

        // Build P_new = U f(ε-μ) U^H
        std::vector<cxd> Pnew(n_tot);
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1=0;k1<(long long)nk1;++k1)
            for (long long k2=0;k2<(long long)nk2;++k2) {
                const auto& U = evecs[(size_t)k1*nk2+(size_t)k2];
                Eigen::VectorXcd occ(d);
                for (size_t j=0;j<d;++j) occ((Eigen::Index)j) = cxd(fermi(bands[(size_t)k1*nk2+(size_t)k2][j]-mu, T), 0.0);
                MatC D(d,d);
                D.noalias() = U * occ.asDiagonal() * U.adjoint();
                const size_t base = offset(nk2,d,(size_t)k1,(size_t)k2,0,0);
                Eigen::Map<MatC>(&Pnew[base], d, d) = D;
            }

        return {Pnew, mu};
    }

    double find_mu(const std::vector<std::vector<double>>& bands,
                double target_electrons) const
    {
        // --- 1. Energy range with safety margin ---
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        for (const auto& v : bands)
            for (double x : v) {
                lo = std::min(lo, x);
                hi = std::max(hi, x);
            }

        const double pad = 10.0 * std::max(1e-6, T);
        lo -= pad;
        hi += pad;

        // --- 2. Define total electrons N(mu) ---
        auto N = [&](double mu) {
            double s = 0.0;
        #ifdef _OPENMP
            #pragma omp parallel for reduction(+:s) schedule(static)
        #endif
            for (long long k1 = 0; k1 < nk1; ++k1) {
                for (long long k2 = 0; k2 < nk2; ++k2) {
                    const std::size_t idx = static_cast<std::size_t>(k1) * nk2 + k2;
                    const double w = weights[idx];

                    double occ = 0.0;
                #ifdef _OPENMP
                    #pragma omp simd reduction(+:occ)
                #endif
                    for (long long j = 0; j < d; ++j)
                        occ += fermi(bands[idx][static_cast<std::size_t>(j)] - mu, T);

                    s += w * occ;
                }
            }
            return s;
        };

        auto f = [&](double mu) { return N(mu) - target_electrons; };

        // --- 3. Bracket the root ---
        double a = lo, b = hi;
        double fa = f(a), fb = f(b);
        int expand = 0;
        while (!(fa <= 0.0 && fb >= 0.0) && expand < 60) {
            const double width = b - a;
            a -= width;
            b += width;
            fa = f(a);
            fb = f(b);
            ++expand;
        }

        if (!(fa <= 0.0 && fb >= 0.0)) {
            // Could not bracket — return best of endpoints
            return (std::abs(fa) < std::abs(fb)) ? a : b;
        }

        // --- 4. Simple robust bisection ---
        const double tol = 1e-10;
        const int max_iter = 200;
        for (int iter = 0; iter < max_iter; ++iter) {
            const double mid = 0.5 * (a + b);
            const double fm = f(mid);

            if (std::abs(fm) < tol || (b - a) < tol)
                return mid;

            if (fa * fm < 0.0) {
                b = mid;
                fb = fm;
            } else {
                a = mid;
                fa = fm;
            }
        }

        // If we reach here, return midpoint
        return 0.5 * (a + b);
    }



    // Build Fock Σ[P] once and compute both F and E. (avoids a second FFT!)
    void fock_and_energy_of(const std::vector<cxd>& P, std::vector<cxd>& F, double& E) const {
        std::vector<cxd> Sigma;
        self_energy_fft(P, Sigma);

        F.resize(n_tot);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (long long t=0;t<(long long)n_tot;++t) F[(size_t)t] = H[(size_t)t] + Sigma[(size_t)t];

        double e = 0.0;
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(+:e) schedule(static)
#endif
        for (long long k1=0;k1<(long long)nk1;++k1)
            for (long long k2=0;k2<(long long)nk2;++k2) {
                const double w = weights[(size_t)k1*nk2+(size_t)k2];
                const size_t base = offset(nk2,d,(size_t)k1,(size_t)k2,0,0);
                double s = 0.0;
#ifdef _OPENMP
                #pragma omp simd reduction(+:s)
#endif
                for (long long t=0;t<(long long)(d*d);++t) {
                    const auto val = F[base + (size_t)t] - 0.5*Sigma[base + (size_t)t] + H[base + (size_t)t]*0.0; // (H + 0.5 Σ) : P  -> using F = H+Σ => (H+0.5Σ) = F - 0.5Σ
                    s += (val * P[base + (size_t)t]).real();
                }
                e += w * s;
            }
        E = e;
    }

    // Build Fock only (kept for API completeness)
    void fock_of(const std::vector<cxd>& P, std::vector<cxd>& F) const {
        double dummy; fock_and_energy_of(P, F, dummy);
    }
};

// ---------------- Python-exposed function ----------------
py::tuple hartreefock_iteration_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,   // (nk1,nk2)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> hamiltonian, // (nk1,nk2,d,d)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> v_coulomb,   // (nk1,nk2,dv1,dv2)
    py::array_t<cxd,    py::array::c_style | py::array::forcecast> p0,          // (nk1,nk2,d,d)
    double electron_density0,
    double T,
    size_t max_iter,
    double comm_tol,
    size_t diis_size,
    double mixing_alpha
) {
    if (hamiltonian.ndim()!=4) throw std::invalid_argument("H must be (nk1,nk2,d,d)");
    const size_t nk1 = hamiltonian.shape(0), nk2 = hamiltonian.shape(1), d = hamiltonian.shape(2);
    if ((size_t)hamiltonian.shape(3)!=d) throw std::invalid_argument("H last two dims must be equal (d,d)");
    if (weights.ndim()!=2 || (size_t)weights.shape(0)!=nk1 || (size_t)weights.shape(1)!=nk2) throw std::invalid_argument("weights must be (nk1,nk2)");
    if (p0.ndim()!=4 || (size_t)p0.shape(0)!=nk1 || (size_t)p0.shape(1)!=nk2 || (size_t)p0.shape(2)!=d || (size_t)p0.shape(3)!=d)
        throw std::invalid_argument("p0 must be (nk1,nk2,d,d)");
    if (v_coulomb.ndim()!=4 || (size_t)v_coulomb.shape(0)!=nk1 || (size_t)v_coulomb.shape(1)!=nk2)
        throw std::invalid_argument("V must be (nk1,nk2,dv1,dv2)");

    HFKernel kernel(nk1,nk2,d, weights, hamiltonian, v_coulomb, T, electron_density0);

    std::vector<cxd> P(p0.data(), p0.data()+ (nk1*nk2*d*d));
    DiisState cdiis(diis_size);
    EdiisState ediis(diis_size);

    double e_fin = 0.0; size_t k_fin = 0; double mu_fin = 0.0;

    py::gil_scoped_release nogil;

    // Mixer state mirrors JAX: start in EDIIS, then switch to Broyden
    bool use_ediis = true;
    const size_t n_flat = nk1*nk2*d*d;
    BroydenState bro_state(diis_size, n_flat);

    for (size_t k=0; k<max_iter; ++k) {
        // 1) Build P_new from F[P]
        auto call_result = kernel.call(P);
        auto P_new = std::move(call_result.first);
        auto mu    = call_result.second;

        // 2) Σ[P_new] once -> F_new and E_new (avoid double FFT)
        std::vector<cxd> F_new;
        double e_new = 0.0;
        kernel.fock_and_energy_of(P_new, F_new, e_new);

        // 3) Commutator residual (JAX convention): raw commutator (scalar weight cancels)
        std::vector<cxd> comm(P_new.size());
        double comm_norm = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for collapse(2) reduction(max:comm_norm) schedule(static)
#endif
        for (long long k1=0;k1<(long long)nk1;++k1)
            for (long long k2=0;k2<(long long)nk2;++k2) {
                const size_t base = offset(nk2,d,(size_t)k1,(size_t)k2,0,0);
                Eigen::Map<MatC> Fk(&F_new[base], d, d);
                Eigen::Map<MatC> Pk(&P_new[base], d, d);
                MatC C(d,d);
                C.noalias() = Fk * Pk - Pk * Fk;
                const double block_max = C.cwiseAbs().maxCoeff();
                if (block_max > comm_norm) comm_norm = block_max;
                Eigen::Map<MatC>(&comm[base], d, d) = C;
            }

        if (comm_norm < comm_tol) {
            P = std::move(P_new); e_fin = e_new; k_fin = k; mu_fin = mu;
            break;
        }

        // 4) Mixer schedule: EDIIS far, Broyden near (JAX style)
        static const double ediis_to_broyden_tol = 0.3;

        std::vector<cxd> P_mix;
        const bool use_ediis_now = use_ediis ? true : (comm_norm > ediis_to_broyden_tol);
        if (use_ediis_now) {
            auto ediis_result = ediis.update(P_new, F_new, e_new, 40);
            P_mix = std::move(ediis_result.first);
            use_ediis = (comm_norm > ediis_to_broyden_tol);
        } else {
            // Prepare preconditioned commutator
            std::vector<cxd> comm_pc;
            orbital_preconditioner(nk1, nk2, d, F_new, comm, comm_pc, 1.0e-3);

            const bool was_ediis = use_ediis;
            if (was_ediis) { bro_state.reset(); }

            // Flatten P_new and comm_pc
            std::vector<cxd> Pflat(P_new.begin(), P_new.end());
            std::vector<cxd> Rflat(comm_pc.begin(), comm_pc.end());
            auto upd = bro_state.update(Pflat, Rflat, mixing_alpha);
            bro_state = std::move(upd.first);
            std::vector<cxd>& Praw = upd.second; // flat
            P_mix.assign(Praw.begin(), Praw.end());

            if (was_ediis) {
                const double w_keep = 0.7, w_new = 0.3;
                for (size_t i=0;i<P_mix.size();++i) P_mix[i] = P[i]*w_keep + P_mix[i]*w_new;
            }
            use_ediis = false;
        }

        P = std::move(P_mix);
        e_fin = e_new; k_fin = k; mu_fin = mu;
    }

    // Final Fock for output (single Σ)
    std::vector<cxd> F_fin;
    kernel.fock_of(P, F_fin);

    // Compute final chemical potential consistently with the final mixed density P
    // (JAX path computes mu from P_fin as well). We already have F_fin, so just
    // diagonalize to obtain bands and solve for mu using the original target density.
    {
        std::vector<std::vector<double>> bands_final(nk1 * nk2);
#ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (long long k1 = 0; k1 < (long long)nk1; ++k1)
            for (long long k2 = 0; k2 < (long long)nk2; ++k2) {
                const size_t base = offset(nk2, d, (size_t)k1, (size_t)k2, 0, 0);
                Eigen::Map<MatC> Fk(&F_fin[base], d, d);
                Eigen::SelfAdjointEigenSolver<MatC> es(Fk);
                if (es.info() != Eigen::Success) throw std::runtime_error("EVD failed (final mu)");
                const auto& ev = es.eigenvalues();
                bands_final[(size_t)k1 * nk2 + (size_t)k2] = std::vector<double>(ev.data(), ev.data() + d);
            }
        // reuse the target passed in at entry (electron_density0)
        mu_fin = kernel.find_mu(bands_final, electron_density0);
    }

    py::gil_scoped_acquire gil;

    py::array_t<cxd> P_out({(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)d,(py::ssize_t)d});
    py::array_t<cxd> F_out({(py::ssize_t)nk1,(py::ssize_t)nk2,(py::ssize_t)d,(py::ssize_t)d});
    std::memcpy(P_out.mutable_data(), P.data(),    P.size()*sizeof(cxd));
    std::memcpy(F_out.mutable_data(), F_fin.data(),F_fin.size()*sizeof(cxd));

    return py::make_tuple(P_out, F_out, e_fin, mu_fin, k_fin);
}

PYBIND11_MODULE(cpp_hf, m) {
    m.doc() = "Hartree–Fock (k-grid) with FFTW + Eigen + OpenMP + optional Boost toms748";
    m.def("hartreefock_iteration_cpp", &hartreefock_iteration_cpp,
          py::arg("weights"), py::arg("hamiltonian"), py::arg("v_coulomb"), py::arg("p0"),
          py::arg("electron_density0"), py::arg("T"),
          py::arg("max_iter"), py::arg("comm_tol"),
          py::arg("diis_size"), py::arg("mixing_alpha"));
}
