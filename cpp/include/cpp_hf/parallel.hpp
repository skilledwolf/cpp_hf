// Minimal OpenMP-or-serial parallel loop helpers.
//
// Threading policy for the whole library, chosen for REPRODUCIBILITY:
// only loops whose iterations write disjoint outputs are parallelized
// (per-k dense transforms, per-ΔG Fock channels).  All reductions
// (inner products, norms, energies, BZ-averaged densities) stay serial,
// so results are bitwise identical to the serial build for any thread
// count — convergence histories, iteration counts and stopping decisions
// do not depend on OMP_NUM_THREADS.  The serial reductions are O(nk·nb²)
// against the O(nk·nb³) parallel work, so the Amdahl cost is small.
#pragma once

#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cpp_hf {

// Static partition: equal-cost iterations (per-k dense linear algebra).
template <class F>
inline void parallel_for(std::size_t n, F&& fn) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        fn(static_cast<std::size_t>(i));
#else
    for (std::size_t i = 0; i < n; ++i) fn(i);
#endif
}

// Dynamic partition: uneven iterations (ΔG channels: empty-channel skip,
// conjugate-partner skip, varying pair counts).
template <class F>
inline void parallel_for_dynamic(std::size_t n, F&& fn) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
        fn(static_cast<std::size_t>(i));
#else
    for (std::size_t i = 0; i < n; ++i) fn(i);
#endif
}

}  // namespace cpp_hf
