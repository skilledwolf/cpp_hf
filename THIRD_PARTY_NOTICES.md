Third-Party Notices for cpp_hf

This project links against and/or uses the following third-party components.
Observe their licenses when redistributing binaries or derivative works. If you
distribute wheels that link these libraries, the combined work must comply with
their licenses (notably GPLv2+ for FFTW when linked).

- FFTW — GNU General Public License v2 or later (GPLv2+)
  - URL: http://www.fftw.org/
  - License: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html
- Eigen — Mozilla Public License 2.0 (MPL-2.0)
  - URL: https://gitlab.com/libeigen/eigen
  - License: https://www.mozilla.org/en-US/MPL/2.0/
- pybind11 — BSD 3-Clause
  - URL: https://github.com/pybind/pybind11
  - License: https://github.com/pybind/pybind11/blob/master/LICENSE
- Boost Headers — Boost Software License 1.0
  - URL: https://www.boost.org/
  - License: https://www.boost.org/users/license.html
- NumPy (Python C API integration) — BSD 3-Clause
  - URL: https://github.com/numpy/numpy
  - License: https://numpy.org/doc/stable/license.html

Optional/conditional components (depending on build flags and platform):
- OpenMP runtime (LLVM libomp or GCC libgomp) — typically Apache 2.0 with LLVM
  exceptions (libomp) or GPL/LGPL terms (libgomp). These are not bundled by
  default wheels unless explicitly included by your build.

If you believe a third-party component is missing from this list, please open
an issue or pull request.

