#!/usr/bin/env bash
# Build & cache FFTW (shared + threads) and vendor Boost **headers** for manylinux.
# Runs inside cibuildwheel's manylinux container (x86_64 & aarch64).
set -euxo pipefail

# -------- Repo/arch paths --------
REPO_ROOT="${REPO_ROOT:-/project}"
ARCH="$(uname -m)"

# =========================================
# ==============  BOOST  ==================
# =========================================
# We only need Boost HEADERS (toms748_solve.hpp etc.) — no libs.
BOOST_VERSION=1.89.0
BOOST_U=1_89_0
BOOST_TARBALL="https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_U}.tar.bz2"
BOOST_SHA256="85a33fa22621b4f314f8e85e1a5e2a9363d22e4f4992925d4bb3bc631b5a0c7a"

BOOST_PREFIX="${REPO_ROOT}/.cache/boost-${BOOST_VERSION}-${ARCH}"
BOOST_BUILD_DIR="${REPO_ROOT}/.cache/_build-boost-${BOOST_VERSION}-${ARCH}"

# Short-circuit if headers already staged
if [ -f "${BOOST_PREFIX}/include/boost/math/tools/toms748_solve.hpp" ]; then
  echo "Boost headers already present at ${BOOST_PREFIX} — skipping."
else
  rm -rf "${BOOST_BUILD_DIR}"
  mkdir -p "${BOOST_BUILD_DIR}"
  cd "${BOOST_BUILD_DIR}"

  curl -L -o "boost_${BOOST_U}.tar.bz2" "${BOOST_TARBALL}"
  echo "${BOOST_SHA256}  boost_${BOOST_U}.tar.bz2" | sha256sum -c

  tar -xf "boost_${BOOST_U}.tar.bz2"
  mkdir -p "${BOOST_PREFIX}/include"
  # Copy only headers; no build needed
  rsync -a --delete "boost_${BOOST_U}/boost" "${BOOST_PREFIX}/include/"
  # (Optional) also stage top-level headers (rarely needed)
  rsync -a --include="*/" --include="*.hpp" --exclude="*" "boost_${BOOST_U}/" "${BOOST_PREFIX}/include/"
fi

# Make Boost discoverable by CMake + compiler for the rest of the build
export BOOST_ROOT="${BOOST_PREFIX}"
export BOOST_INCLUDEDIR="${BOOST_PREFIX}/include"
# Help FindBoost (uses include dirs) & general CMake searches
export CMAKE_PREFIX_PATH="${BOOST_PREFIX}:${CMAKE_PREFIX_PATH:-}"
# Help the compiler find headers even if CMake were to miss it
export CPATH="${BOOST_PREFIX}/include:${CPATH:-}"

# Sanity check
test -f "${BOOST_PREFIX}/include/boost/math/tools/toms748_solve.hpp"

# =========================================
# ===============  FFTW  ==================
# =========================================
FFTW_VERSION=3.3.10
TARBALL="https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
SHA256="56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"

PREFIX="${REPO_ROOT}/.cache/fftw-${FFTW_VERSION}-${ARCH}"
BUILD_DIR="${REPO_ROOT}/.cache/_build-fftw-${FFTW_VERSION}-${ARCH}"

if [ -f "${PREFIX}/lib/pkgconfig/fftw3.pc" ]; then
  echo "FFTW already present at ${PREFIX} — skipping build."
else
  rm -rf "${BUILD_DIR}"
  mkdir -p "${BUILD_DIR}"
  cd "${BUILD_DIR}"

  curl -L -o "fftw-${FFTW_VERSION}.tar.gz" "${TARBALL}"
  echo "${SHA256}  fftw-${FFTW_VERSION}.tar.gz" | sha256sum -c

  tar xf "fftw-${FFTW_VERSION}.tar.gz"
  cd "fftw-${FFTW_VERSION}"

  EXTRA_SIMD=()
  case "${ARCH}" in
    x86_64)  EXTRA_SIMD+=(--enable-sse2) ;;
    aarch64) EXTRA_SIMD+=(--enable-neon) ;;
  esac

  ./configure \
    --prefix="${PREFIX}" \
    --enable-shared --disable-static \
    --disable-fortran \
    --enable-threads \
    "${EXTRA_SIMD[@]}"

  make -j"$(nproc)"
  make install
fi

echo "FFTW installed to ${PREFIX}"

# Make FFTW discoverable for pkg-config & compiler
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
export LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${LIBRARY_PATH:-}"
export CPATH="${PREFIX}/include:${CPATH:-}"

# Final diagnostics (optional)
echo "Using BOOST_INCLUDEDIR=${BOOST_INCLUDEDIR}"
echo "Using PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"
echo "Using CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
