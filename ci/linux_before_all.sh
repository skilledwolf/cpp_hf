#!/usr/bin/env bash
# Build & cache FFTW (shared + threads) for manylinux.
set -euxo pipefail

REPO_ROOT="${REPO_ROOT:-/project}"
ARCH="$(uname -m)"

# ---------------------- FFTW 3.3.10 -----------------------
FFTW_VERSION=3.3.10
TARBALL="https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
SHA256="56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"

PREFIX="${REPO_ROOT}/.cache/fftw-${FFTW_VERSION}-${ARCH}"
BUILD_DIR="${REPO_ROOT}/.cache/_build-fftw-${FFTW_VERSION}-${ARCH}"

if [ ! -f "${PREFIX}/lib/pkgconfig/fftw3.pc" ]; then
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

# Make FFTW discoverable by pkg-config & compiler
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
export LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${LIBRARY_PATH:-}"
export CPATH="${PREFIX}/include:${CPATH:-}"

# Diagnostics (optional)
echo "Using PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"
echo "Using CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
