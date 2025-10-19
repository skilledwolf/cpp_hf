#!/usr/bin/env bash
# Build and cache FFTW for manylinux (x86_64 & aarch64), shared libs + threads.
set -euxo pipefail

FFTW_VERSION=3.3.10
TARBALL="https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
SHA256="56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"

# cibuildwheel mounts the repo at /project inside the container
REPO_ROOT="${REPO_ROOT:-/project}"
ARCH="$(uname -m)"
PREFIX="${REPO_ROOT}/.cache/fftw-${FFTW_VERSION}-${ARCH}"
BUILD_DIR="${REPO_ROOT}/.cache/_build-fftw-${FFTW_VERSION}-${ARCH}"

# Short-circuit if cached install exists
if [ -f "${PREFIX}/lib/pkgconfig/fftw3.pc" ]; then
  echo "FFTW already present at ${PREFIX} — skipping build."
  exit 0
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

curl -L -o "fftw-${FFTW_VERSION}.tar.gz" "${TARBALL}"
echo "${SHA256}  fftw-${FFTW_VERSION}.tar.gz" | sha256sum -c

tar xf "fftw-${FFTW_VERSION}.tar.gz"
cd "fftw-${FFTW_VERSION}"

EXTRA_SIMD=()
case "${ARCH}" in
  x86_64)  EXTRA_SIMD+=(--enable-sse2) ;;
  aarch64) EXTRA_SIMD+=(--enable-neon) ;;  # double-precision NEON is supported on aarch64
esac

./configure \
  --prefix="${PREFIX}" \
  --enable-shared --disable-static \
  --disable-fortran \
  --enable-threads \
  "${EXTRA_SIMD[@]}"

make -j"$(nproc)"
make install

echo "FFTW installed to ${PREFIX}"
