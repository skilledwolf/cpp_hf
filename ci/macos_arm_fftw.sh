#!/usr/bin/env bash
# Build and cache NEON-enabled FFTW for Apple Silicon (arm64).
set -euxo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
  echo "Not macOS; nothing to do."
  exit 0
fi

ARCH="$(uname -m)"
if [ "${ARCH}" != "arm64" ]; then
  echo "Not Apple Silicon; nothing to do (Intel can use Homebrew fftw)."
  exit 0
fi

FFTW_VERSION=3.3.10
TARBALL="https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
SHA256="56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"

# cibuildwheel runs from the repo root on macOS
REPO_ROOT="${REPO_ROOT:-$PWD}"
PREFIX="${REPO_ROOT}/.cache/fftw-${FFTW_VERSION}-macos"
BUILD_DIR="${REPO_ROOT}/.cache/_build-fftw-${FFTW_VERSION}-macos"

if [ -f "${PREFIX}/lib/pkgconfig/fftw3.pc" ]; then
  echo "FFTW already present at ${PREFIX} — skipping build."
  exit 0
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

curl -L -o "fftw-${FFTW_VERSION}.tar.gz" "${TARBALL}"
echo "${SHA256}  fftw-${FFTW_VERSION}.tar.gz" | shasum -a 256 -c

tar xf "fftw-${FFTW_VERSION}.tar.gz"
cd "fftw-${FFTW_VERSION}"

./configure \
  --prefix="${PREFIX}" \
  --enable-shared --disable-static \
  --disable-fortran \
  --enable-threads \
  --enable-neon

make -j"$(sysctl -n hw.ncpu)"
make install

echo "FFTW installed to ${PREFIX}"
