#!/usr/bin/env bash
# Build + cache FFTW on Apple Silicon, or install via Homebrew on Intel.
set -euxo pipefail

if [ "$(uname -s)" != "Darwin" ]; then
  echo "Not macOS; nothing to do."
  exit 0
fi

ARCH="$(uname -m)"
FFTW_VERSION=3.3.10
REPO_ROOT="${REPO_ROOT:-$PWD}"

if [ "${ARCH}" = "arm64" ]; then
  TARBALL="https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
  SHA256="56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467"
  PREFIX="${REPO_ROOT}/.cache/fftw-${FFTW_VERSION}-macos"
  BUILD_DIR="${REPO_ROOT}/.cache/_build-fftw-${FFTW_VERSION}-macos"

  if [ -f "${PREFIX}/lib/pkgconfig/fftw3.pc" ]; then
    echo "FFTW already present at ${PREFIX} — skipping build."
  else
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    curl -L -o "fftw-${FFTW_VERSION}.tar.gz" "${TARBALL}"
    echo "${SHA256}  fftw-${FFTW_VERSION}.tar.gz" | shasum -a 256 -c
    tar xf "fftw-${FFTW_VERSION}.tar.gz"
    cd "fftw-${FFTW_VERSION}"

    # NEON is available by default on aarch64; we also pass --enable-neon (benign).
    # Threads gives libfftw3_threads.dylib. We disable Fortran here.
    ./configure \
      --prefix="${PREFIX}" \
      --host=aarch64-apple-darwin --build=aarch64-apple-darwin \
      --enable-shared \
      --disable-fortran \
      --enable-threads \
      --enable-neon

    make -j"$(sysctl -n hw.ncpu)"
    make install
  fi

  echo "FFTW installed to ${PREFIX}"
else
  # Intel macOS: use Homebrew’s fftw and expose its prefix to CMake/pkg-config
  export HOMEBREW_NO_AUTO_UPDATE="${HOMEBREW_NO_AUTO_UPDATE:-1}"
  brew install fftw libomp cmake ninja || true
  BREW_PREFIX="$(brew --prefix)"
  export PKG_CONFIG_PATH="${BREW_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export CMAKE_PREFIX_PATH="${BREW_PREFIX}:${CMAKE_PREFIX_PATH:-}"
  export LIBRARY_PATH="${BREW_PREFIX}/lib:${LIBRARY_PATH:-}"
  export CPATH="${BREW_PREFIX}/include:${CPATH:-}"
fi
