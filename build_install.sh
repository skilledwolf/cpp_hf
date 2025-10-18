#!/usr/bin/env bash
set -euo pipefail

# Optional: activate a specific conda env if HF_CONDA_ENV is set
if [ -n "${HF_CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)" || true
  if [ -n "${CONDA_BASE:-}" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$HF_CONDA_ENV" || true
  fi
fi

# Prefer user-local prefixes if present to find FFTW/Eigen/etc.
LOCAL_PREFIX="${LOCAL_PREFIX:-$HOME/.local}"
# Only set this to non-empty when the prefix exists and is valid
USE_LOCAL_PREFIX=""
if [ -d "$LOCAL_PREFIX" ]; then
  export PKG_CONFIG_PATH="$LOCAL_PREFIX/lib/pkgconfig:$LOCAL_PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
  export LIBRARY_PATH="$LOCAL_PREFIX/lib:$LOCAL_PREFIX/lib64:${LIBRARY_PATH:-}"
  export CPATH="$LOCAL_PREFIX/include:${CPATH:-}"
  # If a CMake package config exists, point CMake directly to it
  if [ -f "$LOCAL_PREFIX/lib/cmake/fftw3/FFTW3Config.cmake" ]; then
    if [ -f "$LOCAL_PREFIX/lib/cmake/fftw3/FFTW3Targets.cmake" ] || [ -f "$LOCAL_PREFIX/lib/cmake/fftw3/FFTW3LibraryDepends.cmake" ]; then
      export FFTW3_DIR="$LOCAL_PREFIX/lib/cmake/fftw3"
    else
      echo "[cpp_hf] Warning: Incomplete FFTW3 CMake config under $LOCAL_PREFIX; falling back to pkg-config"
    fi
  fi
  # Prefix exists; still may be useful for headers/libs via pkg-config
  USE_LOCAL_PREFIX=1
fi

# Homebrew (macOS): prefer CMake package config if available
# Accept a valid config if FFTW3Config.cmake exists (Homebrew typically ships FFTW3Targets.cmake, not FFTW3LibraryDepends.cmake)
if [ -f "/opt/homebrew/lib/cmake/fftw3/FFTW3Config.cmake" ]; then
  export FFTW3_DIR="/opt/homebrew/lib/cmake/fftw3"
fi
# If brew is available, try to detect the actual FFTW prefix and add pkg-config paths
if command -v brew >/dev/null 2>&1; then
  brew_fftw_prefix="$(brew --prefix fftw 2>/dev/null || true)"
  if [ -n "${brew_fftw_prefix}" ] && [ -f "${brew_fftw_prefix}/lib/cmake/fftw3/FFTW3Config.cmake" ]; then
    export FFTW3_DIR="${brew_fftw_prefix}/lib/cmake/fftw3"
  fi
  brew_prefix="$(brew --prefix 2>/dev/null || true)"
  if [ -d "${brew_prefix}/lib/pkgconfig" ]; then
    export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:${brew_prefix}/lib/pkgconfig"
  fi
  if [ -n "${brew_fftw_prefix}" ] && [ -d "${brew_fftw_prefix}/lib/pkgconfig" ]; then
    export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:${brew_fftw_prefix}/lib/pkgconfig"
  fi
fi

# Avoid CMake user/system package registries picking stale FFTW entries
export CMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
export CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE

# Always add Homebrew pkg-config path if present
if [ -d "/opt/homebrew/lib/pkgconfig" ]; then
  export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:/opt/homebrew/lib/pkgconfig"
fi

echo "[cpp_hf] Building wheel (no build isolation)"
if [ -n "${USE_LOCAL_PREFIX:-}" ]; then
  python -m pip install . --no-build-isolation \
    -C cmake.build-type=Release \
    -C cmake.define.CMAKE_PREFIX_PATH="$LOCAL_PREFIX" \
    -C cmake.define.HF_USE_FFTW_THREADS=ON
else
  python -m pip install . --no-build-isolation \
    -C cmake.build-type=Release \
    -C cmake.define.HF_USE_FFTW_THREADS=ON
fi

echo "[cpp_hf] Done. If FFTW was missing, consider running ./build_fftw_macos_arm.sh (macOS) or install FFTW via your package manager."
