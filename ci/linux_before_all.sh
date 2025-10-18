#!/usr/bin/env bash
set -euxo pipefail
# Runs inside manylinux; use yum/dnf
if command -v dnf >/dev/null 2>&1; then PM=dnf; else PM=yum; fi
$PM -y install pkgconfig ninja-build unzip make which curl ca-certificates xz tar gzip cmake gcc-c++ fftw-libs fftw-devel boost-devel zstd libzstd-devel || true