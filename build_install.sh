#!/bin/bash

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate contimod_env
PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH" pip install . -C cmake.define.HF_USE_OPENMP=ON -C cmake.define.HF_USE_FFTW_THREADS=ON