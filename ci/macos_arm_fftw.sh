# 1) Prereqs
xcode-select --install

# 2) Fetch FFTW
cd /tmp
curl -LO https://fftw.org/fftw-3.3.10.tar.gz
tar xzf fftw-3.3.10.tar.gz
cd fftw-3.3.10

# 3) Choose an install prefix
PREFIX=$HOME/.local

# 4) Common flags (optimize for your M-series CPU)
export CC=clang
export CFLAGS="-O3 -ffp-contract=fast -fstrict-aliasing -fno-math-errno -pipe" #-mcpu=native 
# (Optional) Thin LTO can help: add -flto=thin to CFLAGS and LDFLAGS

# 5) Build & install double precision (default)
./configure --prefix="$PREFIX" --enable-shared \
            --enable-threads --host=aarch64-apple-darwin --build=aarch64-apple-darwin --enable-neon
make -j"$(sysctl -n hw.ncpu)"
make check
make install

# 6) Build & install single precision (float)
# make distclean
# ./configure --prefix="$PREFIX" --enable-shared \
#             --enable-threads --enable-float --host=aarch64-apple-darwin --build=aarch64-apple-darwin --enable-neon
# make -j"$(sysctl -n hw.ncpu)"
# make check
# make install

# Remove source dir and tarball
cd ..
rm -rf fftw-3.3.10 fftw-3.3.10.tar.gz
echo "[*] FFTW installed to $PREFIX" 