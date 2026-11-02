# Install

hartreefock needs Python 3.10 or newer and JAX. Its C++17 extension links
FFTW (double precision) and Eigen and compiles against the XLA headers that
ship with jaxlib, so a build from source needs both libraries:

::::{tab-set}

:::{tab-item} macOS
```bash
brew install fftw eigen
pip install -e .
```
:::

:::{tab-item} Debian and Ubuntu
```bash
sudo apt install libfftw3-dev libeigen3-dev
pip install -e .
```
:::
::::

Run these from a clone of
[skilledwolf/cpp_hf](https://github.com/skilledwolf/cpp_hf). Check what
loaded:

```python
import hartreefock as hf
from hartreefock._jax import ffi
print(hf.__version__, ffi.available(ffi.EIGH))
```

`True` means the C++ kernels are registered with JAX
([Precision and speed](../guides/performance.md)). A source build compiles
for the CPU it runs on (`-march=native`); pass
`-C cmake.define.HF_NATIVE_ARCH=OFF` to `pip` for a build that runs on other
machines.

## License

The source is MIT-licensed. The extension links FFTW, which is GPLv2+, so
binary wheels that bundle FFTW are distributed under the GPL as a whole.
