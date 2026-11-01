# Install

hartreefock needs Python 3.10 or newer. The native backend is a C++17
extension that links FFTW (double precision) and Eigen, so a build from
source needs both:

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
[skilledwolf/cpp_hf](https://github.com/skilledwolf/cpp_hf). The JAX backend
is an extra:

```bash
pip install -e ".[jax]"
```

Check what loaded:

```python
import hartreefock as hf
print(hf.__version__, hf.available_backends())
```

A source build compiles for the CPU it runs on (`-march=native`); pass
`-C cmake.define.HF_NATIVE_ARCH=OFF` to `pip` for a build that runs on other
machines. The package imports even when the extension failed to build; a native solve
then raises a `RuntimeError` that says so.

## Precision

The native backend computes in double precision. The JAX backend uses the
dtype of the arrays you pass; enable `jax_enable_x64` before you build the
kernel when you want double precision there too, as
{func}`~hartreefock.stability_analysis` and {func}`~hartreefock.solve_minimum`
need for their default tolerances.

## License

The source and the JAX backend are MIT-licensed. The native backend links
FFTW, which is GPLv2+, so binary wheels that bundle FFTW are distributed
under the GPL as a whole.
