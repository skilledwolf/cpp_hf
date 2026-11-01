# Development

hartreefock is developed at
[skilledwolf/cpp_hf](https://github.com/skilledwolf/cpp_hf); the
[changelog](https://github.com/skilledwolf/cpp_hf/blob/main/CHANGELOG.md)
lists the changes by release.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -n 4 tests --deselect tests/test_bilayer_regression.py
```

`tests/` has the backend-independent checks (superlattice, continuation,
symmetry, backend parity); `tests/native` and `tests/jax` the backend-specific
ones. `tests/test_bilayer_regression.py` takes about a minute on 4 cores (it needs contimod) and pins
energies against `tests/data/bilayer_reference.npz`. The scripts in
`docs/examples`, which these pages include, run in `tests/test_docs_examples.py`.

## Documentation

```bash
pip install -e ".[docs]"
cd docs
sphinx-build -W --keep-going -b html . _build/html
```

The build fails on any warning: every public name needs a docstring
(NumPy style), every cross-reference has to resolve, and the pages may not
use the words that `docs/_style.py` bans. The `Docs` workflow builds the
site on every push and publishes it to GitHub Pages from `main`.

`docs/theme` is shared with the contimod documentation; change it in both
repositories together.
