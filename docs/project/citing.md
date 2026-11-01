# Citing

If hartreefock contributed to published work, cite the software and the
methods you used.

**The software.** Until there is a paper, cite the repository and the
commit you ran:

```bibtex
@software{hartreefock,
  author = {Wolf, Tobias},
  title  = {hartreefock: finite-temperature Hartree-Fock on two-dimensional k-meshes},
  url    = {https://github.com/skilledwolf/cpp_hf},
  note   = {commit <hash>},
}
```

**The methods.**

| Method | Source |
| --- | --- |
| Finite-temperature Hartree-Fock, the free energy minimized | {cite:t}`Mermin1965` |
| DIIS (`acceleration="diis"`, and the extrapolation in `"rdiis"`) | {cite:t}`Pulay1980` |
| Optimal damping (`acceleration="oda"`) | {cite:t}`Cances2000` |
| The stability test | {cite:t}`Thouless1960` |

The [bibliography](bibliography.md) lists every work the documentation
cites.
