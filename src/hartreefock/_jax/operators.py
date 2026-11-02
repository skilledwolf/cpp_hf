"""Jitted Fock evaluation and its exact JVP, for :mod:`hartreefock.stability`."""

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .fock import fock_builder, hf_energy
from .utils import device_args

_STATIC_FOCK_ARGS = (
    "include_exchange", "include_hartree",
    "exchange_hermitian_channel_packing", "fock_build_fn",
)


@partial(jax.jit, static_argnames=_STATIC_FOCK_ARGS)
def _evaluate_fock(P, *, fock_build_fn, **kwargs):
    sigma, hartree, fock = fock_build_fn(P, project_fn=None, **kwargs)
    energy = hf_energy(P, h=kwargs["h"], Sigma=sigma, H=hartree,
                       weights_b=kwargs["w2d"][..., None, None],
                       refP=kwargs["refP"])
    return fock, energy


@partial(jax.jit, static_argnames=_STATIC_FOCK_ARGS)
def _response_fock(P, direction, *, fock_build_fn, **kwargs):
    return jax.jvp(
        lambda density: _evaluate_fock(
            density, fock_build_fn=fock_build_fn, **kwargs)[0],
        (P,), (direction,),
    )[1]


def _fock_kwargs(kernel):
    args = device_args(kernel)
    names = ("h", "VR", "refP", "HH", "include_exchange", "include_hartree",
             "exchange_hermitian_channel_packing", "contact_g", "contact_Oi",
             "contact_Oj")
    kwargs = {name: args[name] for name in names}
    kwargs.update(w2d=kernel.w2d,
                  fock_build_fn=fock_builder(kernel))
    return kwargs


def fock_operators(kernel):
    """``evaluate(P) -> (F, E)`` and ``response(P, dP) -> dF`` on host arrays."""
    kwargs = _fock_kwargs(kernel)
    dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.result_type(kernel.h.dtype, np.complex64)))

    def evaluate(P):
        F, E = _evaluate_fock(jnp.asarray(P, dtype=dtype), **kwargs)
        return np.asarray(F), float(E) - kernel.embedded_energy_offset

    def response(P, dP):
        return np.asarray(_response_fock(
            jnp.asarray(P, dtype=dtype), jnp.asarray(dP, dtype=dtype), **kwargs))

    return SimpleNamespace(evaluate=evaluate, response=response, dtype=dtype)
