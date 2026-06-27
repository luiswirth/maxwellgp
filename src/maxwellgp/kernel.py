from typing import Literal, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from maxwellgp.utils import fibonacci_sphere, normalize

Trace = Literal["full", "tangential"]


class MaxwellFeatureMap(eqx.Module):
    base_dirs_raw: Float[Array, "n_spectral 3"]
    wavenumber: float = eqx.field(static=True)
    n_spectral: int = eqx.field(static=True)
    n_pol: int = eqx.field(static=True)

    def __init__(
        self,
        n_spectral: int,
        wavenumber: float,
        key=None,
        init_jitter: float = 0.0,
    ):
        self.n_spectral = int(n_spectral)
        self.n_pol = 2
        self.wavenumber = float(wavenumber)

        base = fibonacci_sphere(self.n_spectral)
        if init_jitter > 0.0 and key is not None:
            base = base + init_jitter * jax.random.normal(
                key, base.shape, dtype=jnp.float64
            )
        self.base_dirs_raw = normalize(base)

    def _core(self):
        kdirs = normalize(self.base_dirs_raw)  # (R, 3)

        # Pivot on the coordinate axis least aligned with k (its smallest component).
        pivot = jax.nn.one_hot(jnp.argmin(jnp.abs(kdirs), axis=1), 3, dtype=kdirs.dtype)
        e1 = normalize(jnp.cross(kdirs, pivot, axis=-1))
        e2 = normalize(jnp.cross(kdirs, e1, axis=-1))
        pols = jnp.stack([e1, e2], axis=1)  # (R, 2, 3)

        k_vec = kdirs * jnp.array(self.wavenumber, dtype=jnp.float64)
        cross_k_pi = jnp.cross(k_vec[:, None, :], pols, axis=-1)
        E0 = -cross_k_pi  # (R, 2, 3)
        B0 = jnp.cross(kdirs[:, None, :], cross_k_pi, axis=-1)  # (R, 2, 3)
        return kdirs, k_vec, E0, B0

    def full(self, X: Float[Array, "N 3"]) -> Complex[Array, "F 6N"]:
        N = X.shape[0]
        _, k_vec, E0, B0 = self._core()
        coeff6 = jnp.concatenate([E0, B0], axis=-1).astype(jnp.complex128)  # (R, 2, 6)
        phases = jnp.exp(1j * (X @ k_vec.T))  # (N, R)
        feat = jnp.einsum("rpc,nr->rpnc", coeff6, phases)
        return feat.reshape(self.n_spectral * self.n_pol, N * 6)

    def tangential(self, X: Float[Array, "N 6"]) -> Complex[Array, "F 3N"]:
        N = X.shape[0]
        _, k_vec, E0, _ = self._core()
        normals = X[:, 3:][None, None, :, :]  # (1, 1, N, 3)
        E0_b = E0[:, :, None, :]  # (R, 2, 1, 3)
        tE = E0_b - jnp.sum(E0_b * normals, axis=-1, keepdims=True) * normals
        phases = jnp.exp(1j * (X[:, :3] @ k_vec.T))  # (N, R)
        feat = jnp.einsum("rpnc,nr->rpnc", tE, phases)
        return feat.reshape(self.n_spectral * self.n_pol, N * 3)


class MaxwellKernel(eqx.Module):
    feature_map: MaxwellFeatureMap
    log_weights: Float[Array, "F"]
    trace: Trace = eqx.field(static=True)

    def __init__(self, n_spectral: int, wavenumber: float, key=None, trace: Trace = "full"):
        assert trace in get_args(Trace), f"invalid trace: {trace!r}"
        self.feature_map = MaxwellFeatureMap(n_spectral, wavenumber, key)
        self.log_weights = jnp.zeros(n_spectral * 2, dtype=jnp.float64)
        self.trace = trace

    def features(self, X):
        if self.trace == "tangential":
            return self.feature_map.tangential(X)
        return self.feature_map.full(X)
