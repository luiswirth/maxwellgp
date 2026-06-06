from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from maxwellgp.utils import fibonacci_sphere, normalize


class MaxwellKernelLike(Protocol):
    log_weights: Float[Array, "F"]

    def feature_map(self, X: Float[Array, "N D"]) -> Complex[Array, "F M"]: ...


class MaxwellFeatureMap(eqx.Module):
    """Maxwell-constrained plane-wave feature map.

    A single transverse plane-wave basis (one core), exposed through two linear
    traces of the same fitted directions:
      * ``full``        -> the 6-component field [E, B] at points X (N, 3)
      * ``tangential``  -> the tangential trace pi_t E at oriented points X (N, 6)

    ``trace`` selects which one ``__call__`` returns, so the map plugs into the GP
    as the conditioning operator while the *same* instance still evaluates the full
    field (no separate basis that could drift out of sync).
    """

    base_dirs_raw: Float[Array, "n_spectral 3"]
    omega: float = eqx.field(static=True)
    n_spectral: int = eqx.field(static=True)
    n_pol: int = eqx.field(static=True)
    trace: str = eqx.field(static=True)

    def __init__(
        self,
        n_spectral: int,
        omega: float,
        key=None,
        init_jitter: float = 0.0,
        trace: str = "full",
    ):
        self.n_spectral = int(n_spectral)
        self.n_pol = 2
        self.omega = float(omega)
        self.trace = trace

        base = fibonacci_sphere(self.n_spectral)
        if init_jitter > 0.0 and key is not None:
            base = base + init_jitter * jax.random.normal(
                key, base.shape, dtype=jnp.float64
            )
        self.base_dirs_raw = normalize(base)

    def _core(self):
        """Directions, wavevectors and transverse (E, B) amplitudes per (dir, pol)."""
        kdirs = normalize(self.base_dirs_raw)  # (R, 3)

        # Orthonormal polarization frame in the plane normal to k. Pivot on the
        # coordinate axis least aligned with k (its smallest component): that axis
        # is the most transverse, so the cross products stay well-conditioned for
        # every direction, including those nearly parallel to an axis.
        pivot = jax.nn.one_hot(jnp.argmin(jnp.abs(kdirs), axis=1), 3, dtype=kdirs.dtype)
        e1 = normalize(jnp.cross(kdirs, pivot, axis=-1))
        e2 = normalize(jnp.cross(kdirs, e1, axis=-1))
        pols = jnp.stack([e1, e2], axis=1)  # (R, 2, 3)

        k_vec = kdirs * jnp.array(self.omega, dtype=jnp.float64)
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

    def __call__(self, X):
        return self.full(X) if self.trace == "full" else self.tangential(X)


def FullMaxwellFeatureMap(n_spectral, omega, key=None, init_jitter=0.0):
    return MaxwellFeatureMap(n_spectral, omega, key, init_jitter, trace="full")


def TangentialMaxwellFeatureMap(n_spectral, omega, key=None, init_jitter=0.0):
    return MaxwellFeatureMap(n_spectral, omega, key, init_jitter, trace="tangential")


class FullMaxwellKernel(eqx.Module):
    feature_map: MaxwellFeatureMap
    log_weights: Float[Array, "F"]

    def __init__(self, n_spectral: int, omega: float, key=None):
        self.feature_map = MaxwellFeatureMap(n_spectral, omega, key, trace="full")
        self.log_weights = jnp.zeros(n_spectral * 2, dtype=jnp.float64)


class TangentialMaxwellKernel(eqx.Module):
    feature_map: MaxwellFeatureMap
    log_weights: Float[Array, "F"]  # prior weights

    def __init__(self, n_spectral: int, omega: float, key=None):
        self.feature_map = MaxwellFeatureMap(n_spectral, omega, key, trace="tangential")
        self.log_weights = jnp.zeros(n_spectral * 2, dtype=jnp.float64)
