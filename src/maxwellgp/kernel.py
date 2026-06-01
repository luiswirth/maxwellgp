import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex

from maxwellgp.utils import fibonacci_sphere, normalize


class FullMaxwellFeatureMap(eqx.Module):
    base_dirs_raw: Float[Array, "n_spectral 3"]
    omega: float = eqx.field(static=True)
    n_spectral: int = eqx.field(static=True)
    n_pol: int = eqx.field(static=True)

    def __init__(
        self, n_spectral: int, omega: float, key=None, init_jitter: float = 0.0
    ):
        self.n_spectral = int(n_spectral)
        self.n_pol = 2
        self.omega = float(omega)

        base = fibonacci_sphere(self.n_spectral)
        if init_jitter > 0.0 and key is not None:
            base = base + init_jitter * jax.random.normal(
                key, base.shape, dtype=jnp.float64
            )
        self.base_dirs_raw = normalize(base)

    def __call__(self, X: Float[Array, "N 3"]) -> Complex[Array, "F 6N"]:
        N = X.shape[0]
        # Ensure unit norm
        kdirs = normalize(self.base_dirs_raw)

        # --- Optimized Gram-Schmidt (Avoids constructing Nx3x3 matrices) ---
        # We want to project the standard basis S (x and y axes) onto the plane normal to k
        # V_raw = S - (S . k) * k

        # 1. Basis vectors (1, 0, 0) and (0, 1, 0) broadcasted
        # kdirs is (R, 3)

        # Project Unit X: (1, 0, 0)
        dot_x = kdirs[:, 0:1]  # (R, 1) represents dot(k, [1,0,0])
        v1_raw = jnp.zeros_like(kdirs).at[:, 0].set(1.0) - dot_x * kdirs
        e1 = normalize(v1_raw)  # (R, 3)

        # Project Unit Y: (0, 1, 0)
        dot_y = kdirs[:, 1:2]
        v2_raw = jnp.zeros_like(kdirs).at[:, 1].set(1.0) - dot_y * kdirs

        # Orthogonalize v2 against e1: v2 = v2_raw - (v2_raw . e1) * e1
        dot_v2_e1 = jnp.sum(v2_raw * e1, axis=1, keepdims=True)
        e2 = normalize(v2_raw - dot_v2_e1 * e1)  # (R, 3)

        pols = jnp.stack([e1, e2], axis=1)  # (R, 2, 3)

        # --- Coefficients ---
        w = jnp.array(self.omega, dtype=jnp.float64)
        k_vec = kdirs * w

        # Cross products
        # k_exp: (R, 1, 3), pols: (R, 2, 3)
        cross_k_pi = jnp.cross(k_vec[:, None, :], pols, axis=-1)

        E = -cross_k_pi
        B = jnp.cross(kdirs[:, None, :], cross_k_pi, axis=-1)

        coeff6 = jnp.concatenate([E, B], axis=-1).astype(jnp.complex128)  # (R, 2, 6)

        # --- Phases ---
        # (N, 3) @ (R, 3).T -> (N, R)
        phases = jnp.exp(1j * (X @ k_vec.T))

        # Broadcast multiply: (R, 2, 6) * (N, R).T -> (R, 2, 6) * (R, N)
        # We need output (F, 6N) where F = R*2
        feat = jnp.einsum("rpc,nr->rpnc", coeff6, phases)

        # Flatten r*p -> F, flatten n*c -> 6N
        return feat.reshape(self.n_spectral * self.n_pol, N * 6)


class FullMaxwellKernel(eqx.Module):
    feature_map: FullMaxwellFeatureMap
    log_w: Float[Array, "F"]

    def __init__(self, n_spectral: int, omega: float, key=None):
        self.feature_map = FullMaxwellFeatureMap(
            n_spectral, omega, key, init_jitter=0.0
        )
        self.log_w = jnp.zeros(n_spectral * 2, dtype=jnp.float64)
