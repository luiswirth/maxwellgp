from jaxtyping import Array, Float
import equinox as eqx
import jax.numpy as jnp
import jax

from maxwellgp.kernel import FullMaxwellKernel


class GaussianProcess(eqx.Module):
    kernel: FullMaxwellKernel
    X: Float[Array, "N D"] = eqx.field(static=True)
    log_eps: Float[Array, "1"]  # Learned noise parameter

    def __init__(
        self, kernel: FullMaxwellKernel, X: Array, log_eps_init: float = -12.0
    ):
        self.kernel = kernel
        self.X = X
        self.log_eps = jnp.array([log_eps_init], dtype=jnp.float64)

    # TODO: don't hardcode jitter. scale it appropriatly!
    def compute_A_and_Phi(self, jitter=1e-6):
        Phi = self.kernel.feature_map(self.X)
        W_diag = jnp.exp(self.kernel.log_w).astype(jnp.complex128)
        # Low-rank update structure usually safer with jitter on diagonal
        # TODO: fix implicit sigma^2=1. use log_eps!(nlml already does) BUG!
        A = jnp.diag(W_diag) + Phi @ Phi.conj().T + jitter * jnp.eye(Phi.shape[0])
        return A, Phi

    def nlml(self, y: Array) -> Array:
        y = y.astype(jnp.complex128)
        A, Phi = self.compute_A_and_Phi()
        L = jax.scipy.linalg.cholesky(A, lower=True)

        # alpha = A^{-1} Phi y
        alpha = jax.scipy.linalg.cho_solve((L, True), Phi @ y)

        noise_std = jnp.exp(self.log_eps)[0]

        # Data fit term (Negative Log Likelihood part)
        y_norm2 = (y.conj().T @ y).real.squeeze()
        Fy = Phi.conj().T @ alpha
        quad = (Fy.conj().T @ Fy).real.squeeze()

        term1 = (0.5 / noise_std) * (y_norm2 - quad)
        term2 = jnp.sum(jnp.log(jnp.diagonal(L).real))
        term3 = 0.5 * jnp.sum(jnp.exp(self.kernel.log_w))

        return term1 + term2 + term3

    def posterior_mean(self, X_query: Array, y_train: Array) -> Array:
        y = y_train.astype(jnp.complex128)
        A, Phi_x = self.compute_A_and_Phi()
        Phi_q = self.kernel.feature_map(X_query)

        L = jax.scipy.linalg.cholesky(A, lower=True)
        alpha = jax.scipy.linalg.cho_solve((L, True), Phi_x @ y)
        return Phi_q.conj().T @ alpha
