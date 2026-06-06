import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from maxwellgp.kernel import MaxwellKernelLike


class GaussianProcess(eqx.Module):
    kernel: MaxwellKernelLike
    log_noise: Float[Array, ""]

    def __init__(self, kernel: MaxwellKernelLike, log_noise: float = -12.0):
        self.kernel = kernel
        self.log_noise = jnp.array(log_noise, dtype=jnp.float64)

    def compute_A_and_Phi(self, X: Float[Array, "N D"], jitter=1e-8):
        phi = self.kernel.feature_map(X)
        W_diag = jnp.exp(self.kernel.log_weights).astype(jnp.complex128)
        noise_var = jnp.exp(self.log_noise)
        A = (
            jnp.diag(W_diag)
            + (phi @ phi.conj().T) / noise_var
            + jitter * jnp.eye(phi.shape[0])  # for numerical stability
        )
        return A, phi

    def nlml(
        self, X: Float[Array, "N D"], Y: Complex[Array, "M J"]
    ) -> Float[Array, ""]:
        """Negative log marginal likelihood, summed over the columns of ``Y``.

        ``Y`` may be a single response ``(M,)`` / ``(M, 1)`` or a matrix of ``J``
        independent responses sharing the same covariance (e.g. one column per
        excitation), in which case the determinant/normalizer term is counted J times.
        """
        Y = Y.astype(jnp.complex128)
        if Y.ndim == 1:
            Y = Y[:, None]
        M, J = Y.shape

        A, Phi = self.compute_A_and_Phi(X)
        L = jax.scipy.linalg.cholesky(A, lower=True)
        noise_var = jnp.exp(self.log_noise)

        Phi_Y = (Phi @ Y) / noise_var
        fit = jax.scipy.linalg.cho_solve((L, True), Phi_Y)

        y_norm_sq = jnp.vdot(Y, Y).real
        fit_term = jnp.sum((Phi_Y.conj() * fit).real)
        data_fit = 0.5 * (y_norm_sq / noise_var - fit_term)

        # log det C = log det A + M log(sigma^2) - log det W
        logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L).real))
        logdet_W = jnp.sum(self.kernel.log_weights)
        logdet_C = logdet_A + M * self.log_noise - logdet_W

        return data_fit + J * (0.5 * logdet_C + 0.5 * M * jnp.log(2.0 * jnp.pi))

    def posterior_mean(
        self,
        X_query: Float[Array, "Nq D"],
        X_train: Float[Array, "N D"],
        y_train: Complex[Array, "M 1"],
    ) -> Complex[Array, "Mq 1"]:
        y_train = y_train.astype(jnp.complex128)
        noise_var = jnp.exp(self.log_noise)

        Phi_q = self.kernel.feature_map(X_query)

        A, Phi_t = self.compute_A_and_Phi(X_train)
        Phi_y = Phi_t @ y_train

        L = jax.scipy.linalg.cholesky(A, lower=True)
        fit_weights = jax.scipy.linalg.cho_solve((L, True), Phi_y / noise_var)
        return Phi_q.conj().T @ fit_weights
