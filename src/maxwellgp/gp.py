import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from maxwellgp.kernel import MaxwellKernel


def _as_columns(Y: Complex[Array, "M J"]) -> Complex[Array, "M J"]:
    Y = jnp.asarray(Y).astype(jnp.complex128)
    return Y[:, None] if Y.ndim == 1 else Y


class GaussianProcessPosterior(eqx.Module):
    L: Complex[Array, "F F"]
    mu_w: Complex[Array, "F J"]

    def _Ainv(self, Phi_q: Complex[Array, "F M"]) -> Complex[Array, "F M"]:
        return jax.scipy.linalg.cho_solve((self.L, True), Phi_q)

    def mean(self, Phi_q: Complex[Array, "F M"]) -> Complex[Array, "M J"]:
        return Phi_q.conj().T @ self.mu_w

    def var(self, Phi_q: Complex[Array, "F M"]) -> Float[Array, "M"]:
        diag = jnp.real(jnp.sum(Phi_q.conj() * self._Ainv(Phi_q), axis=0))
        return jnp.clip(diag, min=0.0)

    def cov(self, Phi_q: Complex[Array, "F M"]) -> Complex[Array, "M M"]:
        S = Phi_q.conj().T @ self._Ainv(Phi_q)
        return 0.5 * (S + S.conj().T)

    def sample(
        self, Phi_q: Complex[Array, "F M"], n_samples: int, key
    ) -> Complex[Array, "n_samples M J"]:
        F, J = self.mu_w.shape
        zr = jax.random.normal(key, (2, F, J, n_samples), dtype=jnp.float64)
        z = (zr[0] + 1j * zr[1]) / jnp.sqrt(2.0)
        u = jax.scipy.linalg.solve_triangular(
            self.L.conj().T, z.reshape(F, J * n_samples), lower=False
        ).reshape(F, J, n_samples)
        w = self.mu_w[:, :, None] + u
        return jnp.einsum("fm,fjn->nmj", Phi_q.conj(), w)


class GaussianProcess(eqx.Module):
    kernel: MaxwellKernel
    log_noise: Float[Array, ""]

    def __init__(self, kernel: MaxwellKernel, log_noise: float = -12.0):
        self.kernel = kernel
        self.log_noise = jnp.array(log_noise, dtype=jnp.float64)

    @property
    def noise_var(self) -> Float[Array, ""]:
        return jnp.exp(self.log_noise)

    def _factorize(self, X: Float[Array, "N D"], jitter=1e-8):
        phi = self.kernel.features(X)
        W_diag = jnp.exp(self.kernel.log_weights).astype(jnp.complex128)
        A = (
            jnp.diag(W_diag)
            + (phi @ phi.conj().T) / self.noise_var
            + jitter * jnp.eye(phi.shape[0])
        )
        return jax.scipy.linalg.cholesky(A, lower=True), phi

    def nlml(
        self, X: Float[Array, "N D"], Y: Complex[Array, "M J"]
    ) -> Float[Array, ""]:
        Y = _as_columns(Y)
        M, J = Y.shape

        L, Phi = self._factorize(X)
        Phi_Y = (Phi @ Y) / self.noise_var
        fit = jax.scipy.linalg.cho_solve((L, True), Phi_Y)

        y_norm_sq = jnp.vdot(Y, Y).real
        fit_term = jnp.sum((Phi_Y.conj() * fit).real)
        data_fit = 0.5 * (y_norm_sq / self.noise_var - fit_term)

        logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L).real))
        logdet_W = jnp.sum(self.kernel.log_weights)
        logdet_C = logdet_A + M * self.log_noise - logdet_W

        return data_fit + J * (0.5 * logdet_C + 0.5 * M * jnp.log(2.0 * jnp.pi))

    def condition(
        self,
        X_train: Float[Array, "N D"],
        y_train: Complex[Array, "M J"],
        jitter=1e-8,
    ) -> GaussianProcessPosterior:
        Y = _as_columns(y_train)
        L, Phi = self._factorize(X_train, jitter)
        mu_w = jax.scipy.linalg.cho_solve((L, True), (Phi @ Y) / self.noise_var)
        return GaussianProcessPosterior(L, mu_w)
