import jax
import jax.numpy as jnp
import numpy as np

from maxwellgp import GaussianProcess, TangentialMaxwellKernel

jax.config.update("jax_enable_x64", True)


def green_scalar(r, k):
    return np.exp(1j * k * r) / (4 * np.pi * r)


def green_dyadic(rv, k):
    r = np.linalg.norm(rv)
    rhat = rv / r
    phi = green_scalar(r, k)
    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2
    return (1j / k) * phi * (transverse * np.eye(3) + radial * np.outer(rhat, rhat))


def incident_field(x, z, k, p):
    return green_dyadic(x - z, k) @ p


def tangential_projection(v, n):
    return v - np.dot(v, n) * n


SEMIAXES = np.array([4.0, 4.0, 6.0])
K = 2.0


def ellipsoid_point(theta, phi):
    return SEMIAXES * np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def ellipsoid_normal(x):
    n = 2 * x / SEMIAXES**2
    return n / np.linalg.norm(n)


def tangent_pair(n):
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tangential_projection(seed, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1
    )


def boundary_data(z, p, bnd_points, bnd_normals):
    h = np.stack(
        [
            -tangential_projection(incident_field(x, z, K, p), n)
            for x, n in zip(bnd_points, bnd_normals, strict=True)
        ]
    )
    return jnp.asarray(h.reshape(-1, 1))


def main():
    n_recv = 12
    n_spectral = 256
    log_noise = -8.0

    lam_points = fibonacci_sphere(n_recv)
    lam_normals = lam_points

    configs = []
    for x, n in zip(lam_points, lam_normals, strict=True):
        e1, e2 = tangent_pair(n)
        configs.append((x, n, e1))
        configs.append((x, n, e2))
    n_cfg = len(configs)

    n_theta, n_phi = 24, 48
    th, ph = np.meshgrid(
        np.linspace(0, np.pi, n_theta),
        np.linspace(0, 2 * np.pi, n_phi, endpoint=False),
        indexing="ij",
    )
    th, ph = th.ravel(), ph.ravel()
    bnd_points = np.stack([ellipsoid_point(t, p) for t, p in zip(th, ph, strict=True)])
    bnd_normals = np.stack([ellipsoid_normal(x) for x in bnd_points])
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))

    X_query = jnp.asarray(np.stack([np.concatenate([x, n]) for x, n, _ in configs]))

    kernel = TangentialMaxwellKernel(n_spectral=n_spectral, omega=K)
    model = GaussianProcess(kernel, log_noise=log_noise)

    A, phi_train = model.compute_A_and_Phi(X_train)
    L = jax.scipy.linalg.cholesky(A, lower=True)
    noise_var = jnp.exp(model.log_noise)
    phi_query = kernel.feature_map(X_query)

    T = np.zeros((n_cfg, n_cfg), dtype=complex)
    for j, (z, _n, p) in enumerate(configs):
        y = boundary_data(z, p, bnd_points, bnd_normals)
        weights = jax.scipy.linalg.cho_solve((L, True), (phi_train @ y) / noise_var)
        field = np.asarray(phi_query.conj().T @ weights).reshape(n_cfg, 3)
        for i, (_, _, q) in enumerate(configs):
            T[i, j] = np.dot(q, field[i])

    asym = np.linalg.norm(T - T.T) / np.linalg.norm(T)
    print(f"operator shape: {T.shape}")
    print(f"||T - T^T|| / ||T|| = {asym:.3e}")
    print(f"||T|| = {np.linalg.norm(T):.4f}")


if __name__ == "__main__":
    main()
