import os

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


def load_config(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    k, a, b, c, n = lines[0].split()
    k = float(k)
    semiaxes = np.array([float(a), float(b), float(c)])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    points = data[:, 0:3]
    e1 = data[:, 3:6]
    e2 = data[:, 6:9]
    return k, semiaxes, points, e1, e2


def ellipsoid_point(theta, phi, semiaxes):
    return semiaxes * np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def ellipsoid_normal(x, semiaxes):
    n = 2 * x / semiaxes**2
    return n / np.linalg.norm(n)


def boundary_collocation(semiaxes, n_theta=24, n_phi=48):
    th, ph = np.meshgrid(
        np.linspace(0, np.pi, n_theta),
        np.linspace(0, 2 * np.pi, n_phi, endpoint=False),
        indexing="ij",
    )
    th, ph = th.ravel(), ph.ravel()
    points = np.stack(
        [ellipsoid_point(t, p, semiaxes) for t, p in zip(th, ph, strict=True)]
    )
    normals = np.stack([ellipsoid_normal(x, semiaxes) for x in points])
    return points, normals


def boundary_data(z, p, k, bnd_points, bnd_normals):
    h = np.stack(
        [
            -tangential_projection(incident_field(x, z, k, p), n)
            for x, n in zip(bnd_points, bnd_normals, strict=True)
        ]
    )
    return jnp.asarray(h.reshape(-1, 1))


def main():
    n_spectral = 256
    log_noise = -8.0

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default="res/config.txt")
    args = p.parse_args()

    k, semiaxes, points, e1, e2 = load_config(args.config)

    # configs in 2*i + c order: point i, polarization e_c
    configs = []
    for i in range(len(points)):
        n = points[i] / np.linalg.norm(points[i])
        configs.append((points[i], n, e1[i]))
        configs.append((points[i], n, e2[i]))
    n_cfg = len(configs)

    bnd_points, bnd_normals = boundary_collocation(semiaxes)
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))
    X_query = jnp.asarray(np.stack([np.concatenate([x, n]) for x, n, _ in configs]))

    kernel = TangentialMaxwellKernel(n_spectral=n_spectral, omega=k)
    model = GaussianProcess(kernel, log_noise=log_noise)

    A, phi_train = model.compute_A_and_Phi(X_train)
    L = jax.scipy.linalg.cholesky(A, lower=True)
    noise_var = jnp.exp(model.log_noise)
    phi_query = kernel.feature_map(X_query)

    T = np.zeros((n_cfg, n_cfg), dtype=complex)
    for j, (z, _, p) in enumerate(configs):
        y = boundary_data(z, p, k, bnd_points, bnd_normals)
        weights = jax.scipy.linalg.cho_solve((L, True), (phi_train @ y) / noise_var)
        field = np.asarray(phi_query.conj().T @ weights).reshape(n_cfg, 3)
        for i, (_, _, q) in enumerate(configs):
            T[i, j] = np.dot(q, field[i])

    asym = np.linalg.norm(T - T.T) / np.linalg.norm(T)
    print(f"operator shape: {T.shape}")
    print(f"||T - T^T|| / ||T|| = {asym:.3e}")
    print(f"||T|| = {np.linalg.norm(T):.4f}")

    os.makedirs("out", exist_ok=True)
    np.save("out/T_epgp.npy", T)


if __name__ == "__main__":
    main()
