import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxwellgp import GaussianProcess, TangentialMaxwellKernel
from maxwellgp.utils import fibonacci_sphere

jax.config.update("jax_enable_x64", True)

JITTER = 1e-8


def incident_field_batch(X, z, k, p):
    rv = X - z
    r = np.linalg.norm(rv, axis=1)
    rhat = rv / r[:, None]
    phi = np.exp(1j * k * r) / (4 * np.pi * r)
    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2
    rhat_p = rhat @ p
    G_p = (1j / k) * phi[:, None] * (
        transverse[:, None] * p + radial[:, None] * rhat_p[:, None] * rhat
    )
    return G_p


def load_config(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    k, a, b, c, _n = lines[0].split()
    k = float(k)
    semiaxes = np.array([float(a), float(b), float(c)])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    return k, semiaxes, data[:, 0:3], data[:, 3:6], data[:, 6:9]


def boundary_collocation(semiaxes, n):
    u = np.asarray(fibonacci_sphere(n))
    points = u * semiaxes
    normals = points / semiaxes**2
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def boundary_data_matrix(configs, k, bnd_points, bnd_normals):
    cols = []
    for z, _, p in configs:
        Ei = incident_field_batch(bnd_points, z, k, p)
        En = np.sum(Ei * bnd_normals, axis=1, keepdims=True)
        h = -(Ei - En * bnd_normals)
        cols.append(h.reshape(-1))
    return jnp.asarray(np.stack(cols, axis=1))


def total_nlml(kernel, log_noise, X_train, Y):
    phi = kernel.feature_map(X_train)
    W = jnp.exp(kernel.log_weights).astype(jnp.complex128)
    noise_var = jnp.exp(log_noise)
    A = jnp.diag(W) + (phi @ phi.conj().T) / noise_var + JITTER * jnp.eye(phi.shape[0])
    L = jax.scipy.linalg.cholesky(A, lower=True)

    Phi_Y = (phi @ Y) / noise_var
    fit = jax.scipy.linalg.cho_solve((L, True), Phi_Y)

    M, J = Y.shape
    y_norm_sq = jnp.vdot(Y, Y).real
    fit_term = jnp.sum((Phi_Y.conj() * fit).real)
    data_fit = 0.5 * (y_norm_sq / noise_var - fit_term)

    logdet_A = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L).real))
    logdet_W = jnp.sum(kernel.log_weights)
    logdet_C = logdet_A + M * log_noise - logdet_W
    return data_fit + J * (0.5 * logdet_C + 0.5 * M * jnp.log(2.0 * jnp.pi))


def optimize_log_noise(kernel, log_noise0, X_train, Y, steps, lr=0.05):
    ln = jnp.asarray(log_noise0)
    opt = optax.adam(lr)
    state = opt.init(ln)

    @jax.jit
    def step(ln, state):
        loss, g = jax.value_and_grad(lambda v: total_nlml(kernel, v, X_train, Y))(ln)
        updates, state = opt.update(g, state)
        ln = jnp.clip(optax.apply_updates(ln, updates), -12.0, 0.0)
        return ln, state, loss

    loss = total_nlml(kernel, ln, X_train, Y)
    for _ in range(steps):
        ln, state, loss = step(ln, state)
    return float(ln), float(loss)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", nargs="?", default="res/config.txt")
    p.add_argument("--n-spectral", type=int, default=256)
    p.add_argument("--n-boundary", type=int, default=1200)
    p.add_argument("--log-noise", type=float, default=-8.0, help="initial/fixed log_noise")
    p.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=True,
                   help="tune log_noise by marginal likelihood (default: on)")
    p.add_argument("--opt-steps", type=int, default=200)
    p.add_argument("--out", default="out/T_epgp.npy")
    args = p.parse_args()

    k, semiaxes, points, e1, e2 = load_config(args.config)

    configs = []
    for i in range(len(points)):
        n = points[i] / np.linalg.norm(points[i])
        configs.append((points[i], n, e1[i]))
        configs.append((points[i], n, e2[i]))
    n_cfg = len(configs)

    bnd_points, bnd_normals = boundary_collocation(semiaxes, args.n_boundary)
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))
    X_query = jnp.asarray(np.stack([np.concatenate([x, nrm]) for x, nrm, _ in configs]))
    Y = boundary_data_matrix(configs, k, bnd_points, bnd_normals)

    kernel = TangentialMaxwellKernel(n_spectral=args.n_spectral, omega=k)
    model = GaussianProcess(kernel, log_noise=args.log_noise)

    if args.opt_noise:
        ln, nlml = optimize_log_noise(kernel, args.log_noise, X_train, Y, args.opt_steps)
        model = GaussianProcess(kernel, log_noise=ln)
        print(f"tuned log_noise = {ln:.4f} (eps={np.exp(ln):.3e}), nlml={nlml:.4e}")

    A, phi_train = model.compute_A_and_Phi(X_train, jitter=JITTER)
    L = jax.scipy.linalg.cholesky(A, lower=True)
    noise_var = jnp.exp(model.log_noise)
    weights = jax.scipy.linalg.cho_solve((L, True), (phi_train @ Y) / noise_var)

    phi_query = kernel.feature_map(X_query)
    field = np.asarray(phi_query.conj().T @ weights).reshape(n_cfg, 3, n_cfg)
    Q = np.stack([q for _, _, q in configs])
    T = np.einsum("ic,icj->ij", Q, field)

    asym = np.linalg.norm(T - T.T) / np.linalg.norm(T)
    condA = float(np.linalg.cond(np.asarray(A)))
    print(f"operator shape: {T.shape}  n_spectral={args.n_spectral}  n_boundary={args.n_boundary}")
    print(f"||T - T^T|| / ||T|| = {asym:.3e}")
    print(f"||T|| = {np.linalg.norm(T):.4f}   cond(A) = {condA:.3e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, T)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
