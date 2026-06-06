import argparse

import jax
import jax.numpy as jnp
import numpy as np

from cavity_operator import (
    JITTER,
    boundary_collocation,
    incident_field_batch,
    load_config,
    optimize_log_noise,
)
from maxwellgp import GaussianProcess, TangentialMaxwellKernel
from maxwellgp.kernel import FullMaxwellFeatureMap

jax.config.update("jax_enable_x64", True)


def tangential_rhs(z, p, k, bnd_points, bnd_normals):
    Ei = incident_field_batch(bnd_points, z, k, p)
    En = np.sum(Ei * bnd_normals, axis=1, keepdims=True)
    h = -(Ei - En * bnd_normals)
    return jnp.asarray(h.reshape(-1, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default="res/config.txt")
    ap.add_argument("--n-spectral", type=int, default=512)
    ap.add_argument("--n-boundary", type=int, default=3000)
    ap.add_argument("--log-noise", type=float, default=-8.0)
    ap.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--opt-steps", type=int, default=200)
    ap.add_argument(
        "--source",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="dipole position (required)",
    )
    ap.add_argument(
        "--pol",
        type=float,
        nargs=3,
        required=True,
        metavar=("PX", "PY", "PZ"),
        help="dipole polarization (required)",
    )
    ap.add_argument("--ngrid", type=int, default=400)
    ap.add_argument(
        "--batch", type=int, default=4000, help="query points per eval batch"
    )
    ap.add_argument("--out", default="out/field_slice.npz")
    args = ap.parse_args()

    k, semiaxes, *_ = load_config(args.config)
    a, b, c = semiaxes
    z = np.array(args.source, dtype=float)
    p = np.array(args.pol, dtype=float)

    bnd_points, bnd_normals = boundary_collocation(semiaxes, args.n_boundary)
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))
    y = tangential_rhs(z, p, k, bnd_points, bnd_normals)

    kernel = TangentialMaxwellKernel(n_spectral=args.n_spectral, omega=k)
    model = GaussianProcess(kernel, log_noise=args.log_noise)
    if args.opt_noise:
        ln, _ = optimize_log_noise(kernel, args.log_noise, X_train, y, args.opt_steps)
        model = GaussianProcess(kernel, log_noise=ln)
        print(f"tuned log_noise = {ln:.4f}")

    A, phi_train = model.compute_A_and_Phi(X_train, jitter=JITTER)
    L = jax.scipy.linalg.cholesky(A, lower=True)
    noise_var = jnp.exp(model.log_noise)
    weights = jax.scipy.linalg.cho_solve((L, True), (phi_train @ y) / noise_var)

    xs = np.linspace(-1.05 * a, 1.05 * a, args.ngrid)
    zs = np.linspace(-1.05 * c, 1.05 * c, args.ngrid)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()], axis=1)

    full = FullMaxwellFeatureMap(n_spectral=args.n_spectral, omega=k)
    chunks = []
    for i in range(0, len(pts), args.batch):
        phi = full(jnp.asarray(pts[i : i + args.batch]))
        chunks.append(np.asarray(phi.conj().T @ weights))
    field6 = np.concatenate(chunks).reshape(-1, 6)
    Escat = field6[:, :3]
    Einc = incident_field_batch(pts, z, k, p)
    Etot = Einc + Escat

    inside = (
        pts[:, 0] ** 2 / a**2 + pts[:, 1] ** 2 / b**2 + pts[:, 2] ** 2 / c**2
    ) <= 1.0
    ng = args.ngrid
    np.savez(
        args.out,
        xs=xs,
        zs=zs,
        Escat=Escat.reshape(ng, ng, 3),
        Einc=Einc.reshape(ng, ng, 3),
        Etot=Etot.reshape(ng, ng, 3),
        mask=inside.reshape(ng, ng),
        semiaxes=semiaxes,
        source=z,
        pol=p,
        k=k,
    )
    print(f"wrote {args.out}  (slice {ng}x{ng}, source={z.tolist()}, pol={p.tolist()})")


if __name__ == "__main__":
    main()
