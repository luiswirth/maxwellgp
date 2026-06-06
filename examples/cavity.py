import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from maxwellgp import GaussianProcess, TangentialMaxwellKernel

jax.config.update("jax_enable_x64", True)

JITTER = 1e-8


# --- shared physics / geometry ------------------------------------------------

def incident_field_batch(X, z, k, p):
    rv = X - z
    r = np.linalg.norm(rv, axis=1)
    rhat = rv / r[:, None]
    phi = np.exp(1j * k * r) / (4 * np.pi * r)
    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2
    rhat_p = rhat @ p
    return (1j / k) * phi[:, None] * (
        transverse[:, None] * p + radial[:, None] * rhat_p[:, None] * rhat
    )


def load_config(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    k, a, b, c, _n = lines[0].split()
    semiaxes = np.array([float(a), float(b), float(c)])
    data = np.array([[float(v) for v in ln.split()] for ln in lines[1:]])
    return float(k), semiaxes, data[:, 0:3], data[:, 3:6], data[:, 6:9]


def boundary_collocation(semiaxes, n):
    from maxwellgp.utils import fibonacci_sphere

    u = np.asarray(fibonacci_sphere(n))
    points = u * semiaxes
    normals = points / semiaxes**2
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def tangential_trace(Ei, normals):
    """Tangential trace of the (negated) incident field: PEC boundary data."""
    En = np.sum(Ei * normals, axis=1, keepdims=True)
    return -(Ei - En * normals)


# --- conditioning (shared fit) ------------------------------------------------

def optimize_log_noise(kernel, log_noise0, X_train, Y, steps, lr=0.05):
    ln = jnp.asarray(log_noise0)
    opt = optax.adam(lr)
    state = opt.init(ln)

    def loss_of(ln):
        return GaussianProcess(kernel, log_noise=ln).nlml(X_train, Y)

    @jax.jit
    def step(ln, state):
        loss, g = jax.value_and_grad(loss_of)(ln)
        updates, state = opt.update(g, state)
        ln = jnp.clip(optax.apply_updates(ln, updates), -12.0, 0.0)
        return ln, state, loss

    loss = loss_of(ln)
    for _ in range(steps):
        ln, state, loss = step(ln, state)
    return float(ln), float(loss)


def fit(args, semiaxes, k, Y):
    """Build the boundary, condition the Maxwell-GP, return (model, weights)."""
    bnd_points, bnd_normals = boundary_collocation(semiaxes, args.n_boundary)
    X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))

    kernel = TangentialMaxwellKernel(n_spectral=args.n_spectral, omega=k)
    log_noise = args.log_noise
    if args.opt_noise:
        log_noise, _ = optimize_log_noise(kernel, log_noise, X_train, Y, args.opt_steps)
        print(f"tuned log_noise = {log_noise:.4f} (eps={np.exp(log_noise):.3e})")
    model = GaussianProcess(kernel, log_noise=log_noise)

    A, phi_train = model.compute_A_and_Phi(X_train, jitter=JITTER)
    L = jax.scipy.linalg.cholesky(A, lower=True)
    noise_var = jnp.exp(model.log_noise)
    weights = jax.scipy.linalg.cho_solve((L, True), (phi_train @ Y) / noise_var)
    return model, weights, A


# --- subcommand: reaction operator --------------------------------------------

def run_operator(args):
    k, semiaxes, points, e1, e2 = load_config(args.config)

    configs = []
    for i in range(len(points)):
        n = points[i] / np.linalg.norm(points[i])
        configs.append((points[i], n, e1[i]))
        configs.append((points[i], n, e2[i]))
    n_cfg = len(configs)

    bnd_points, bnd_normals = boundary_collocation(semiaxes, args.n_boundary)
    cols = [tangential_trace(incident_field_batch(bnd_points, z, k, p), bnd_normals).reshape(-1)
            for z, _, p in configs]
    Y = jnp.asarray(np.stack(cols, axis=1))

    model, weights, A = fit(args, semiaxes, k, Y)

    X_query = jnp.asarray(np.stack([np.concatenate([x, nrm]) for x, nrm, _ in configs]))
    phi_query = model.kernel.feature_map(X_query)
    field = np.asarray(phi_query.conj().T @ weights).reshape(n_cfg, 3, n_cfg)
    Q = np.stack([q for _, _, q in configs])
    T = np.einsum("ic,icj->ij", Q, field)

    asym = np.linalg.norm(T - T.T) / np.linalg.norm(T)
    print(f"operator shape: {T.shape}  n_spectral={args.n_spectral}  n_boundary={args.n_boundary}")
    print(f"||T - T^T|| / ||T|| = {asym:.3e}")
    print(f"||T|| = {np.linalg.norm(T):.4f}   cond(A) = {float(np.linalg.cond(np.asarray(A))):.3e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, T)
    print(f"wrote {args.out}")


# --- subcommand: field slice --------------------------------------------------

def run_field(args):
    k, semiaxes, *_ = load_config(args.config)
    a, b, c = semiaxes
    z = np.array(args.source, dtype=float)
    p = np.array(args.pol, dtype=float)

    bnd_points, bnd_normals = boundary_collocation(semiaxes, args.n_boundary)
    y = jnp.asarray(tangential_trace(incident_field_batch(bnd_points, z, k, p),
                                     bnd_normals).reshape(-1, 1))

    model, weights, _ = fit(args, semiaxes, k, y)

    xs = np.linspace(-1.05 * a, 1.05 * a, args.ngrid)
    zs = np.linspace(-1.05 * c, 1.05 * c, args.ngrid)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()], axis=1)

    chunks = []
    for i in range(0, len(pts), args.batch):
        phi = model.kernel.feature_map.full(jnp.asarray(pts[i : i + args.batch]))
        chunks.append(np.asarray(phi.conj().T @ weights))
    field6 = np.concatenate(chunks).reshape(-1, 6)
    Escat = field6[:, :3]
    Einc = incident_field_batch(pts, z, k, p)
    Etot = Einc + Escat

    inside = (pts[:, 0] ** 2 / a**2 + pts[:, 1] ** 2 / b**2 + pts[:, 2] ** 2 / c**2) <= 1.0
    ng = args.ngrid
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        xs=xs, zs=zs,
        Escat=Escat.reshape(ng, ng, 3),
        Einc=Einc.reshape(ng, ng, 3),
        Etot=Etot.reshape(ng, ng, 3),
        mask=inside.reshape(ng, ng),
        semiaxes=semiaxes, source=z, pol=p, k=k,
    )
    print(f"wrote {args.out}  (slice {ng}x{ng}, source={z.tolist()}, pol={p.tolist()})")


# --- CLI ----------------------------------------------------------------------

def add_common(sp):
    sp.add_argument("config", nargs="?", default="res/config.txt")
    sp.add_argument("--n-spectral", type=int, default=256)
    sp.add_argument("--n-boundary", type=int, default=1200)
    sp.add_argument("--log-noise", type=float, default=-8.0)
    sp.add_argument("--opt-noise", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--opt-steps", type=int, default=200)


def main():
    ap = argparse.ArgumentParser(description="Maxwell-GP PEC ellipsoidal cavity")
    sub = ap.add_subparsers(dest="cmd", required=True)

    op = sub.add_parser("operator", help="assemble the dipole reaction operator T")
    add_common(op)
    op.add_argument("--out", default="out/T_epgp.npy")
    op.set_defaults(func=run_operator)

    fld = sub.add_parser("field", help="evaluate the field on a slice for one dipole")
    add_common(fld)
    fld.add_argument("--source", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    fld.add_argument("--pol", type=float, nargs=3, required=True, metavar=("PX", "PY", "PZ"))
    fld.add_argument("--ngrid", type=int, default=400)
    fld.add_argument("--batch", type=int, default=4000)
    fld.add_argument("--out", default="out/field_slice.npz")
    fld.set_defaults(func=run_field)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
