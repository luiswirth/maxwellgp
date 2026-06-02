"""Single-operator-entry validation of the EP-GP forward pipeline on the
PEC ellipsoidal cavity.

The deterministic reaction-field operator
    T(z_a, p_a ; z_b, p_b) = p_b . pi_t E^s(z_b ; transmit (z_a, p_a))
is reciprocal: swapping (transmit) <-> (receive) leaves it invariant. This
is a property of the *true* operator, so it is a self-consistency check on
the EP-GP reconstruction that requires no external reference.

We condition an EP-GP plane-wave prior on the boundary trace h = -pi_t E^i
on dD, evaluate the posterior tangential field pi_t E^s at interior points
on Lambda, and contract with the physical polarization vectors (never an
arbitrary tangent direction -- that can be accidentally orthogonal to the
field and yields a meaningless near-zero entry).
"""

import jax
import jax.numpy as jnp
import numpy as np

from maxwellgp import GaussianProcess, TangentialMaxwellKernel

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# Deterministic physics: dipole incident field and boundary forcing
# --------------------------------------------------------------------------
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
    """pi_t v = v - (v.n) n, the (non-rotated) tangential trace."""
    return v - np.dot(v, n) * n


# --------------------------------------------------------------------------
# Geometry: ellipsoid boundary dD and interior sphere Lambda
# --------------------------------------------------------------------------
SEMIAXES = np.array([4.0, 4.0, 6.0])


def ellipsoid_point(theta, phi):
    return SEMIAXES * np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def ellipsoid_normal(x):
    n = 2 * x / SEMIAXES**2
    return n / np.linalg.norm(n)


def boundary_forcing(theta, phi, z, k, p):
    """h = -pi_t E^i  on dD  (matches the projection-trace feature map)."""
    x = ellipsoid_point(theta, phi)
    n = ellipsoid_normal(x)
    return -tangential_projection(incident_field(x, z, k, p), n)


def tangent_basis(n, seed=np.array([1.0, 0.0, 0.0])):
    """One unit tangent vector at a point on Lambda with outward normal n.
    Falls back to a second seed if the first is parallel to n."""
    if abs(np.dot(seed, n)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    t = tangential_projection(seed, n)
    return t / np.linalg.norm(t)


# --------------------------------------------------------------------------
# Boundary collocation (fixed transmitter for the residual/field diagnostics)
# --------------------------------------------------------------------------
K = 2.0
N_THETA, N_PHI = 24, 48

_TH, _PH = np.meshgrid(
    np.linspace(0, np.pi, N_THETA),
    np.linspace(0, 2 * np.pi, N_PHI, endpoint=False),
    indexing="ij",
)
THETA, PHI = _TH.ravel(), _PH.ravel()

BND_POINTS = np.stack([ellipsoid_point(t, p) for t, p in zip(THETA, PHI, strict=True)])
BND_NORMALS = np.stack([ellipsoid_normal(x) for x in BND_POINTS])
# X_train packs position || normal -- the tangential feature map reads both.
X_TRAIN = jnp.asarray(np.concatenate([BND_POINTS, BND_NORMALS], axis=1))


def build_forcing(z, p):
    """Stacked boundary data h for a given transmit dipole (z, p)."""
    h = np.stack(
        [boundary_forcing(t, ph, z, K, p) for t, ph in zip(THETA, PHI, strict=True)]
    )
    return jnp.asarray(h.reshape(-1, 1))


def tangential_field(model, z, p, x_recv):
    """Posterior tangential field pi_t E^s at x_recv, from transmit (z, p)."""
    y = build_forcing(z, p)
    n_recv = x_recv / np.linalg.norm(x_recv)
    x_query = jnp.asarray(np.concatenate([x_recv, n_recv])[None, :])
    return np.asarray(model.posterior_mean(x_query, X_TRAIN, y)).ravel()


def operator_entry(model, z_a, p_a, z_b, p_b):
    """T = p_b . pi_t E^s(z_b ; transmit (z_a, p_a)).

    Contraction uses the physical polarization p_b, which is tied to the
    excitation and therefore not accidentally orthogonal to the field."""
    return np.dot(p_b, tangential_field(model, z_a, p_a, z_b))


# --------------------------------------------------------------------------
# Random generic transmit/receive configurations (off symmetry axes)
# --------------------------------------------------------------------------
def random_config(rng):
    """A random point on Lambda (|x|=1) with a random tangent polarization."""
    z = rng.standard_normal(3)
    z /= np.linalg.norm(z)  # on the unit sphere Lambda
    seed = rng.standard_normal(3)
    p = tangential_projection(seed, z)
    p /= np.linalg.norm(p)  # tangential at z, unit
    return z, p


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def boundary_residual(model):
    """Relative misfit of the posterior on the data it conditioned on.
    Uses posterior_mean directly (same code path as everything else)."""
    y = build_forcing(Z0, P0)
    m = model.posterior_mean(X_TRAIN, X_TRAIN, y)
    return float(jnp.linalg.norm(m - y) / jnp.linalg.norm(y))


def cond_A(model):
    A, _ = model.compute_A_and_Phi(X_TRAIN)
    return float(jnp.linalg.cond(A))


# A fixed reference transmitter used only for the scalar diagnostics above.
Z0 = np.array([1.0, 0.0, 0.0])
P0 = tangent_basis(Z0)


def run(n_spectral, log_noise, n_pairs=6, seed=0):
    rng = np.random.default_rng(seed)
    kernel = TangentialMaxwellKernel(n_spectral=n_spectral, omega=K)
    model = GaussianProcess(kernel, log_noise=log_noise)

    resid = boundary_residual(model)
    cond = cond_A(model)
    print(f"  boundary residual : {resid:.3e}")
    print(f"  cond(A)           : {cond:.3e}")

    # Reciprocity over several generic random pairs.
    print(f"  reciprocity over {n_pairs} random pairs:")
    rels = []
    for i in range(n_pairs):
        z_a, p_a = random_config(rng)
        z_b, p_b = random_config(rng)
        t_ab = operator_entry(model, z_a, p_a, z_b, p_b)
        t_ba = operator_entry(model, z_b, p_b, z_a, p_a)
        rel = abs(t_ab - t_ba) / abs(t_ab)
        rels.append(rel)
        print(
            f"    pair {i}: T_ab={t_ab:+.4e}  T_ba={t_ba:+.4e}  "
            f"|T_ab|={abs(t_ab):.3e}  rel={rel:.3e}"
        )
    print(f"  median reciprocity : {np.median(rels):.3e}")
    return float(np.median(rels))


def main():
    print(
        f"||h|| (reference transmitter) = {float(jnp.linalg.norm(build_forcing(Z0, P0))):.4f}\n"
    )

    print("=== refinement study (log_noise = -8) ===")
    for n_spectral in [64, 128, 256]:
        print(f"n_spectral = {n_spectral}")
        run(n_spectral, log_noise=-8.0)
        print()

    print("=== noise sweep (n_spectral = 256) ===")
    for log_noise in [-12, -10, -8, -6, -4, -2]:
        print(f"log_noise = {log_noise}")
        run(256, log_noise=float(log_noise), n_pairs=4)
        print()


if __name__ == "__main__":
    main()
