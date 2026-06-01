import jax
import jax.numpy as jnp
import numpy as np

from maxwellgp import GaussianProcess, TangentialMaxwellKernel


def green_scalar(r: np.floating, k: float):
    return np.exp(1j * k * r) / (4 * np.pi * r)


def green_dyadic(rv: np.ndarray, k: float) -> np.ndarray:
    r = np.linalg.norm(rv)
    rhat = rv / r

    Phi = green_scalar(r, k)

    id = np.eye(3)
    rr = np.outer(rhat, rhat)  # r-hat dyad, shape (3,3)

    transverse = k**2 + 1j * k / r - 1 / r**2
    radial = -(k**2) - 3j * k / r + 3 / r**2

    return (1j / k) * Phi * (transverse * id + radial * rr)


def incident_field(x: np.ndarray, z: np.ndarray, k: float, p: np.ndarray) -> np.ndarray:
    return green_dyadic(x - z, k) @ p


def ellipsoid_point(
    theta: float, phi: float, semiaxes: tuple[float, float, float]
) -> np.ndarray:
    x = np.array(
        [
            semiaxes[0] * np.sin(theta) * np.cos(phi),
            semiaxes[1] * np.sin(theta) * np.sin(phi),
            semiaxes[2] * np.cos(theta),
        ]
    )
    return x


def ellipsoid_normal(x: np.ndarray, semiaxes: tuple[float, float, float]) -> np.ndarray:
    a = semiaxes
    n = np.array([2 * x[0] / a[0] ** 2, 2 * x[1] / a[1] ** 2, 2 * x[2] / a[2] ** 2])
    n /= np.linalg.norm(n)
    return n


def compute_boundary_forcing(
    theta: float,
    phi: float,
    z: np.ndarray,
    k: float,
    p: np.ndarray,
    semiaxes: tuple[float, float, float],
) -> np.ndarray:
    """Computes h = -n x E^i on the boundary."""
    x = ellipsoid_point(theta, phi, semiaxes)
    Ei = incident_field(x, z, k, p)
    n = ellipsoid_normal(x, semiaxes)
    return -np.cross(n, Ei)


jax.config.update("jax_enable_x64", True)

semiaxes = (4, 4, 6)
k = 2.0

# --- transmitter (one dipole) ---
z = np.array([1.0, 0.0, 0.0])  # on Λ
pol = np.array([0.0, 1.0, 0.0])  # tangential at z (pol·z = 0)

# --- boundary collocation on ∂D (resolve h) ---
n_theta, n_phi = 24, 48
THETA, PHI = np.meshgrid(
    np.linspace(0, np.pi, n_theta),
    np.linspace(0, 2 * np.pi, n_phi, endpoint=False),
    indexing="ij",
)
theta, phi = THETA.ravel(), PHI.ravel()
bnd_points = np.stack(
    [ellipsoid_point(t, p, semiaxes) for t, p in zip(theta, phi, strict=True)]
)
bnd_normals = np.stack([ellipsoid_normal(x, semiaxes) for x in bnd_points])
bnd_forcing = np.stack(
    [
        compute_boundary_forcing(t, p, z, k, pol, semiaxes)
        for t, p in zip(theta, phi, strict=True)
    ]
)

X_train = jnp.asarray(np.concatenate([bnd_points, bnd_normals], axis=1))
y_train = jnp.asarray(bnd_forcing.reshape(-1, 1))

kernel = TangentialMaxwellKernel(n_spectral=64, omega=k)  # F = 128 features
model = GaussianProcess(kernel)

# --- receiver (one point, one polarization) on Λ ---
x_r = np.array([0.0, 1.0, 0.0])
n_r = x_r / np.linalg.norm(x_r)  # |x_r|=1 ⇒ n_Λ = x_r

t_r = np.array([0.0, 0.0, 1.0])  # tangential at x_r
X_query = jnp.asarray(np.concatenate([x_r, n_r])[None, :])  # (1,6)

m = np.asarray(
    model.posterior_mean(X_query, X_train, y_train)
).ravel()  # (3,) = n_Λ × E^s at x_r
T_entry = np.dot(t_r, m)  # eq.(5) tested against t_r
print("T_11 (EP-GP):", T_entry)
