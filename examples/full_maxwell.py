import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from maxwellgp import FullMaxwellKernel, GaussianProcess
from maxwellgp.utils import normalize

# Enable x64
jax.config.update("jax_enable_x64", True)


def get_ground_truth(X, omega):
    EE0s = jnp.array(
        [[-2, 0, 1], [1, 1, 0], [1, -1, -1], [3, 2, 1], [-7, 2, 3]], dtype=jnp.float64
    )
    k0_dirs = jnp.array(
        [[1, 0, 2], [0, 0, 1], [0, -1, 1], [-1, 1, 1], [0, 3, -2]], dtype=jnp.float64
    )

    k_norm = normalize(k0_dirs)
    k_vecs = k_norm * omega
    phases = jnp.exp(1j * jnp.dot(X, k_vecs.T))
    BB0s = jnp.cross(k_norm, EE0s)

    E_tot = jnp.dot(phases, EE0s.astype(jnp.complex128))
    B_tot = jnp.dot(phases, BB0s.astype(jnp.complex128))
    return jnp.concatenate([E_tot, B_tot], axis=-1)


def main():
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)

    # 1. Data Gen
    omega = 2.0 * jnp.pi
    axis = jnp.linspace(-1, 1, 20, dtype=jnp.float64)  # Reduced size for speed in demo
    X1, X2, X3 = jnp.meshgrid(axis, axis, axis, indexing="ij")
    X_total = jnp.stack([X1.ravel(), X2.ravel(), X3.ravel()], axis=-1)

    y_truth_matrix = get_ground_truth(X_total, omega)

    n_train = 100
    indices = jax.random.permutation(k1, X_total.shape[0])[:n_train]
    X_train = X_total[indices]
    y_train_flat = y_truth_matrix[indices].reshape(-1, 1)

    # 2. Model Init
    kernel = FullMaxwellKernel(n_spectral=12, omega=omega, key=k2)
    # Note: We pass X_train, but it is stored as a static field now
    model = GaussianProcess(kernel, log_eps_init=-12.0)

    # 3. Optimizer Setup
    # We partition parameters to apply different settings
    lr_map, lr_gp = 2e-3, 5e-3

    # More robust: filter by instance
    filter_spec = jax.tree.map(lambda _: "gp", model)
    filter_spec = eqx.tree_at(
        lambda m: m.kernel.feature_map,
        filter_spec,
        jax.tree.map(lambda _: "map", model.kernel.feature_map),
    )

    optim = optax.multi_transform(
        {
            "map": optax.adam(lr_map),
            "gp": optax.adam(lr_gp),
        },
        filter_spec,
    )

    # Filter out static fields (Equinox handles this, but optax expects pure arrays)
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optim.init(params)

    # 4. Update Step
    @eqx.filter_jit
    def step(model, opt_state, X, y):
        def loss_fn(m):
            return m.nlml(X, y)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = optim.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)

        new_model = eqx.tree_at(
            lambda m: m.kernel.log_w,
            new_model,
            jnp.clip(new_model.kernel.log_w, -20.0, 10.0),
        )

        return loss, new_model, new_opt_state

    # 5. Loop
    print(f"Training on {n_train} points...")
    for i in range(1001):
        loss_val, model, opt_state = step(model, opt_state, X_train, y_train_flat)

        if i % 100 == 0:
            noise_val = jnp.exp(model.log_eps)
            mu_train = model.posterior_mean(X_train, X_train, y_train_flat)
            train_rmse = jnp.sqrt(jnp.mean((mu_train.real - y_train_flat.real) ** 2))
            print(
                f"[{i:04d}] NLML: {loss_val.item():.4e} | "
                f"eps: {noise_val:.2e} | Train RMSE: {train_rmse:.4e}"
            )

    # 6. Eval
    mu_flat = model.posterior_mean(X_total, X_train, y_train_flat)
    mu_matrix = mu_flat.reshape(X_total.shape[0], 6)
    diff = mu_matrix - y_truth_matrix
    rmse_complex = jnp.sqrt(jnp.mean((diff.conj() * diff).real))
    print(f"\nFinal RMSE (Complex): {rmse_complex.item():.4e}")


if __name__ == "__main__":
    main()
