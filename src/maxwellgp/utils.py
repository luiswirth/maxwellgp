import jax.numpy as jnp
from jaxtyping import Array, Float, UInt


def fibonacci_sphere(n: UInt, dtype=jnp.float64) -> Float[Array, "n 3"]:
    k = jnp.arange(n, dtype=dtype) + 0.5
    golden_ratio = (1.0 + jnp.sqrt(5.0)) / 2.0
    phi = 2.0 * jnp.pi / golden_ratio
    z = 1.0 - 2.0 * k / n
    r = jnp.sqrt(jnp.clip(1.0 - z * z, a_min=0.0))
    theta = phi * k
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), z], axis=-1)


def normalize(v: Array, axis: int = 1, eps: float = 1e-12) -> Array:
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)
