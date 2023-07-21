import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

# ============================================================================
# Mean functions
# ============================================================================


def zero_mean(y: ArrayLike) -> Array:
    return jnp.array(0.0)


def data_mean(y: ArrayLike) -> Array:
    return jnp.mean(y)
