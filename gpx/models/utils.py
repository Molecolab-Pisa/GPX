from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import Array, random
from jax._src import prng
from jax.typing import ArrayLike


def sample(
    key: prng.PRNGKeyArray,
    mean: ArrayLike,
    cov: ArrayLike,
    n_samples: Optional[int] = 1,
) -> Array:
    if mean.ndim > 1:
        samples = []
        for dim in range(mean.shape[1]):
            subkey, key = random.split(key)
            sample = random.multivariate_normal(
                key=key,
                mean=mean[:, dim],
                cov=cov,
                shape=(n_samples,),
            )
            samples.append(sample)
    else:
        sample = random.multivariate_normal(key=key, mean=mean, cov=cov)
    return jnp.array(sample)
