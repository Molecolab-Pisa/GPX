from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
from jax import Array, random
from jax._src import prng
from jax.typing import ArrayLike


def _check_object_is_callable(obj: Any, name: str) -> None:
    if not callable(obj):
        raise ValueError(f"{name} must be a callable, you provided {type(obj)}")


def _check_object_is_type(obj: Any, ref_type: Any, name: str) -> None:
    if not isinstance(obj, ref_type):
        raise ValueError(
            f"{name} must be a {ref_type} instance, you provided {type(obj)}"
        )


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
