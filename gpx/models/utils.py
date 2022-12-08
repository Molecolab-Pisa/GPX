import jax.numpy as jnp
from jax import random


def sample(key, model, x, n_samples=1):
    """Sample from a gaussian process model

    If the model is not trained, returns samples from the
    prior mean and covariance.
    If the model is trained, returns samples from the
    posterior mean and covariance.
    Arguments
    ---------
    key: a PRNG key used as the random key.
    model: a GPX gaussian process model instance.
    x: a JAX.Array of observation locations.
    n_samples: integer specifying the number of samples drawn.
    Returns
    -------
    samples: a JAX.Array of shape (num_y, num_samples, num_observations)
    """
    y_mean, y_cov = model.predict(x, full_covariance=True)
    # small jitter for stabilization
    y_cov = y_cov + 1e-10 * jnp.eye(y_cov.shape[0])
    if y_mean.ndim > 1:
        samples = []
        for dim in range(y_mean.shape[1]):
            subkey, key = random.split(key)
            sample = random.multivariate_normal(
                key=key, mean=y_mean[:, dim], cov=y_cov, shape=(n_samples,)
            )
            samples.append(sample)
        samples = jnp.array(samples)
    else:
        sample = random.multivariate_normal(key=key, mean=y_mean, cov=y_cov)
        samples = jnp.array([sample])
    return samples
