from collections import namedtuple

import jax

GPX_DEFAULTS = {
    # number of steps in stochastic trace estimation
    "num_evals": 10,
    # number of lanczos evaluations
    "num_lanczos": 8,
    # default "random" key for lanczos
    "lanczos_key": jax.random.PRNGKey(2023),
}

gpxargs = namedtuple(
    "GPX_DEFAULTS_ARGUMENTS",
    GPX_DEFAULTS.keys(),
    defaults=GPX_DEFAULTS.values(),
)()
