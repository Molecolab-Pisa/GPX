from jax.config import config

from . import kernels, models, parameters

# Always enable double precision when importing this module
config.update("jax_enable_x64", True)
