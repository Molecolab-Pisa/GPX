import jax

from . import kernels, models, parameters

# Always enable double precision when importing this module
jax.config.update("jax_enable_x64", True)
