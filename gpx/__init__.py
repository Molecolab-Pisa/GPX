from jax.config import config

# Always enable double precision when importing this module
config.update("jax_enable_x64", True)
