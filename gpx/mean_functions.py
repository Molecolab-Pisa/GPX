# GPX: gaussian process regression in JAX
# Copyright (C) 2023  GPX authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
