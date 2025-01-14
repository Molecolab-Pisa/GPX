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
from collections import namedtuple

import jax

GPX_DEFAULTS = {
    # number of steps in stochastic trace estimation
    "num_evals": 5,
    # number of lanczos evaluations
    "num_lanczos": 8,
    # default "random" key for lanczos
    "key_lanczos": jax.random.PRNGKey(2023),
    # default "random" key for preconditioner
    "key_precond": jax.random.PRNGKey(2023),
}

# using a namedtuple to have immutable defaults
gpxargs = namedtuple(
    "GPX_DEFAULTS_ARGUMENTS",
    GPX_DEFAULTS.keys(),
    defaults=GPX_DEFAULTS.values(),
)()
