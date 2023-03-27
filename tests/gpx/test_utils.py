import jax.numpy as jnp
import pytest
from numpy.testing import assert_equal

from gpx.utils import identity, inverse_softplus, softplus

# ============================================================================
# Bijectors tests
# ============================================================================


@pytest.mark.parametrize("value", [-1e4, -1e2, -50.0, 0.0, 1e3])
def test_softplus(value):
    """
    Checks that the output of softplus gives a positive number
    """
    x = jnp.array([value])
    y = softplus(x)
    assert y >= 0


@pytest.mark.parametrize("value", [-600.0, -1e2, 0.0, 1e2, 1e5, 1e10])
def test_inverse_softplus(value):
    """
    Checks that the inverse softplus, applied to the output of softplus,
    recovers the correct input. Note that for the current implementation
    this is limited from below: too negative numbers will not result in
    a perfect recovery of the input.
    """
    x = jnp.array([value])
    y = inverse_softplus(softplus(x))
    assert_equal(x, y)


@pytest.mark.parametrize("value", [-1e10, -1e3, -1e-5, -1e-300])
def test_inverse_softplus_negative(value):
    """
    Checks that the inverse softplus is not defined for negative
    numbers, and gives NaN as a result
    """
    x = jnp.array([value])
    y = inverse_softplus(x)
    assert jnp.isnan(y)


@pytest.mark.parametrize("value", [-1e10, -1e1, 0.0, 1e1, 1e10])
def test_softplus_plus_inversion_stability(value):
    """
    Checks that inverting the output of softplus always yields a
    finite number (not NaN)
    """
    x = jnp.array([value])
    y = inverse_softplus(softplus(x))
    assert jnp.isfinite(y)


def test_identity():
    """
    Checks that the output of identity has the same value as the input
    """
    x = jnp.array([1])
    y = identity(x)
    assert_equal(x, y)
