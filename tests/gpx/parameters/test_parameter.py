import jax
import jax.numpy as jnp
import pytest

from gpx.bijectors import Softplus
from gpx.parameters import Parameter
from gpx.priors import NormalPrior


def test_value_and_prior_consistency():
    # same shape, same dtype
    p = Parameter(1.0, True, Softplus(), NormalPrior(shape=(), dtype=jnp.float64))
    assert isinstance(p, Parameter)

    # same shape, different dtype
    with pytest.raises(ValueError):
        p = Parameter(
            1,
            True,
            Softplus(),
            NormalPrior(shape=(), dtype=jnp.float64),
        )

    # different shape, same dtype
    with pytest.raises(ValueError):
        p = Parameter(
            1.0,
            True,
            Softplus(),
            NormalPrior(shape=(2, 1), dtype=jnp.float64),
        )

    # different shape, different dtype
    with pytest.raises(ValueError):
        p = Parameter(
            1.0,
            True,
            Softplus(),
            NormalPrior(shape=(2, 1), dtype=jnp.int64),
        )


def test_flatten_unflatten():
    p1 = Parameter(1.0, True, Softplus(), NormalPrior())

    leaves, treedef = jax.tree_util.tree_flatten(p1)
    p2 = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(p2, Parameter)


@pytest.mark.parametrize(
    "update_dict",
    [
        dict(value=2.0),
        dict(value=jnp.array([2.0, 1.0]), prior=NormalPrior(shape=(2,))),
    ],
)
def test_update(update_dict):
    p1 = Parameter(1.0, True, Softplus(), NormalPrior())
    p2 = p1.update(update_dict)
    assert isinstance(p2, Parameter)


@pytest.mark.parametrize(
    "update_dict",
    [
        dict(value=2.0, prior=NormalPrior(shape=(2,))),
        dict(value=1.0, prior=NormalPrior(dtype=jnp.int64)),
    ],
)
def test_update_fails(update_dict):
    p1 = Parameter(1.0, True, Softplus(), NormalPrior())
    with pytest.raises(ValueError):
        p2 = p1.update(update_dict)  # noqa


def test_copy():
    p1 = Parameter(1.0, True, Softplus(), NormalPrior())
    p2 = p1.copy()

    # two different objects
    with pytest.raises(AssertionError):
        assert p1 is p2

    assert p1.value == p2.value

    # modified one parameter does not affect the other
    p1.value = 5.0
    with pytest.raises(AssertionError):
        assert p1.value == p2.value


def test_sample_prior():
    p1 = Parameter(1.0, True, Softplus(), NormalPrior())
    p2 = p1.sample_prior(jax.random.PRNGKey(2023))

    assert isinstance(p2, Parameter)
    assert p2.value != p1.value
