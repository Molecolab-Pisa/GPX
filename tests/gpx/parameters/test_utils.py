import jax.numpy as jnp
import numpy as np
import pytest

from gpx.parameters.utils import _is_numeric


# auxiliary function
def identity(x, *args, **kwargs):
    return x


@pytest.mark.parametrize("func_cast", [identity, np.array, jnp.array])
@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
def test_is_numeric_true(func_cast, dtype):
    value = func_cast(1, dtype=dtype)
    assert _is_numeric(value) is True


@pytest.mark.parametrize("func_cast", [identity, np.array])
@pytest.mark.parametrize("value", [True, False, "foo"])
def test_is_numeric_false(func_cast, value):
    value = func_cast(value)
    assert _is_numeric(value) is False
