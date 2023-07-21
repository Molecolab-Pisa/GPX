import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpy.testing import assert_equal

from gpx.kernels import SquaredExponential
from gpx.mean_functions import data_mean
from gpx.parameters import ModelState, Parameter
from gpx.priors import NormalPrior
from gpx.utils import inverse_softplus, softplus


def create_model_state():
    state = ModelState(
        kernel=SquaredExponential(),
        mean_function=data_mean,
        params={
            "kernel_params": {
                "lengthscale": Parameter(
                    1.0, True, softplus, inverse_softplus, NormalPrior()
                )
            },
            "sigma": Parameter(0.1, False, softplus, inverse_softplus, NormalPrior()),
        },
    )
    return state


@pytest.mark.parametrize(
    "aux",
    [
        1,
        2.0,
        np.array(3.0),
        jnp.array([4.0]),
        np.array("e"),
        "f",
        np.array([1.0, 2.0]),
        jnp.array([[3, 4]]),
        True,
        False,
        None,
    ],
)
def test_save_and_reload_state(tmp_path, aux):
    d = tmp_path / "sub_model_test"
    d.mkdir()
    state_file = d / "tmp_state_file.npz"
    state = create_model_state()
    # Update the model with the current auxiliary data
    state = state.update({"aux": aux})
    # Save and reload
    state.save(state_file)
    new_state = state.load(state_file)

    # Check that kernel is the same object as before
    assert state.kernel is new_state.kernel

    # Check attributes
    old_params = [state.params["sigma"], state.params["kernel_params"]["lengthscale"]]
    new_params = [
        new_state.params["sigma"],
        new_state.params["kernel_params"]["lengthscale"],
    ]
    for param, new_param in zip(old_params, new_params):
        # Check that they are Parameter instances
        assert isinstance(param, Parameter)
        assert isinstance(new_param, Parameter)

        # Check that the new value is equal to the old one
        assert_equal(param.value, new_param.value)

        # Check that the two values are different objects
        with pytest.raises(AssertionError):
            assert param.value is new_param.value

        # Check that all the other attributes of Parameter are the same
        # Python objects
        assert param.forward_transform is new_param.forward_transform
        assert param.backward_transform is new_param.backward_transform
        assert param.trainable is new_param.trainable

    # Check auxiliary data
    assert hasattr(state, "aux")
    assert hasattr(new_state, "aux")

    if isinstance(state.aux, (np.ndarray, jnp.ndarray)):
        assert np.all(state.aux == new_state.aux)
    else:
        assert state.aux == new_state.aux


def test_shallow_copy():
    state = create_model_state()
    state_copied = state.copy()

    # copy returns two different objects
    with pytest.raises(AssertionError):
        assert state is state_copied

    # kernel is not copied
    assert state.kernel is state_copied.kernel

    # params are copied
    with pytest.raises(AssertionError):
        assert state.params is state_copied.params

    # if the kernel is reassigned to another object,
    # the change takes place only in one state
    state.kernel = softplus
    with pytest.raises(AssertionError):
        assert state.kernel is state_copied.kernel


def test_randomize_model_state():
    state = create_model_state()
    randomized_state = state.randomize(random.PRNGKey(2023))

    # check that the new object is a ModelState class
    assert isinstance(randomized_state, ModelState)

    # the state is not modified inplace (a copy with the .copy() method is made)
    with pytest.raises(AssertionError):
        assert state == randomized_state

    # the kernel is not modified nor copied
    assert state.kernel is randomized_state.kernel

    # check parameters attributes
    old_lensc = state.params["kernel_params"]["lengthscale"]
    new_lensc = randomized_state.params["kernel_params"]["lengthscale"]
    old_sigma = state.params["sigma"]
    new_sigma = randomized_state.params["sigma"]

    for param, new_param in zip([old_lensc, old_sigma], [new_lensc, new_sigma]):
        # check that they are Parameter instances
        assert isinstance(param, Parameter)
        assert isinstance(new_param, Parameter)

        # check that all the other attributes of Parameter are the same python objects
        assert param.forward_transform is new_param.forward_transform
        assert param.backward_transform is new_param.backward_transform
        assert param.trainable is new_param.trainable

    # lenghtscale is trainable:
    # check that the new value is NOT equal to the old one
    with pytest.raises(AssertionError):
        assert_equal(old_lensc.value, new_lensc.value)

    # sigma is not trainable:
    # check that the new value is equal to the old one
    assert_equal(old_sigma.value, new_sigma.value)

    # Check that the two values are different objects
    with pytest.raises(AssertionError):
        assert old_sigma.value is new_sigma.value
