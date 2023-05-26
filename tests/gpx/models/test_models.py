import jax.numpy as jnp
import pytest
from jax import random

from gpx.kernels import SquaredExponential
from gpx.models import GPR, SGPR, RBFNet
from gpx.parameters import ModelState, Parameter
from gpx.priors import NormalPrior
from gpx.utils import identity, inverse_softplus, softplus


def create_gpr_model():
    model_minimal = GPR(
        kernel=SquaredExponential(),
    )

    model = GPR(
        kernel=SquaredExponential(),
        kernel_params={
            "lengthscale": Parameter(
                1.0, True, softplus, inverse_softplus, NormalPrior()
            )
        },
        sigma=Parameter(0.1, False, softplus, inverse_softplus, NormalPrior()),
    )
    return model_minimal, model


def create_sgpr_model():
    X_locs = random.normal(random.PRNGKey(2023), shape=(10, 1))

    model_minimal = SGPR(
        kernel=SquaredExponential(),
        x_locs=Parameter(
            X_locs, False, identity, identity, NormalPrior(shape=X_locs.shape)
        ),
    )

    model = SGPR(
        kernel=SquaredExponential(),
        x_locs=Parameter(
            X_locs, False, identity, identity, NormalPrior(shape=X_locs.shape)
        ),
        kernel_params={
            "lengthscale": Parameter(
                1.0, True, softplus, inverse_softplus, NormalPrior()
            )
        },
        sigma=Parameter(0.1, False, softplus, inverse_softplus, NormalPrior()),
    )
    return model_minimal, model


def create_rbfnet_model():
    I_points = random.normal(random.PRNGKey(2023), shape=(10, 1))

    def diabatization_layer(pred):
        n, _ = pred.shape
        H = jnp.zeros((n, 2, 2))
        H = H.at[:, 0, 0].set(pred[:, 0])
        H = H.at[:, 1, 1].set(pred[:, 1])
        H = H.at[:, 0, 1].set(pred[:, 2])
        H = H.at[:, 1, 0].set(pred[:, 2])
        return jnp.linalg.eigvalsh(H)

    model_minimal = RBFNet(
        key=random.PRNGKey(2023),
        kernel=SquaredExponential(),
        inducing_points=Parameter(
            I_points, False, identity, identity, NormalPrior(shape=I_points.shape)
        ),
        num_output=3,
    )

    model = RBFNet(
        key=random.PRNGKey(2023),
        kernel=SquaredExponential(),
        inducing_points=Parameter(
            I_points, False, identity, identity, NormalPrior(shape=I_points.shape)
        ),
        num_output=3,
        kernel_params={
            "lengthscale": Parameter(
                1.0, True, softplus, inverse_softplus, NormalPrior()
            )
        },
        alpha=Parameter(1.0, True, softplus, inverse_softplus, NormalPrior()),
        output_layer=diabatization_layer,
    )
    return model_minimal, model


# ======================================================================
# Create models from ModelState
# ======================================================================


def create_model_state():
    state_gpr = ModelState(
        kernel=SquaredExponential(),
        params={
            "kernel_params": {
                "lengthscale": Parameter(
                    1.0, True, softplus, inverse_softplus, NormalPrior()
                )
            },
            "sigma": Parameter(0.1, False, softplus, inverse_softplus, NormalPrior()),
        },
    )

    X_locs = random.normal(random.PRNGKey(2023), shape=(10, 1))
    state_sgpr = ModelState(
        kernel=SquaredExponential(),
        params={
            "kernel_params": {
                "lengthscale": Parameter(
                    1.0, True, softplus, inverse_softplus, NormalPrior()
                )
            },
            "sigma": Parameter(0.1, False, softplus, inverse_softplus, NormalPrior()),
        },
        x_locs=Parameter(
            X_locs, False, identity, identity, NormalPrior(shape=X_locs.shape)
        ),
    )

    I_points = random.normal(random.PRNGKey(2023), shape=(10, 1))
    state_rbfnet = ModelState(
        kernel=SquaredExponential(),
        params={
            "alpha": Parameter(1.0, True, softplus, inverse_softplus, NormalPrior()),
            "inducing_points": Parameter(
                I_points, False, identity, identity, NormalPrior(shape=I_points.shape)
            ),
            "kernel_params": {
                "lengthscale": Parameter(
                    1.0, True, softplus, inverse_softplus, NormalPrior()
                )
            },
            "weights": Parameter(
                random.normal(random.PRNGKey(2023), shape=(1, 3)),
                True,
                identity,
                identity,
                NormalPrior(shape=(1, 3)),
            ),
        },
    )
    return state_gpr, state_sgpr, state_rbfnet


@pytest.mark.parametrize(
    "create_model, model",
    [
        (create_gpr_model, GPR),
        (create_sgpr_model, SGPR),
        (create_rbfnet_model, RBFNet),
    ],
)
def test_init_models(create_model, model):
    m1, m2 = create_model()
    assert isinstance(m1, model)
    assert isinstance(m2, model)


@pytest.mark.parametrize(
    "model_class, state",
    [
        (GPR, create_model_state()[0]),
        (SGPR, create_model_state()[1]),
        (RBFNet, create_model_state()[2]),
    ],
)
def test_models_from_state(model_class, state):
    model = model_class.from_state(state)
    assert isinstance(model, model_class)
