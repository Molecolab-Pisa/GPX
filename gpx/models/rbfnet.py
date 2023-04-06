from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, random
from jax._src import prng
from jax.typing import ArrayLike
from typing_extensions import Self

from ..optimize import scipy_minimize
from ..parameters import ModelState
from ..parameters.parameter import Parameter, parse_param
from ..utils import identity, inverse_softplus, softplus


@partial(jit, static_argnums=[3, 4])
def _train_loss(
    params: Dict[str, Parameter],
    x: ArrayLike,
    y: ArrayLike,
    kernel: Callable,
    output_layer: Callable,
) -> Array:
    y_pred = _predict(params=params, x=x, kernel=kernel, output_layer=output_layer)

    alpha = params["alpha"].value
    weights = params["weights"].value
    n_samples = y.shape[0]

    return (
        jnp.mean((y_pred - y) ** 2)
        + alpha * jnp.sum(weights.ravel().T @ weights.ravel()) / n_samples
    )


def train_loss(state: ModelState, x: ArrayLike, y: ArrayLike) -> Array:
    return _train_loss(
        params=state.params,
        x=x,
        y=y,
        kernel=state.kernel,
        output_layer=state.output_layer,
    )


@partial(jit, static_argnums=[2])
def _predict_linear(
    params: Dict[str, Parameter], x: ArrayLike, kernel: Callable
) -> Array:
    kernel_params = params["kernel_params"]
    weights = params["weights"].value
    x_inducing = params["inducing_points"].value

    gram = kernel(x, x_inducing, kernel_params)
    pred = jnp.dot(gram, weights)

    return pred


@partial(jit, static_argnums=[2, 3])
def _predict(
    params: Dict[str, Parameter],
    x: ArrayLike,
    kernel: Callable,
    output_layer: Callable,
) -> Array:
    pred = _predict_linear(params=params, x=x, kernel=kernel)
    pred = output_layer(pred)

    return pred


def predict(state: ModelState, x: ArrayLike, linear_only: bool = False) -> Array:
    if linear_only:
        return _predict_linear(
            params=state.params,
            x=x,
            kernel=state.kernel,
        )
    else:
        return _predict(
            params=state.params,
            x=x,
            kernel=state.kernel,
            output_layer=state.output_layer,
        )


def init(
    key: prng.PRNGKeyArray,
    kernel: Callable,
    kernel_params: Dict[str, Tuple],
    inducing_points: Tuple,
    num_output: int,
    output_layer: Callable = identity,
    alpha: Tuple = (1.0, True, softplus, inverse_softplus),
    loss_fn: Callable = train_loss,
) -> ModelState:
    if not callable(kernel):
        raise RuntimeError(
            f"kernel must be provided as a callable function, you provided"
            f" {type(kernel)}"
        )

    if not isinstance(kernel_params, dict):
        raise RuntimeError(
            f"kernel_params must be provided as a dictionary, you provided"
            f" {type(kernel_params)}"
        )

    inducing_points = parse_param(inducing_points)

    # number of inducing points
    num_input = inducing_points.value.shape[0]

    weights = Parameter(
        random.normal(key, shape=(num_input, num_output)),
        True,
        identity,
        identity,
    )

    kp = {}
    for key in kernel_params:
        param = kernel_params[key]
        kp[key] = parse_param(param)

    alpha = parse_param(alpha)

    # Careful, as here the order matters (thought it shouldn't for a good api)
    params = {
        "alpha": alpha,
        "inducing_points": inducing_points,
        "kernel_params": kp,
        "weights": weights,
    }
    opt = {"output_layer": output_layer, "loss_fn": loss_fn}

    return ModelState(kernel, params, **opt)


# =============================================================================
# RBF Network: interface
# =============================================================================


class RadialBasisFunctionNetwork:
    "Radial Basis Function Network"

    def __init__(
        self,
        key: prng.PRNGKeyArray,
        kernel: Callable,
        kernel_params: Dict[str, Tuple],
        inducing_points: Tuple,
        num_output: int,
        output_layer: Callable = identity,
        alpha: Tuple = (1.0, True, softplus, inverse_softplus),
        loss_fn: Callable = train_loss,
    ) -> None:
        """
        Args:
            key: JAX random PRNGKey.
                 used for instantiating the model weights (they
                 are initialized by sampling a normal distribution)
            kernel: kernel function.
            kernel_params: kernel parameters.
                           should be provided as a dictionary mapping
                           the parameter name to a 4-tuple, which specifies
                           the value, whether the parameter is trainable, and the
                           forward and backward transformation functions for the
                           parameter.
            inducing_points: inducing points.
                             the kernel matrix is evaluated between points provided
                             by the user and the inducing points.
                             if the number of inducing points is smaller than the
                             train set points, then the model is sparse.
                             should be provided as a 4-tuple, as they are casted as
                             a parameter of the model.
            num_output: number of outputs of the RBF layer.
            output_layer: output_layer, taking as input the prediction of the RBF
                          layer, and outputting a transformed prediction.
                          should accept a ArrayLike with shape (n, num_output),
                          where n is the number of samples, and should output another
                          ArrayLike of shape (n, num_output')
            alpha: regularization parameter of the L2 regularization term in the
                   default loss function. If another loss is used, alpha is ignored.
            loss_fn: loss function used to optimize the model parameters.
                     by default, it minimizes the squared error plus the L2
                     regularization term.
        """
        self.key = key
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.inducing_points = inducing_points
        self.num_output = num_output
        self.alpha = alpha
        self.output_layer = output_layer
        self.loss_fn = loss_fn
        self.state = init(
            key=key,
            kernel=kernel,
            kernel_params=kernel_params,
            inducing_points=inducing_points,
            num_output=num_output,
            alpha=alpha,
            output_layer=output_layer,
            loss_fn=loss_fn,
        )

    def print(self) -> None:
        "prints the model parameters"
        return self.state.print_params()

    def fit(self, x: ArrayLike, y: ArrayLike) -> Self:
        self.state, optres = scipy_minimize(
            self.state, x=x, y=y, loss_fn=self.state.loss_fn
        )
        self.optimize_results_ = optres
        self.x_train = x

    def predict(self, x: ArrayLike, linear_only: bool = False) -> Array:
        return predict(self.state, x=x, linear_only=linear_only)


# Alias
RBFNet = RadialBasisFunctionNetwork
