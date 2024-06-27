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
from __future__ import annotations

import warnings
from typing import Dict, Optional

from jax import Array
from jax.typing import ArrayLike
from typing_extensions import Self

from ..defaults import gpxargs
from ..optimizers import (
    NLoptWrapper,
    scipy_minimize,
    scipy_minimize_derivs,
    scipy_minimize_ol,
)
from ..parameters import ModelState, Parameter
from .utils import (
    loss_fn_with_args,
    randomized_minimization,
    randomized_minimization_derivs,
    randomized_minimization_ol,
)

KeyArray = Array


class BaseGP:
    def __repr__(self):
        return repr(self.print())

    @property
    def c_(self):
        return self.state.c

    @property
    def mu_(self):
        return self.state.mu

    @property
    def x_train(self):
        return self.state.x_train

    @property
    def y_train(self):
        return self.state.y_train

    @property
    def jacobian_train(self):
        return self.state.jacobian_train

    @classmethod
    def from_state(cls, state: ModelState):
        "Instantiate the GP class from a ModelState"
        self = cls.__new__(cls)
        self.state = state
        return self

    def save(self, state_file: str) -> Dict:
        "Saves the ModelState values to a file"
        return self.state.save(state_file)

    def load(self, state_file: str) -> Self:
        "Loads the ModelState values from file"
        self.state = self.state.load(state_file)
        return self

    def init(self, *args, **kwargs) -> ModelState:
        "Resets the model"
        return self._init_fun(*args, **kwargs)

    def default_params(self) -> Dict[str, Parameter]:
        "Default model parameters"
        return self._default_params_fun()

    def randomize(self, key: KeyArray, reset: Optional[bool] = True):
        "Creates a new ModelState with randomized parameter values"
        if reset:
            new_state = self.state.randomize(key, opt=self._init_default)
        else:
            new_state = self.state.randomize(key)
        return self.from_state(new_state)

    def sample(
        self,
        key: KeyArray,
        x: ArrayLike,
        n_samples: Optional[int] = 1,
        kind: Optional[str] = "prior",
    ) -> Array:
        """Draws samples from a GP.

        Args:
            key: JAX PRNGKey
            x: observations
            n_samples: number of samples to draw
            kind: whether to draw samples from the prior ('prior')
                  or from the posterior ('posterior')
        Returns:
            samples: drawn samples
        """
        if kind == "prior":
            return self._sample_prior_fun(
                key, state=self.state, x=x, n_samples=n_samples
            )
        elif kind == "posterior":
            return self._sample_posterior_fun(
                key, state=self.state, x=x, n_samples=n_samples
            )
        else:
            raise ValueError(
                f"kind can be either 'prior' or 'posterior', you provided {kind}"
            )

    def sample_derivs(
        self,
        key: KeyArray,
        x: ArrayLike,
        jacobian: ArrayLike,
        n_samples: Optional[int] = 1,
        kind: Optional[str] = "prior",
    ) -> Array:
        """Draws samples from a GP using the hessian kernel.

        Args:
            key: JAX PRNGKey
            x: observations
            jacobian: jacobian of x
            n_samples: number of samples to draw
            kind: whether to draw samples from the prior ('prior')
                  or from the posterior ('posterior')
        Returns:
            samples: drawn samples
        """
        if kind == "prior":
            return self._sample_prior_derivs_fun(
                key, state=self.state, x=x, jacobian=jacobian, n_samples=n_samples
            )
        elif kind == "posterior":
            return self._sample_posterior_derivs_fun(
                key, state=self.state, x=x, jacobian=jacobian, n_samples=n_samples
            )
        else:
            raise ValueError(
                f"kind can be either 'prior' or 'posterior', you provided {kind}"
            )

    def print(self) -> None:
        "Print the model parameters"
        return self.state.print_params()

    def predict(
        self,
        x: ArrayLike,
        full_covariance: Optional[bool] = False,
        iterative: Optional[bool] = False,
    ) -> Array:
        """Predicts the output on new data

        Args:
            x: observations
            full_covariance: whether to return the covariance matrix too
            iterative: whether to predict iteratively
                       (i.e., the kernel is never instantiated)
        Returns:
            μ: predicted mean
            C_nn: predicted covariance
        """
        self._check_is_fitted()
        return self._predict_fun(
            self.state, x=x, full_covariance=full_covariance, iterative=iterative
        )

    def predict_derivs(
        self,
        x: ArrayLike,
        jacobian: ArrayLike,
        full_covariance: Optional[bool] = False,
        iterative: Optional[bool] = False,
    ) -> Array:
        """Predicts the output derivatives on new data.

        Args:
            x: observations
            jacobian: jacobian of x
            full_covariance: whether to return the covariance matrix too
            iterative: whether to predict iteratively
                       (i.e., the kernel is never instantiated)
        Returns:
            μ: predicted mean
            C_nn: predicted covariance
        """
        self._check_is_fitted()
        return self._predict_derivs_fun(
            self.state,
            x=x,
            jacobian=jacobian,
            full_covariance=full_covariance,
            iterative=iterative,
        )

    def predict_y_derivs(
        self,
        x: ArrayLike,
        full_covariance: Optional[bool] = False,
        iterative: Optional[bool] = False,
    ) -> Array:
        """Predicts the output derivatives on new data.

        Args:
            x: observations
            full_covariance: whether to return the covariance matrix too
            iterative: whether to predict iteratively
                       (i.e., the kernel is never instantiated)
        Returns:
            μ: predicted mean
            C_nn: predicted covariance
        """
        self._check_is_fitted()
        return self._predict_y_derivs_fun(
            self.state,
            x=x,
            full_covariance=full_covariance,
            iterative=iterative,
        )

    def predict_ol(
        self,
        x: ArrayLike,
        jacobian: ArrayLike,
        full_covariance: Optional[bool] = False,
    ) -> Array:
        """predicts with operator learning gaussian process

        Args:
            x: observations
            full_covariance: whether to return the covariance matrix too
        Returns:
            μ: predicted mean
            ∂μ/∂x: predicted derivative
            C_nn: predicted covariance
        """
        self._check_is_fitted()
        return self._predict_ol_fun(
            self.state,
            x=x,
            jacobian=jacobian,
            full_covariance=full_covariance,
        )

    def log_marginal_likelihood(
        self,
        x: ArrayLike,
        y: ArrayLike,
        return_negative: Optional[bool] = False,
        iterative: Optional[bool] = False,
        num_evals: Optional[int] = gpxargs.num_evals,
        num_lanczos: Optional[int] = gpxargs.num_lanczos,
        lanczos_key: Optional[ArrayLike] = gpxargs.lanczos_key,
    ) -> Array:
        """Computes the log marginal likelihood.

        Args:
            x: observations
            y: labels
            return_negative: whether to return the negative of the lml
            iterative: whether to compute the lml iteratively
                       (e.g., never instantiating the kernel)
            num_evals: number of monte carlo evaluations for estimating
                       log|K| (used only if iterative=True)
            num_lanczos: number of Lanczos evaluations for estimating
                         log|K| (used only if iterative=True)
            lanczos_key: random key for Lanczos tridiagonalization
        """
        lml = self._lml_fun(
            self.state,
            x=x,
            y=y,
            iterative=iterative,
            num_evals=num_evals,
            num_lanczos=num_lanczos,
            lanczos_key=lanczos_key,
        )

        if return_negative:
            return -lml
        return lml

    def log_marginal_likelihood_derivs(
        self,
        x: ArrayLike,
        y: ArrayLike,
        jacobian: ArrayLike,
        return_negative: Optional[bool] = False,
        iterative: Optional[bool] = False,
        num_evals: Optional[int] = gpxargs.num_evals,
        num_lanczos: Optional[int] = gpxargs.num_lanczos,
        lanczos_key: Optional[ArrayLike] = gpxargs.lanczos_key,
    ) -> Array:
        """Computes the log marginal likelihood using the hessian kernel.

        Args:
            x: observations
            y: labels
            jacobian: jacobian of x
            return_negative: whether to return the negative of the lml
            iterative: whether to compute the lml iteratively
                       (e.g., never instantiating the kernel)
            num_evals: number of monte carlo evaluations for estimating
                       log|K| (used only if iterative=True)
            num_lanczos: number of Lanczos evaluations for estimating
                         log|K| (used only if iterative=True)
            lanczos_key: random key for Lanczos tridiagonalization
        """
        lml = self._lml_derivs_fun(
            self.state,
            x=x,
            jacobian=jacobian,
            y=y,
            iterative=iterative,
            num_evals=num_evals,
            num_lanczos=num_lanczos,
            lanczos_key=lanczos_key,
        )

        if return_negative:
            return -lml
        return lml

    def mse_loss(
        self,
        x: ArrayLike,
        y: ArrayLike,
        jacobian: ArrayLike,
        y_derivs: ArrayLike,
        coeff: ArrayLike = 1.0,
    ) -> Array:
        """Computes the mean squared loss for the target and its derivative

            mse = 1/N * Σ_i(μ_i - y_i)**2
                 + coeff * 1/N * 1/Natoms * Σ_j||∂μ_j/∂x - ∂y_j/∂x||**2

        Args:
            state: model state
            x: observations
            y: labels
            jacobian: jacobian of x
            y_derivs: derivatives of y
            coeff: coefficient to scale forces error
        """
        return self.mse_fun(
            self.state,
            x=x,
            y=y,
            jacobian=jacobian,
            y_derivs=y_derivs,
            coeff=coeff,
        )

    def is_fitted(self):
        return hasattr(self.state, "c")

    def is_fitted_derivs(self):
        return hasattr(self.state, "c")

    def _check_is_fitted(self):
        if not self.is_fitted() and not self.is_fitted_derivs:
            class_name = self.__class__.__name__
            raise RuntimeError(
                f"{class_name} is not fitted yet."
                "Call 'fit' before using this model for prediction."
            )

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        minimize: Optional[bool] = True,
        num_restarts: Optional[int] = 0,
        key: Optional[ArrayLike] = None,
        return_history: Optional[bool] = False,
        iterative: Optional[bool] = False,
        loss_kwargs: Optional[Dict] = None,
        opt_kwargs: Optional[Dict] = None,
        n_pivots: Optional[int] = None,
        key_precond: Optional[KeyArray] = None
    ) -> Self:
        """fits the model

        Fits the model, optimizing the hyperparameters if requested.
        The optimization is carried out with scipy's L-BFGS.
        Multiple restarts with random initializations from the
        hyperparameter's prior can be performed.

        Args:
            x: observations
            y: labels
            minimize: whether to tune the parameters to optimize the loss
            num_restarts: number of restarts with randomization to do.
                          If 0, the model is fitted once without any randomization.
            key: random key used to randomize the model when doing restarts
            return_history: whether to return all the losses and model states
                            obtained from the random restarts.
            iterative: whether to fit the model iteratively
                       WARNING: this refers *only* to the fit of linear
                       coefficients. If the loss instantiates the full kernel,
                       this keyword cannot avoid that. If you train using
                       the lml provided by GPX, run with `iterable=True` in
                       the loss_kwargs, along with the other options for Lanczos.
            loss_kwargs: additional keyword arguments passed to the loss
                         function.

        Notes:

        (1) m(y) is the mean function of the real distribution of data. By default,
            we don't make assumptions on the mean of the prior distribution, so it
            is set to the mean value of the input y:

                 μ = (1/n) Σ_i y_i.

        (2) Randomized_minimization requires to optimize the loss.
            In order to optimize with randomized restarts you need to provide a valid
            JAX PRNGKey.
        """
        # we tell the loss that it should be iterative
        # note that this overrides an eventual 'iterative' keyword
        if loss_kwargs is None:
            loss_kwargs = {}
        loss_kwargs["iterative"] = iterative
        loss_fn = loss_fn_with_args(self.state.loss_fn, loss_kwargs)

        if minimize:
            minimization_function = scipy_minimize
            self.state, optres, *history = randomized_minimization(
                key=key,
                state=self.state,
                x=x,
                y=y,
                loss_fn=loss_fn,
                minimization_function=minimization_function,
                num_restarts=num_restarts,
                return_history=return_history,
                opt_kwargs=opt_kwargs,
            )
            self.optimize_results_ = optres

            # if the optimization is failed, print a warning
            if not optres.success:
                warnings.warn(
                    "optimization returned with error: {:d}. ({:s})".format(
                        optres.status, optres.message
                    ),
                    stacklevel=2,
                )

        self.state = self._fit_fun(self.state, x=x, y=y, iterative=iterative, n_pivots=n_pivots, key_precond=key_precond)

        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self

    def fit_derivs(
        self,
        x: ArrayLike,
        y: ArrayLike,
        jacobian: ArrayLike,
        minimize: Optional[bool] = True,
        num_restarts: Optional[int] = 0,
        key: Optional[ArrayLike] = None,
        return_history: Optional[bool] = False,
        iterative: Optional[bool] = False,
        loss_kwargs: Optional[Dict] = None,
        opt_kwargs: Optional[Dict] = None,
        n_pivots: Optional[int] = None,
        key_precond: Optional[KeyArray] = None
    ) -> Self:
        """fits the model

        Fits the model, optimizing the hyperparameters if requested.
        The fit is performed using the hessian kernel and derivatives
        of the output.
        The optimization is carried out with scipy's L-BFGS.
        Multiple restarts with random initializations from the
        hyperparameter's prior can be performed.

        Args:
            x: observations
            y: labels
            jacobian: jacobian of x
            minimize: whether to tune the parameters to optimize the
                          log marginal likelihood
            num_restarts: number of restarts with randomization to do.
                          If 0, the model is fitted once without any randomization.
            key: random key used to randomize the model when doing restarts
            return_history: whether to return all the losses and model states
                            obtained from the random restarts.
            iterative: whether to fit the model iteratively
                       WARNING: this refers *only* to the fit of linear
                       coefficients. If the loss instantiates the full kernel,
                       this keyword cannot avoid that. If you train using
                       the lml provided by GPX, run with `iterable=True` in
                       the loss_kwargs, along with the other options for Lanczos.
            loss_kwargs: additional keyword arguments passed to the loss
                         function.

        Notes:

        (1) m(y) is the mean function of the real distribution of data. By default,
            we don't make assumptions on the mean of the prior distribution, so it
            is set to the mean value of the input y:

                 μ = (1/n) Σ_i y_i.

        (2) Randomized_minimization requires to optimize the loss.
            In order to optimize with randomized restarts you need to provide a valid
            JAX PRNGKey.
        """
        # we tell the loss that it should be iterative
        if loss_kwargs is None:
            loss_kwargs = {}
        loss_kwargs["iterative"] = iterative
        loss_fn = loss_fn_with_args(self.state.loss_fn, loss_kwargs)

        if minimize:
            minimization_function = scipy_minimize_derivs
            self.state, optres, *history = randomized_minimization_derivs(
                key=key,
                state=self.state,
                x=x,
                y=y,
                jacobian=jacobian,
                loss_fn=loss_fn,
                minimization_function=minimization_function,
                num_restarts=num_restarts,
                return_history=return_history,
                opt_kwargs=opt_kwargs,
            )
            self.optimize_results_ = optres

            # if the optimization is failed, print a warning
            if not optres.success:
                warnings.warn(
                    "optimization returned with error: {:d}. ({:s})".format(
                        optres.status, optres.message
                    ),
                    stacklevel=2,
                )

        self.state = self._fit_derivs_fun(
            self.state,
            x=x,
            y=y,
            jacobian=jacobian,
            iterative=iterative,
            n_pivots=n_pivots,
            key_precond=key_precond,
        )

        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self

    def fit_ol(
        self,
        x: ArrayLike,
        y: ArrayLike,
        jacobian: ArrayLike,
        y_derivs: ArrayLike,
        minimize: Optional[bool] = True,
        num_restarts: Optional[int] = 0,
        key: Optional[ArrayLike] = None,
        return_history: Optional[bool] = False,
    ) -> Self:
        """fits the model

        Fits the model, optimizing the hyperparameters if requested.
        The optimization is carried out with scipy's L-BFGS.
        Multiple restarts with random initializations from the
        hyperparameter's prior can be performed.

        Args:
            x: observations
            y: labels
            minimize: whether to tune the parameters to optimize the loss
            num_restarts: number of restarts with randomization to do.
                          If 0, the model is fitted once without any randomization.

        Notes:

        (1) m(y) is the mean function of the real distribution of data. By default,
            we don't make assumptions on the mean of the prior distribution, so it
            is set to the mean value of the input y:

                 μ = (1/n) Σ_i y_i.

        (2) Randomized_minimization requires to optimize the loss.
            In order to optimize with randomized restarts you need to provide a valid
            JAX PRNGKey.
        """
        if minimize:
            minimization_function = scipy_minimize_ol
            self.state, optres, *history = randomized_minimization_ol(
                key=key,
                state=self.state,
                x=x,
                y=y,
                jacobian=jacobian,
                y_derivs=y_derivs,
                loss_fn=self.state.loss_fn,
                minimization_function=minimization_function,
                num_restarts=num_restarts,
                return_history=return_history,
            )
            self.optimize_results_ = optres

            # if the optimization is failed, print a warning
            if not optres.success:
                warnings.warn(
                    "optimization returned with error: {:d}. ({:s})".format(
                        optres.status, optres.message
                    ),
                    stacklevel=2,
                )

        self.state = self._fit_ol_fun(
            self.state, x=x, y=y, y_derivs=y_derivs, jacobian=jacobian
        )

        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self

    def fit_nlopt(
        self,
        x: ArrayLike,
        y: ArrayLike,
        opt: NLoptWrapper,
        minimize: Optional[bool] = True,
        key: Optional[ArrayLike] = None,
        num_restarts: Optional[int] = 0,
        return_history: Optional[bool] = False,
        iterative: Optional[bool] = False,
        loss_kwargs: Optional[Dict] = None,
        opt_kwargs: Optional[Dict] = None,
    ) -> Self:
        # we tell the loss that it should be iterative
        if loss_kwargs is None:
            loss_kwargs = {}
        loss_kwargs["iterative"] = iterative
        loss_fn = loss_fn_with_args(self.state.loss_fn, loss_kwargs)

        if minimize:
            minimization_function = opt.optimize
            self.state, optres, *history = randomized_minimization(
                key=key,
                state=self.state,
                x=x,
                y=y,
                loss_fn=loss_fn,
                minimization_function=minimization_function,
                num_restarts=num_restarts,
                return_history=return_history,
                opt_kwargs=opt_kwargs,
            )
            self.optimize_results_ = optres

            if optres < 0:
                warnings.warn(
                    "optimization returned with error: {:d}".format(optres),
                    stacklevel=2,
                )

        self.state = self._fit_fun(self.state, x=x, y=y, iterative=iterative)

        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self
