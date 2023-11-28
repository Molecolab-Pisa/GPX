from __future__ import annotations

import warnings
from typing import Dict, Optional

from jax import Array
from jax._src import prng
from jax.typing import ArrayLike
from typing_extensions import Self

from ..optimizers import NLoptWrapper, scipy_minimize, scipy_minimize_derivs
from ..parameters import ModelState, Parameter
from .utils import randomized_minimization, randomized_minimization_derivs


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

    def randomize(self, key: prng.PRNGKeyArray, reset: Optional[bool] = True):
        "Creates a new ModelState with randomized parameter values"
        if reset:
            new_state = self.state.randomize(key, opt=self._init_default)
        else:
            new_state = self.state.randomize(key)
        return self.from_state(new_state)

    def sample(
        self,
        key: prng.PRNGKeyArray,
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
        key: prng.PRNGKeyArray,
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
        Returns:
            μ: predicted mean
            C_nn: predicted covariance
        """
        self._check_is_fitted()
        if iterative:
            if full_covariance:
                raise RuntimeError(
                    "full_covariance=True is not compatible with an"
                    " iterative prediction"
                )
            return self._predict_iter_fun(self.state, x=x)
        else:
            return self._predict_dense_fun(
                self.state, x=x, full_covariance=full_covariance
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
        Returns:
            μ: predicted mean
            C_nn: predicted covariance
        """
        self._check_is_fitted()
        if iterative:
            if full_covariance:
                raise RuntimeError(
                    "full_covariance=True is not compatible with an"
                    " iterative prediction"
                )
            return self._predict_derivs_iter_fun(self.state, x=x, jacobian=jacobian)
        else:
            return self._predict_derivs_dense_fun(
                self.state, x=x, jacobian=jacobian, full_covariance=full_covariance
            )

    def log_marginal_likelihood(
        self,
        x: ArrayLike,
        y: ArrayLike,
        return_negative: Optional[bool] = False,
        iterative=False,
        num_evals=None,
        num_lanczos=None,
        lanczos_key=None,
    ) -> Array:
        """Computes the log marginal likelihood.

        Args:
            x: observations
            y: labels
            return_negative: whether to return the negative of the lml
        """
        if iterative:
            lml = self._lml_iter_fun(
                self.state,
                x=x,
                y=y,
                num_evals=num_evals,
                num_lanczos=num_lanczos,
                lanczos_key=lanczos_key,
            )
        else:
            lml = self._lml_dense_fun(self.state, x=x, y=y)

        if return_negative:
            return -lml
        return lml

    def log_marginal_likelihood_derivs(
        self,
        x: ArrayLike,
        y: ArrayLike,
        jacobian: ArrayLike,
        return_negative: Optional[bool] = False,
    ) -> Array:
        """Computes the log marginal likelihood using the hessian kernel.

        Args:
            x: observations
            y: labels
            jacobian: jacobian of x
            return_negative: whether to return the negative of the lml
        """
        lml = self._lml_derivs_dense_fun(self.state, x=x, y=y, jacobian=jacobian)
        if return_negative:
            return -lml
        return lml

    def is_fitted(self):
        return hasattr(self.state, "c")

    def _check_is_fitted(self):
        if not self.is_fitted():
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
        key: prng.PRNGKeyArray = None,
        return_history: Optional[bool] = False,
        iterative: Optional[bool] = False,
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
            minimization_function = scipy_minimize
            self.state, optres, *history = randomized_minimization(
                key=key,
                state=self.state,
                x=x,
                y=y,
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

        if iterative:
            self.state = self._fit_iter_fun(self.state, x=x, y=y)
        else:
            self.state = self._fit_dense_fun(self.state, x=x, y=y)

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
        key: prng.PRNGKeyArray = None,
        return_history: Optional[bool] = False,
        iterative: Optional[bool] = False,
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
            minimization_function = scipy_minimize_derivs
            self.state, optres, *history = randomized_minimization_derivs(
                key=key,
                state=self.state,
                x=x,
                y=y,
                jacobian=jacobian,
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

        if iterative:
            self.state = self._fit_derivs_iter_fun(
                self.state, x=x, y=y, jacobian=jacobian
            )
        else:
            self.state = self._fit_derivs_dense_fun(
                self.state, x=x, y=y, jacobian=jacobian
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
        minimize=True,
        key=None,
        num_restarts=0,
        return_history=False,
        iterative: Optional[bool] = False,
    ) -> Self:
        if minimize:
            minimization_function = opt.optimize
            self.state, optres, *history = randomized_minimization(
                key=key,
                state=self.state,
                x=x,
                y=y,
                minimization_function=minimization_function,
                num_restarts=num_restarts,
                return_history=return_history,
            )
            self.optimize_results_ = optres

            if optres < 0:
                warnings.warn(
                    "optimization returned with error: {:d}".format(optres),
                    stacklevel=2,
                )

        if iterative:
            self.state = self._fit_iter_fun(self.state, x=x, y=y)
        else:
            self.state = self._fit_dense_fun(self.state, x=x, y=y)

        if return_history:
            self.states_history_ = history[0]
            self.losses_history_ = history[1]

        return self
