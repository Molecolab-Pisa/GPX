import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

from scipy.optimize import minimize

from ..utils import constrain_parameters, uncostrain_parameters, split_params, print_model


# =============================================================================
# Standard Gaussian Process Regression: functions
# =============================================================================


def log_marginal_likelihood(params, x, y, kernel, return_negative=False):
    """
    Computes the log marginal likelihood for standard Gaussian Process Regression.
    Arguments
    ---------
    params  : dict
            Dictionary of parameters. Should have a 'kernel_params' keyword
            to specify kernel parameters (a ictionary) and a 'sigma' keyword
            to specify the noise.
    x       : jnp.ndarray, (M, F)
            Input matrix of M samples and F features
    y       : jnp.ndarray, (M, 1)
            Target matrix of M samples and 1 target
    kernel  : callable
            Kernel function
    Returns
    -------
    lml     : jnp.ndarray, ()
            Log marginal likelihood
    """

    kernel_params, sigma = split_params(params)

    y_mean = jnp.mean(y)
    y = y - y_mean
    m = y.shape[0]

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(m)
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)
    L_m = jsp.linalg.cholesky(C_mm, lower=True)
    mll = (
        -0.5 * jnp.squeeze(jnp.dot(y.T, c))
        - jnp.sum(jnp.log(jnp.diag(L_m)))
        - m * 0.5 * jnp.log(2.0 * jnp.pi)
    )
    if return_negative:
        return -mll
    return mll


def fit(params, x, y, kernel):
    """
    Fits a Gaussian Process Regression model.
    Arguments
    ---------
    params  : dict
            Dictionary of parameters. Should have a 'kernel_params' keyword
            to specify kernel parameters (a ictionary) and a 'sigma' keyword
            to specify the noise.
    x       : jnp.ndarray, (M, F)
            Input matrix of M samples and F features
    y       : jnp.ndarray, (M, 1)
            Target matrix of M samples and 1 target
    kernel  : callable
            Kernel function
    Returns
    -------
    c       : jnp.ndarray, (M, 1)
            Dual coefficients
    y_mean  : jnp.ndarray, ()
            Target mean
    """

    kernel_params, sigma = split_params(params)

    y_mean = jnp.mean(y)
    y = y - y_mean

    C_mm = kernel(x, x, kernel_params) + sigma**2 * jnp.eye(x.shape[0])
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)

    return c, y_mean


def predict(
    params,
    x_train,
    x,
    c,
    y_mean,
    kernel,
    full_covariance=False,
):
    """
    Predict with a Gaussian Process Regression model.
    Arguments
    ---------
    params          : dict
                    Dictionary of parameters. Should have a 'kernel_params' keyword
                    to specify kernel parameters (a ictionary) and a 'sigma' keyword
                    to specify the noise.
    x               : jnp.ndarray, (M, F)
                    Input matrix of M samples and F features
    y               : jnp.ndarray, (M, 1)
                    Target matrix of M samples and 1 target
    c               : jnp.ndarray, (M, 1)
                    Dual coefficients
    y_mean          : jnp.ndarray, ()
                    Target mean
    kernel          : callable
                    Kernel function
    full_covariance : bool
                    Whether to return also the full posterior covariance
    Returns
    -------
    mu              : jnp.ndarray, (M, 1)
                    Predicted mean
    C_nn            : jnp.ndarray, (M, M)
                    Predicted covariance
    """

    kernel_params, sigma = split_params(params)

    K_mn = kernel(x_train, x, kernel_params)
    mu = jnp.dot(c.T, K_mn).reshape(-1, 1) + y_mean

    if full_covariance:
        C_mm = kernel(x_train, x_train, kernel_params) + sigma**2 * jnp.eye(
            x_train.shape[0]
        )
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = kernel(x, x, kernel_params) - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn

    return mu


# =============================================================================
# Standard Gaussian Process Regression: interface
# =============================================================================


class GaussianProcessRegression:
    def __init__(self, kernel, kernel_params, sigma):

        self.kernel = kernel
        self.kernel_params = tree_map(lambda p: jnp.array(p), kernel_params)
        self.sigma = jnp.array(sigma)

        self.params = {"sigma": self.sigma, "kernel_params": self.kernel_params}
        self.params_uncostrained = uncostrain_parameters(self.params)

    def print(self, **kwargs):
        return print_model(self, **kwargs)

    def log_marginal_likelihood(self, x, y, return_negative=False):
        return log_marginal_likelihood(
            self.params, x, y, kernel=self.kernel, return_negative=return_negative
        )

    def fit(self, x, y):

        x0, treedef = tree_flatten(self.params_uncostrained)

        def loss(xt):
            params = tree_unflatten(treedef, xt)
            params = constrain_parameters(params)
            return log_marginal_likelihood(
                params, x=x, y=y, kernel=self.kernel, return_negative=True
            )

        grad_loss = grad(loss)

        optres = minimize(loss, x0, method="L-BFGS-B", jac=grad_loss)

        self.params_uncostrained = tree_unflatten(treedef, optres.x)
        self.params = constrain_parameters(self.params_uncostrained)

        self.c_, self.y_mean_ = fit(self.params, x, y, self.kernel)
        self.x_train = x
        self.y_train = y

        return self

    def predict(self, x, full_covariance=False):
        return predict(
            self.params,
            self.x_train,
            x,
            self.c_,
            self.y_mean_,
            self.kernel,
            full_covariance=full_covariance,
        )


# Alias
GPR = GaussianProcessRegression


# Export
__all__ = [
    "GaussianProcessRegression",
    "GPR",
]
