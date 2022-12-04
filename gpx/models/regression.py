from functools import partial
from collections import defaultdict, namedtuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit

from .utils import softplus, split_params



# def gpr_optimize(params, x, y, kernel, n_steps=100, step_size=0.01, verbose=20):
#    '''
#    Optimize a Gaussian Process Regression model with the Adam optimizer.
#    Arguments
#    ---------
#    params    : dict
#              Dictionary of parameters. Should have a 'kernel_params' keyword
#              to specify kernel parameters (a ictionary) and a 'sigma' keyword
#              to specify the noise.
#    x         : jnp.ndarray, (M, F)
#              Input matrix of M samples and F features
#    y         : jnp.ndarray, (M, 1)
#              Target matrix of M samples and 1 target
#    kernel    : callable
#              Kernel function
#    n_steps   : int
#              Number of optimization steps
#    step_size : float
#              Step size / learning rate
#    verbose   : int
#              Frequency for printing the value of the loss function
#              (log marginal likelihood)
#    Returns
#    -------
#    params   : dict
#             Optimized parameters
#    '''
#
#    opt_init, opt_update, get_params = jax_optim.adam(step_size=step_size)
#    opt_state = opt_init(params)
#    loss_fn = partial(gpr_log_marginal_likelihood, kernel=kernel, return_negative=True)
#
#    @jit
#    def train_step(step_i, opt_state, x, y):
#        params = get_params(opt_state)
#        grads = grad(loss_fn, argnums=0)(params, x, y)
#        return opt_update(step_i, grads, opt_state)
#
#    for step_i in range(n_steps):
#        opt_state = train_step(step_i, opt_state, x, y)
#        if step_i % verbose == 0:
#            params = get_params(opt_state)
#            loss = loss_fn(params, x, y)
#            print(" loss : {:.3f}".format(float(loss)))
#
#    params = get_params(opt_state)
#    return params


## =============================================================================
## Sparse Gaussian Process Regression
## =============================================================================
#
#
# def sgpr_log_marginal_likelihood(params, x, y, x_locs, kernel, return_negative=False):
#    '''
#    Optimize a Gaussian Process Regression model (Projected Processes) with the Adam optimizer.
#    Arguments
#    ---------
#    params          : dict
#                    Dictionary of parameters. Should have a 'kernel_params' keyword
#                    to specify kernel parameters (a dictionary) and a 'sigma' keyword
#                    to specify the noise. If the input locations should be optimized, they
#                    must be included in `params` dictionary under the keyword 'x_locs'.
#                    In this case, `x_locs` is ignored.
#    x               : jnp.ndarray, (M, F)
#                    Input matrix of M samples and F features
#    y               : jnp.ndarray, (M, 1)
#                    Target matrix of M samples and 1 target
#    x_locs          : jnp.ndarray, (N, F)
#                    Input locations (N <= M) with F features
#    kernel          : callable
#                    Kernel function
#    return_negative : bool
#                    Whether to return the negative marginal log likelihood
#    Returns
#    -------
#    mll       : jnp.ndarray, ()
#              Log marginal likelihood
#    '''
#
#    kernel_params, sigma = split_params(params)
#    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
#    sigma = softplus(sigma)
#    x_locs = params["x_locs"] if "x_locs" in params.keys() else x_locs
#
#    y_mean = jnp.mean(y)
#    y = y - y_mean
#    m = y.shape[0]
#
#    K_mm = kmap(kernel, x_locs, x_locs, kernel_params)
#    K_mn = kmap(kernel, x_locs, x, kernel_params)
#    L_m = jsp.linalg.cholesky(K_mm, lower=True)
#    G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
#    C_nn = jnp.dot(G_mn.T, G_mn) + sigma**2 * jnp.eye(m)
#    L_m = jsp.linalg.cholesky(C_nn, lower=True)
#    c_n = jnp.linalg.solve(C_nn, y)
#    mll = (
#        -0.5 * jnp.squeeze(jnp.dot(y.T, c_n))
#        - jnp.sum(jnp.log(jnp.diag(L_m)))
#        - m * 0.5 * jnp.log(2.0 * jnp.pi)
#    )
#
#    if return_negative:
#        return -mll
#
#    return mll
#
#
# def sgpr_fit(params, x, y, x_locs, kernel):
#    '''
#    Fits a Sparse Gaussian Process Regression model (Projected Processes).
#    Arguments
#    ---------
#    params          : dict
#                    Dictionary of parameters. Should have a 'kernel_params' keyword
#                    to specify kernel parameters (a dictionary) and a 'sigma' keyword
#                    to specify the noise. If the input locations should be optimized, they
#                    must be included in `params` dictionary under the keyword 'x_locs'.
#                    In this case, `x_locs` is ignored.
#    x               : jnp.ndarray, (M, F)
#                    Input matrix of M samples and F features
#    y               : jnp.ndarray, (M, 1)
#                    Target matrix of M samples and 1 target
#    x_locs          : jnp.ndarray, (N, F)
#                    Input locations (N <= M) with F features
#    kernel          : callable
#                    Kernel function
#    Returns
#    -------
#    c       : jnp.ndarray, (M, 1)
#            Dual coefficients
#    y_mean  : jnp.ndarray, ()
#            Target mean
#    '''
#
#    kernel_params, sigma = split_params(params)
#    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
#    sigma = softplus(sigma)
#    x_locs = params["x_locs"] if "x_locs" in params.keys() else x_locs
#
#    y_mean = jnp.mean(y)
#    y = y - y_mean
#
#    K_mn = kmap(kernel, x_locs, x, kernel_params)
#    C_mm = sigma**2 * kmap(kernel, x_locs, x_locs, kernel_params) + jnp.dot(
#        K_mn, K_mn.T
#    )
#    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y)).reshape(-1, 1)
#
#    return c, y_mean
#
#
# def sgpr_predict(
#    params, x_locs, x, c, y_mean, kernel, full_covariance=False, jitter=1e-6
# ):
#    '''
#    Predicts using a Sparse Gaussian Process Regression model (Projected Processes).
#    Arguments
#    ---------
#    params          : dict
#                    Dictionary of parameters. Should have a 'kernel_params' keyword
#                    to specify kernel parameters (a dictionary) and a 'sigma' keyword
#                    to specify the noise. If the input locations should be optimized, they
#                    must be included in `params` dictionary under the keyword 'x_locs'.
#                    In this case, `x_locs` is ignored.
#    x_locs          : jnp.ndarray, (N, F)
#                    Input locations (N <= M) with F features
#    x               : jnp.ndarray, (M, F)
#                    Input matrix of M samples and F features
#    c               : jnp.ndarray, (N, 1)
#                    Dual coefficients
#    y_mean          : jnp.ndarray, ()
#                    Target mean
#    kernel          : callable
#                    Kernel function
#    full_covariance : bool
#                    Whether to return the full posterior covariance
#    Returns
#    -------
#    mu              : jnp.ndarray, (M, 1)
#                    Posterior mean
#    C_nn            : jnp.ndarray, (M, M)
#                    Posterior covariance
#    '''
#
#    kernel_params, sigma = split_params(params)
#    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
#    sigma = softplus(sigma)
#    x_locs = params["x_locs"] if "x_locs" in params.keys() else x_locs
#
#    K_mn = kmap(kernel, x_locs, x, kernel_params)
#    mu = jnp.dot(c.T, K_mn).reshape(-1, 1) + y_mean
#
#    if full_covariance:
#        m = x_locs.shape[0]
#        K_mm = kmap(kernel, x_locs, x_locs, kernel_params)
#        L_m = jsp.linalg.cholesky(K_mm + jnp.eye(m) * jitter, lower=True)
#        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
#        L_m = jsp.linalg.cholesky(
#            (sigma**2 * K_mm + jnp.dot(K_mn, K_mn.T)) + jnp.eye(m) * jitter,
#            lower=True,
#        )
#        H_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
#        C_nn = (
#            kmap(kernel, x, x, kernel_params)
#            - jnp.dot(G_mn.T, G_mn)
#            + jnp.dot(H_mn.T, H_mn)
#        )
#        return mu, C_nn
#
#    return mu
#
#
# def sgpr_optimize(
#    params,
#    x,
#    y,
#    x_locs,
#    kernel,
#    n_steps=100,
#    step_size=0.01,
#    verbose=20,
# ):
#    '''
#    Optimize a Sparse Gaussian Process Regression model (Projected Processes).
#    Arguments
#    ---------
#    params          : dict
#                    Dictionary of parameters. Should have a 'kernel_params' keyword
#                    to specify kernel parameters (a dictionary) and a 'sigma' keyword
#                    to specify the noise. If the input locations should be optimized, they
#                    must be included in `params` dictionary under the keyword 'x_locs'.
#                    In this case, `x_locs` is ignored.
#    x               : jnp.ndarray, (M, F)
#                    Input matrix of M samples and F features
#    y               : jnp.ndarray, (N, 1)
#                    Target matrix with M samples and 1 target
#    x_locs          : jnp.ndarray, (N, F)
#                    Input locations (N <= M) with F features
#    kernel          : callable
#                    Kernel function
#    n_steps         : int
#                    Number of optimization steps
#    step_size       : float
#                    Step size / learning rate
#    verbose         : int
#                    Frequency for printing the loss (negative log marginal likelihood)
#    Returns
#    -------
#    params          : dict
#                    Optimized parameters
#    '''
#
#    opt_init, opt_update, get_params = jax_optim.adam(step_size=step_size)
#    opt_state = opt_init(params)
#    loss_fn = partial(sgpr_log_marginal_likelihood, kernel=kernel, return_negative=True)
#
#    @jit
#    def train_step(step_i, opt_state, x, y, x_locs):
#        params = get_params(opt_state)
#        grads = grad(loss_fn, argnums=0)(params, x, y, x_locs)
#        return opt_update(step_i, grads, opt_state)
#
#    for step_i in range(n_steps):
#        opt_state = train_step(step_i, opt_state, x, y, x_locs)
#        if step_i % verbose == 0:
#            params = get_params(opt_state)
#            loss = loss_fn(params, x, y, x_locs)
#            print(" loss : {:.3f}".format(float(loss)))
#
#    params = get_params(opt_state)
#
#    return params
#
#
## =============================================================================
## Sparse Gaussian Process Regression with Derivatives only
## =============================================================================
#
#
# def sgprd_fit(params, x, dy, x_locs, kernel):
#    '''
#    Fit a Sparse Gaussian Process Regression (Projected Processes) model
#    with target derivatives only.
#    '''
#
#    kernel_params, sigma = split_params(params)
#    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
#    sigma = softplus(sigma)
#    x_locs = params['x_locs'] if 'x_locs' in params.keys() else x_locs
#
#    dy_mean = jnp.mean(dy)
#    dy = dy - dy_mean
#
#    D, _ = dy.shape
#    M, F = x_locs.shape
#    N, _ = x.shape
#
#    LK_nm = kmap(grad(kernel, argnums=0), x, x_locs, kernel_params) # [N, M, F]
#    LK_nm = jnp.transpose(LK_nm, axes=(0, 2, 1)) # [N, F, M]
#    LK_nm = jnp.reshape(LK_nm, (N*F, M)) # [D, M]
#    C_mm = sigma**2 * kmap(kernel, x_locs, x_locs, kernel_params) + jnp.dot(LK_nm.T, LK_nm)
#    c = jnp.linalg.solve(C_mm, jnp.dot(LK_nm.T, dy)).reshape(-1, 1)
#
#    return c, dy_mean
#
#
# def sgprd_predict(params, x_locs, x, c, dy_mean, kernel, full_covariance=False, jitter=1e-6):
#    '''
#    Predict with a Sparse Gaussian Process Regression (Projected Processes) model
#    trained on derivative values.
#    '''
#
#    kernel_params, sigma = split_params(params)
#    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
#    sigma = softplus(sigma)
#    x_locs = params['x_locs'] if 'x_locs' in params.keys() else x_locs
#
#    M, F = x_locs.shape
#    N, _ = x.shape
#
#    LK_nm = kmap(grad(kernel, argnums=0), x, x_locs, kernel_params) # [N, M, F]
#    LK_nm = jnp.transpose(LK_nm, axes=(0, 2, 1)) # [N, F, M]
#    LK_nm = jnp.reshape(LK_nm, (N*F, M)) # [D, M]
#
#    mu = jnp.dot(LK_nm, c).reshape(-1, 1) + dy_mean
#
#    if full_covariance:
#        K_mm = kmap(kernel, x_locs, x_locs, kernel_params)
#        L_m = jsp.linalg.cholesky(K_mm + jnp.eye(M) * jitter, lower=True)
#        G_mn = jsp.linalg.solve_triangular(L_m, LK_nm.T, lower=True)
#        L_m = jsp.linalg.cholesky(
#            (sigma**2 * K_mm + jnp.dot(LK_nm.T, LK_nm)) + jnp.eye(M) * jitter,
#            lower=True,
#        )
#        H_mn = jsp.linalg.solve_triangular(L_m, LK_nm.T, lower=True)
#        C_nn = (
#            kmap(kernel, x, x, kernel_params)
#            - jnp.dot(G_mn.T, G_mn)
#            + jnp.dot(H_mn.T, H_mn)
#        )
#        return mu, C_nn
#
#    return mu
