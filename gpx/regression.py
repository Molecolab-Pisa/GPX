from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit
from jax.example_libraries import optimizers as jax_optim

from .kernels import kmap
from .utils import softplus, split_params


# =============================================================================
# Standard Gaussian Process Regression
# =============================================================================


def gpr_log_marginal_likelihood(y, c, C_mm):
    m = y.shape[0]
    L_m = jsp.linalg.cholesky(C_mm, lower=True)
    mll = (
        -0.5 * jnp.squeeze(jnp.dot(y.T, c))
        - jnp.sum(jnp.log(jnp.diag(L_m)))
        - m * 0.5 * jnp.log(2.0 * jnp.pi)
    )
    return mll


def gpr_fit(params, x, y, kernel, return_negative_mll=False):
    kernel_params, sigma = split_params(params)
    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
    sigma = softplus(sigma)
    y_mean = jnp.mean(y)
    y = y - y_mean
    C_mm = kmap(kernel, x, x, kernel_params) + sigma**2 * jnp.eye(x.shape[0])
    c = jnp.linalg.solve(C_mm, y).reshape(-1, 1)
    if return_negative_mll:
        mll = gpr_log_marginal_likelihood(y, c, C_mm)
        return -mll
    return c, y_mean


def gpr_predict(
    params,
    x_train,
    x,
    c,
    y_mean,
    kernel,
    full_covariance=False,
):
    kernel_params, sigma = split_params(params)
    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
    sigma = softplus(sigma)
    K_mn = kmap(kernel, x_train, x, kernel_params)
    mu = jnp.dot(c.T, K_mn).reshape(-1, 1) + y_mean
    if full_covariance:
        C_mm = kmap(kernel, x_train, x_train, kernel_params) + sigma**2 * jnp.eye(
            x_train.shape[0]
        )
        L_m = jsp.linalg.cholesky(C_mm, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = kmap(kernel, x, x, kernel_params) - jnp.dot(G_mn.T, G_mn)
        return mu, C_nn
    return mu


def gpr_optimize(params, x, y, kernel, n_steps=100, step_size=0.01, verbose=20):

    opt_init, opt_update, get_params = jax_optim.adam(step_size=step_size)
    opt_state = opt_init(params)
    loss_fn = partial(gpr_fit, kernel=kernel, return_negative_mll=True)

    @jit
    def train_step(step_i, opt_state, x, y):
        params = get_params(opt_state)
        grads = grad(loss_fn, argnums=0)(params, x, y)
        return opt_update(step_i, grads, opt_state)

    for step_i in range(n_steps):
        opt_state = train_step(step_i, opt_state, x, y)
        if step_i % verbose == 0:
            params = get_params(opt_state)
            loss = loss_fn(params, x, y)
            print(' loss : {:.3f}'.format(float(loss)))

    params = get_params(opt_state)
    return params


#def gpr_optimize(params, x, y, kernel, n_steps=1000, lr=0.1):
#    params_flat, params_tree = jax.tree_flatten(params)
#    momentums = [p * 0.0 for p in params_flat]
#    scales = [p * 0.0 + 1 for p in params_flat]
#    loss_fn = partial(gpr_fit, kernel=kernel, return_negative_mll=True)
#    grad_loss_fn = jit(grad(loss_fn))
#    for i in range(n_steps):
#        grads = grad_loss_fn(params, x, y)
#        grads, _ = jax.tree_flatten(grads)
#        for k in range(len(params_flat)):
#            momentums[k] = 0.9 * momentums[k] + 0.1 * grads[k]
#            scales[k] = 0.9 * scales[k] + 0.1 * grads[k] ** 2
#            params_flat[k] -= lr * momentums[k] / jnp.sqrt(scales[k] + 1e-5)
#        params = jax.tree_unflatten(params_tree, params_flat)
#        if i % 50 == 0:
#            print("MLL =", loss_fn(params, x, y))
#    return params


# =============================================================================
# Sparse Gaussian Process Regression
# =============================================================================


def sgpr_fit(params, x, y, x_locs, kernel, return_negative_mll=False):
    kernel_params, sigma = split_params(params)
    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
    sigma = softplus(sigma)
    y_mean = jnp.mean(y)
    y = y - y_mean
    K_mn = kmap(kernel, x_locs, x, kernel_params)
    C_mm = sigma**2 * kmap(kernel, x_locs, x_locs, kernel_params) + jnp.dot(
        K_mn, K_mn.T
    )
    c = jnp.linalg.solve(C_mm, jnp.dot(K_mn, y)).reshape(-1, 1)
    if return_negative_mll:
        raise NotImplementedError("Marginal log likelihood currently not implemented.")
    return c, y_mean


def sgpr_predict(
    params, x_locs, x, c, y_mean, kernel, full_covariance=False, jitter=1e-6
):
    kernel_params, sigma = split_params(params)
    kernel_params = {p: softplus(v) for p, v in kernel_params.items()}
    sigma = softplus(sigma)
    K_mn = kmap(kernel, x_locs, x, kernel_params)
    mu = jnp.dot(c.T, K_mn).reshape(-1, 1) + y_mean
    if full_covariance:
        m = x_locs.shape[0]
        K_mm = kmap(kernel, x_locs, x_locs, kernel_params)
        L_m = jsp.linalg.cholesky(K_mm + jnp.eye(m) * jitter, lower=True)
        G_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        L_m = jsp.linalg.cholesky(
            (sigma**2 * K_mm + jnp.dot(K_mn, K_mn.T)) + jnp.eye(m) * jitter,
            lower=True,
        )
        H_mn = jsp.linalg.solve_triangular(L_m, K_mn, lower=True)
        C_nn = (
            kmap(kernel, x, x, kernel_params)
            - jnp.dot(G_mn.T, G_mn)
            + jnp.dot(H_mn.T, H_mn)
        )
        return mu, C_nn
    return mu
