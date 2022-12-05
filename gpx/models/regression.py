from functools import partial
from collections import defaultdict, namedtuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit

from ..utils import softplus, split_params


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
