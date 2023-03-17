# from __future__ import annotations
# from typing import Callable, Tuple, Union
#
# from functools import partial
# import jax.numpy as jnp
# from jax import jvp, jacrev, jacfwd
# from .kernelizers import kernelize
#
#
# # =============================================================================
# # Derivative kernel functions
# # =============================================================================
# # WARNING: these functions do not work in more than one dimension.
# #          in that case use the kernelize decorators above.
#
#
# def d0_k(k: Callable) -> Tuple[Callable, Callable]:
#     docstr = """
#     Derivative kernel with respect to the first argument.
#
#         d/d0(k) = cov(w, y) with y = f(x) and w = d/dx(f)(x).
#
#     Returns:
#     kernel: original kernel
#     d0_kernel: derivative kernel with respect to the first argument
#     """
#
#     def wrapper(x1, x2, params):
#         return jvp(partial(k, x2=x2, params=params), (x1,), (jnp.ones(x1.shape),))
#
#     wrapper.__doc__ = docstr + k.__doc__ if k.__doc__ is not None else docstr
#
#     return wrapper
#
#
# def d1_k(k: Callable) -> Tuple[Callable, Callable]:
#     docstr = """
#     Derivative kernel with respect to the seoncd argument.
#
#         d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x).
#
#     Returns:
#     kernel: original kernel
#     d1_kernel: derivative kernel with respect to the second argument
#     """
#
#     def wrapper(x1, x2, params):
#         return jvp(partial(k, x1, params=params), (x2,), (jnp.ones(x2.shape),))
#
#     wrapper.__doc__ = docstr + k.__doc__ if k.__doc__ is not None else docstr
#
#     return wrapper
#
#
# def d0d1_k(k: Callable) -> Tuple[Callable, Callable, Callable, Callable]:
#     docstr = """
#     Derivative kernel with respect to the second argument.
#
#         d^2/d0d1(k) = cov(w, w) with y = f(x) and w = d/dx(f)(x).
#
#     Returns:
#     kernel: original kernel
#     d0_kernel: derivative kernel with respect to the first argument
#     d1_kernel: derivative kernel with respect to the second argument
#     d0d1_kernel: derivative kernel with respect to the first and second argument
#     """
#
#     def wrapper(x1, x2, params):
#         return jvp(partial(d0_k(k), x1, params=params), (x2,), (jnp.ones(x2.shape),))
#
#     wrapper.__doc__ = docstr + k.__doc__ if k.__doc__ is not None else docstr
#
#     return wrapper
#
#
# def grad_kernel(k: Callable, argnums: Union[int, Tuple[int, int]]):
#     """
#     Generate a derivative kernel function.
#
#     argnums == 0:      d/d0(k) = cov(w, y) with y = f(x) and w = d/dx(f)(x)
#     argnums == 1:      d/d1(k) = cov(y, w) with y = f(x) and w = d/dx(f)(x)
#     argnums == (0, 1): d^2/d0d1(k) = cov(w, w) with y = f(x) and w = d/dx(f)(x)
#
#     Returns:
#     func with arguments (x1, x2, params)
#     """
#     if argnums == 0:
#         return d0_k(k)
#     elif argnums == 1:
#         return d1_k(k)
#     elif argnums == (0, 1):
#         return d0d1_k(k)
#     else:
#         raise ValueError(
#             f"argnums={argnums} is not valid. Allowed argnums: 0, 1, (0, 1)"
#         )
