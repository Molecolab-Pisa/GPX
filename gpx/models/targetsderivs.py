from gpx.models.utils import _check_random_key
from gpx.optimizers.utils import ravel_backward_trainables, unravel_forward_trainables
from jax import value_and_grad, grad
from scipy.optimize import minimize
import warnings
import jax.numpy as jnp

def scipy_minimize(
    state,
    x,
    y,
    y_derivs,
    jacobian,
    loss_fn,
    callback = None,
):

    # x0: flattened trainables (1D) in unbound space
    # tdef: definition of trainables tree (non-trainables are None)
    # unravel_fn: callable to unflatten x0
    x0, tdef, unravel_fn = ravel_backward_trainables(state.params)

    # function to unravel and unflatten trainables and go in bound space
    unravel_forward = unravel_forward_trainables(unravel_fn, tdef, state.params)

    def loss(xt):
        # go in bound space and reconstruct params
        params = unravel_forward(xt)
        ustate = state.update(dict(params=params))
        return loss_fn(state=ustate, x=x, y=y, y_derivs=y_derivs, jacobian=jacobian)

    loss_and_grad = jit(value_and_grad(loss))
    jax.debug.print("{z}",z=jit(grad(loss))(x0))

    optres = minimize(
        loss_and_grad, x0=x0, method="L-BFGS-B", jac=True, callback=callback
    )

    params = unravel_forward(optres.x)
    state = state.update(dict(params=params))

    return state, optres

def randomized_minimization(
    key,
    state,
    x,
    y,
    y_derivs,
    jacobian,
    loss_fn,
    minimization_function = scipy_minimize,
    num_restarts = 0,
    return_history = False,
    opt_kwargs = None,
):
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs

    _check_random_key(key=key, num_restarts=num_restarts)

    states = []
    losses = []
    opt_info = []

    state, *optres = minimization_function(
        state=state, x=x, y=y, y_derivs=y_derivs, jacobian=jacobian, loss_fn=loss_fn, **opt_kwargs
    )
    loss = loss_fn(state=state, x=x, y=y, y_derivs=y_derivs, jacobian=jacobian)

    states.append(state)
    losses.append(loss)
    opt_info.append(optres)

    for _restart in range(num_restarts):
        subkey, key = jax.random.split(key)
        state = state.randomize(key)

        state, *optres = minimization_function(
            state=state,
            x=x,
            y=y,
            y_derivs=y_derivs,
            jacobian=jacobian,
            loss_fn=loss_fn,
            **opt_kwargs,
        )
        loss = loss_fn(
            state=state,
            x=x,
            y=y,
            y_derivs=y_derivs,
            jacobian=jacobian,
        )

        states.append(state)
        losses.append(loss)
        opt_info.append(optres)

    idx = losses.index(min(losses))
    state = states[idx]
    optres = opt_info[idx]

    if return_history:
        return state, *optres, states, losses
    else:
        return state, *optres

from gpx.parameters import ModelState
from gpx.models._gpr import _A_lhs, _A_derivs_lhs
from functools import partial
from jax import jit
import jax.scipy as jsp

class TargetsDerivs:
    
    def __init__(self,
    kernel,
    mean_function,
    kernel_params,
    sigma_targets,
    sigma_derivs1,
    sigma_derivs2=None,
    # sigma_derivs3=None,
    ):
        params = {"kernel_params": kernel_params, 
                  "sigma_targets": sigma_targets, 
                  "sigma_derivs1": sigma_derivs1,
                  "sigma_derivs2": sigma_derivs2,
                  # "sigma_derivs3": sigma_derivs3,
                 }
        opt = {
            "x_train": None,
            "jacobian1_train": None,
            "jacobian2_train": None,
            # "jacobian3_train": None,
            "jaccoef1": None,
            "jaccoef2": None,
            # "jaccoef3": None,
            "y_train": None,
            "y_derivs1_train": None,
            "y_derivs2_train": None,
            # "y_derivs3_train": None,
            "is_fitted": False,
            "is_fitted_derivs": False,
            "c": None,
            "c_targets": None,
            "mu": None,
        }

        self.state = ModelState(kernel, mean_function, params, **opt)
    
    @partial(jit,static_argnums=(0,6,7))
    def _lml(self,
             params,
             x,
             jacobian1,
             jacobian2,
             jacobian3,
             y,
             y_derivs1,
             y_derivs2,
             y_derivs3,
             kernel,
             mean_function
            ):
        m = y.shape[0]
        kernel_params = params["kernel_params"]
        sigma_targets = params["sigma_targets"].value
        sigma_derivs1 = params["sigma_derivs1"].value

        mu = mean_function(y)
        y = y - mu
        y = y.reshape(-1,1)
        y_derivs1 = y_derivs1.reshape(-1,1)
        y_m = jnp.concatenate((y,y_derivs1))
        
        if y_derivs2 is not None:
            sigma_derivs2 = params["sigma_derivs2"].value
            y_derivs2 = y_derivs2.reshape(-1,1)
            y_m = jnp.concatenate((y_m,y_derivs2))
        if y_derivs3 is not None:
            sigma_derivs3 = params["sigma_derivs3"].value
            y_derivs3 = y_derivs3.reshape(-1,1)
            y_m = jnp.concatenate((y_m,y_derivs3))
               
        # build kernel with target and derivatives
        K = kernel(
            x1=x,
            x2=x,
            params=kernel_params
        )
        K = K + sigma_targets**2 * jnp.eye(K.shape[0])


        D01kj_1 = kernel.d01kj(
            x1=x,
            jacobian1=jacobian1,
            x2=x,
            jacobian2=jacobian1,
            params=kernel_params,
        )
        D01kj_1 = D01kj_1 + sigma_derivs1**2 * jnp.eye(D01kj_1.shape[0])

        D0kj_1 = kernel.d0kj(
            x1=x,
            x2=x,
            params=kernel_params,
            jacobian=jacobian1,
        )

        if jacobian2 is None:
            C_mm = jnp.concatenate((jnp.concatenate((K,D0kj_1.T),axis=1),jnp.concatenate((D0kj_1,D01kj_1),axis=1)),axis=0)
            c = jnp.linalg.solve(C_mm,y_m)
            return c, mu
        else:

            D01kj_2 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian2,
                x2=x,
                jacobian2=jacobian2,
                params=kernel_params,
            )
            D01kj_2 = D01kj_2 + sigma_derivs2**2 * jnp.eye(D01kj_2.shape[0])

            D01kj_12 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian1,
                x2=x,
                jacobian2=jacobian2,
                params=kernel_params,
            )

            D0kj_2 = kernel.d0kj(
                x1=x,
                x2=x,
                params=kernel_params,
                jacobian=jacobian2,
            )

        if jacobian3 is None:
            C_mm = jnp.concatenate((jnp.concatenate((K,D0kj_1.T,D0kj_2.T),axis=1),
                        jnp.concatenate((D0kj_1,D01kj_1,D01kj_12),axis=1),
                        jnp.concatenate((D0kj_2,D01kj_12.T,D01kj_2),axis=1)),axis=0)
            c = jnp.linalg.solve(C_mm,y_m)
            return c, mu
        else:

            D01kj_3 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian3,
                x2=x,
                jacobian2=jacobian3,
                params=kernel_params,
            )
            D01kj_3 = D01kj_3 + sigma_derivs3**2 * jnp.eye(D01kj_3.shape[0])

            D0kj_3 = kernel.d0kj(
                x1=x,
                x2=x,
                params=kernel_params,
                jacobian=jacobian3,
            )

            D01kj_23 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian2,
                x2=x,
                jacobian2=jacobian3,
                params=kernel_params,
            )

            D01kj_13 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian1,
                x2=x,
                jacobian2=jacobian3,
                params=kernel_params,
            )

            C_mm = jnp.concatenate((jnp.concatenate((K,D0kj_1.T,D0kj_2.T,D0kj_3.T),axis=1),
                                    jnp.concatenate((D0kj_1,D01kj_1,D01kj_12,D01kj_13),axis=1),
                                    jnp.concatenate((D0kj_2,D01kj_12.T,D01kj_2,D01kj_23),axis=1),
                                    jnp.concatenate((D0kj_3,D01kj_13.T,D01kj_23.T,D01kj_3),axis=1)),axis=0)
        
        L_m = jsp.linalg.cholesky(C_mm,lower=True)
        cy = jsp.linalg.solve_triangular(L_m,y_m,lower=True)
        
        mll = -0.5 * jnp.sum(jnp.square(cy))
        mll -= jnp.sum(jnp.log(jnp.diag(L_m)))
        mll -= m * 0.5 * jnp.log(2.0 * jnp.pi)
        
        # normalize by the number of samples
        mll = mll / m
        
        return mll
    
    def log_marginal_likelihood(self,
                                state,
                                x,
                                jacobian,
                                y,
                                y_derivs,
                               ):
        return self._lml(
                    params=state.params,
                    x=x,
                    jacobian=jacobian,
                    y=y,
                    y_derivs=y_derivs,
                    kernel=state.kernel,
                    mean_function=state.mean_function
                   )
    
    def neg_log_marginal_likelihood(self,
                                    state,
                                    x,
                                    jacobian,
                                    y,
                                    y_derivs,
                                   ):
        return - self.log_marginal_likelihood(
                                    state=state,
                                    x=x,
                                    jacobian=jacobian,
                                    y=y,
                                    y_derivs=y_derivs,
                                    )
    
    @partial(jit,static_argnums=(0,10,11))
    def _fit(self,
        params,
        x,
        jacobian1,
        jacobian2,
        jacobian3,
        y,
        y_derivs1,
        y_derivs2,
        y_derivs3,
        kernel,
        mean_function,
    ):

        kernel_params = params["kernel_params"]
        sigma_targets = params["sigma_targets"].value
        sigma_derivs1 = params["sigma_derivs1"].value

        mu = mean_function(y)
        y = y - mu
        y = y.reshape(-1,1)
        y_derivs1 = y_derivs1.reshape(-1,1)
        y_m = jnp.concatenate((y,y_derivs1))
        
        if y_derivs2 is not None:
            sigma_derivs2 = params["sigma_derivs2"].value
            y_derivs2 = y_derivs2.reshape(-1,1)
            y_m = jnp.concatenate((y_m,y_derivs2))
        if y_derivs3 is not None:
            sigma_derivs3 = params["sigma_derivs3"].value
            y_derivs3 = y_derivs3.reshape(-1,1)
            y_m = jnp.concatenate((y_m,y_derivs3))
               
        # build kernel with target and derivatives
        K = kernel(
            x1=x,
            x2=x,
            params=kernel_params
        )
        K = K + sigma_targets**2 * jnp.eye(K.shape[0])


        D01kj_1 = kernel.d01kj(
            x1=x,
            jacobian1=jacobian1,
            x2=x,
            jacobian2=jacobian1,
            params=kernel_params,
        )
        D01kj_1 = D01kj_1 + sigma_derivs1**2 * jnp.eye(D01kj_1.shape[0])

        D0kj_1 = kernel.d0kj(
            x1=x,
            x2=x,
            params=kernel_params,
            jacobian=jacobian1,
        )

        if jacobian2 is None:
            C_mm = jnp.concatenate((jnp.concatenate((K,D0kj_1.T),axis=1),jnp.concatenate((D0kj_1,D01kj_1),axis=1)),axis=0)
            c = jnp.linalg.solve(C_mm,y_m)
            return c, mu
        else:

            D01kj_2 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian2,
                x2=x,
                jacobian2=jacobian2,
                params=kernel_params,
            )
            D01kj_2 = D01kj_2 + sigma_derivs2**2 * jnp.eye(D01kj_2.shape[0])

            D01kj_12 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian1,
                x2=x,
                jacobian2=jacobian2,
                params=kernel_params,
            )

            D0kj_2 = kernel.d0kj(
                x1=x,
                x2=x,
                params=kernel_params,
                jacobian=jacobian2,
            )

        if jacobian3 is None:
            C_mm = jnp.concatenate((jnp.concatenate((K,D0kj_1.T,D0kj_2.T),axis=1),
                        jnp.concatenate((D0kj_1,D01kj_1,D01kj_12),axis=1),
                        jnp.concatenate((D0kj_2,D01kj_12.T,D01kj_2),axis=1)),axis=0)
            c = jnp.linalg.solve(C_mm,y_m)
            return c, mu
        else:

            D01kj_3 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian3,
                x2=x,
                jacobian2=jacobian3,
                params=kernel_params,
            )
            D01kj_3 = D01kj_3 + sigma_derivs3**2 * jnp.eye(D01kj_3.shape[0])

            D0kj_3 = kernel.d0kj(
                x1=x,
                x2=x,
                params=kernel_params,
                jacobian=jacobian3,
            )

            D01kj_23 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian2,
                x2=x,
                jacobian2=jacobian3,
                params=kernel_params,
            )

            D01kj_13 = kernel.d01kj(
                x1=x,
                jacobian1=jacobian1,
                x2=x,
                jacobian2=jacobian3,
                params=kernel_params,
            )

            C_mm = jnp.concatenate((jnp.concatenate((K,D0kj_1.T,D0kj_2.T,D0kj_3.T),axis=1),
                                    jnp.concatenate((D0kj_1,D01kj_1,D01kj_12,D01kj_13),axis=1),
                                    jnp.concatenate((D0kj_2,D01kj_12.T,D01kj_2,D01kj_23),axis=1),
                                    jnp.concatenate((D0kj_3,D01kj_13.T,D01kj_23.T,D01kj_3),axis=1)),axis=0)
        
        
        c = jnp.linalg.solve(C_mm,y_m)
        return c, mu
            
    def fit(self,
        x,
        jacobian1,
        y,
        y_derivs1,
        jacobian2 = None,
        jacobian3 = None,
        y_derivs2 = None,
        y_derivs3 = None,
        key = None,
        num_restarts = 0,
        minimize = True,
        return_history = False
    ):
        
        if minimize:
            minimization_function = scipy_minimize
            self.state, optres, *history = randomized_minimization(
                key=key,
                state=self.state,
                x=x,
                y=y,
                y_derivs=y_derivs,
                jacobian=jacobian,
                loss_fn=self.neg_log_marginal_likelihood,
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

        c, mu = self._fit(
            params=self.state.params,
            x=x,
            jacobian1=jacobian1,
            jacobian2=jacobian2,
            jacobian3=jacobian3,
            y=y,
            y_derivs1=y_derivs1,
            y_derivs2=y_derivs2,
            y_derivs3=y_derivs3,
            kernel=self.state.kernel,
            mean_function=self.state.mean_function,
        )
        ns, _, nv1 = jacobian1.shape
        
        c_targets = c[:ns].reshape(-1)
        c_derivs1 = c[ns:ns+nv1*ns]
        jaccoef1 = jnp.einsum("sv,sfv->sf", c_derivs1.reshape(ns, nv1), jacobian1)
                                   
        if jacobian2 is not None:
            ns, _, nv2 = jacobian2.shape
            c_derivs2 = c[ns+nv1*ns:ns+nv1*ns+nv2*ns]
            jaccoef2 = jnp.einsum("sv,sfv->sf", c_derivs2.reshape(ns, nv2), jacobian2)
        else: 
            jaccoef2 = None
                                   
        if jacobian3 is not None:
            ns, _, nv3 = jacobian3.shape
            c_derivs3 = c[ns+nv1*ns+nv2*ns:ns+nv1*ns+nv2*ns+nv3*ns]
            jaccoef3 = jnp.einsum("sv,sfv->sf", c_derivs3.reshape(ns, nv3), jacobian3)
        else:
            jaccoef3 = None

        self.state = self.state.update(
        dict(x_train=x, 
             jacobian1_train=jacobian1, 
             jacobian2_train=jacobian2, 
             jacobian3_train=jacobian3, 
             jaccoef1=jaccoef1,
             jaccoef2=jaccoef2,
             jaccoef3=jaccoef3,
             y_train=y, 
             y_derivs1_train=y_derivs1, 
             y_derivs2_train=y_derivs2, 
             y_derivs3_train=y_derivs3, 
             c=c, 
             c_targets=c_targets, 
             mu=mu, 
             is_fitted=True))
        return self
    
    @partial(jit,static_argnums=(0,12))
    def _predict(self,
         params,
         x_train,
         jacobian1_train,
         jacobian2_train,
         jacobian3_train,
         x,
         jacobian1,
         jacobian2,
         jacobian3,
         c,
         mu,
         kernel
        ):
        
        kernel_params=params["kernel_params"]
        
        K = kernel(
            x1=x_train,
            x2=x,
            params=kernel_params
        )

        D01kj_1 = kernel.d01kj(
            x1=x_train,
            jacobian1=jacobian1_train,
            x2=x,
            jacobian2=jacobian1,
            params=kernel_params,
        )

        D0kj_1 = kernel.d0kj(
            x1=x_train,
            x2=x,
            params=kernel_params,
            jacobian=jacobian1_train,
        )

        D1kj_1 = kernel.d1kj(
            x1=x_train,
            x2=x,
            params=kernel_params,
            jacobian=jacobian1,
        )
                                   
        if jacobian2 is None:
            K_mn = jnp.concatenate((jnp.concatenate((K,D1kj_1),axis=1),jnp.concatenate((D0kj_1,D01kj_1),axis=1)),axis=0)
        else:

            D01kj_2 = kernel.d01kj(
                x1=x_train,
                jacobian1=jacobian2_train,
                x2=x,
                jacobian2=jacobian2,
                params=kernel_params,
            )

            D01kj_12 = kernel.d01kj(
                x1=x_train,
                jacobian1=jacobian1_train,
                x2=x,
                jacobian2=jacobian2,
                params=kernel_params,
            )
            D01kj_21 = kernel.d01kj(
                x1=x_train,
                jacobian1=jacobian2_train,
                x2=x,
                jacobian2=jacobian1,
                params=kernel_params,
            )

            D0kj_2 = kernel.d0kj(
                x1=x_train,
                x2=x,
                params=kernel_params,
                jacobian=jacobian2_train,
            )
            D1kj_2 = kernel.d1kj(
                x1=x_train,
                x2=x,
                params=kernel_params,
                jacobian=jacobian2,
            )

            if jacobian3 is None:
                K_mn = jnp.concatenate((jnp.concatenate((K,D1kj_1,D1kj_2),axis=1),
                                        jnp.concatenate((D0kj_1,D01kj_1,D01kj_12),axis=1),
                                        jnp.concatenate((D0kj_2,D01kj_21,D01kj_2),axis=1)),axis=0)
            else:

                D01kj_3 = kernel.d01kj(
                    x1=x_train,
                    jacobian1=jacobian3_train,
                    x2=x,
                    jacobian2=jacobian3,
                    params=kernel_params,
                )

                D0kj_3 = kernel.d0kj(
                    x1=x_train,
                    x2=x,
                    params=kernel_params,
                    jacobian=jacobian3_train,
                )
                D1kj_3 = kernel.d1kj(
                    x1=x_train,
                    x2=x,
                    params=kernel_params,
                    jacobian=jacobian3,
                )

                D01kj_23 = kernel.d01kj(
                    x1=x_train,
                    jacobian1=jacobian2_train,
                    x2=x,
                    jacobian2=jacobian3,
                    params=kernel_params,
                )
                D01kj_32 = kernel.d01kj(
                    x1=x_train,
                    jacobian1=jacobian3_train,
                    x2=x,
                    jacobian2=jacobian2,
                    params=kernel_params,
                )

                D01kj_13 = kernel.d01kj(
                    x1=x_train,
                    jacobian1=jacobian1_train,
                    x2=x,
                    jacobian2=jacobian3,
                    params=kernel_params,
                )
                D01kj_31 = kernel.d01kj(
                    x1=x_train,
                    jacobian1=jacobian3_train,
                    x2=x,
                    jacobian2=jacobian1,
                    params=kernel_params,
                )


                K_mn = jnp.concatenate((jnp.concatenate((K,D1kj_1,D1kj_2,D1kj_3),axis=1),
                                jnp.concatenate((D0kj_1,D01kj_1,D01kj_12,D01kj_13),axis=1),
                                jnp.concatenate((D0kj_2,D01kj_21,D01kj_2,D01kj_23),axis=1),
                                jnp.concatenate((D0kj_3,D01kj_31,D01kj_32,D01kj_3),axis=1)),axis=0)
                                   
        pred = jnp.dot(K_mn.T,c)
        
        ns, _, nv1 = jacobian1.shape
                                   
        pred = pred.at[:ns,:].add(mu)
        y_pred = pred[:ns]
        y_derivs1_pred = pred[ns:ns+nv1*ns].reshape(ns,-1)
        
        if jacobian2 is not None:
            _, _, nv2 = jacobian2.shape
            y_derivs2_pred = pred[ns+nv1*ns:ns+nv1*ns+nv2*ns].reshape(ns,-1)
        else:
            return y_pred, y_derivs1_pred
        
        if jacobian3 is not None:
            _, _, nv3 = jacobian3.shape
            y_derivs3_pred = pred[ns+nv1*ns+nv2*ns:ns+nv1*ns+nv2*ns+nv3*ns].reshape(ns,-1)
        else:
            return y_pred, y_derivs1_pred, y_derivs2_pred,
        
        return y_pred, y_derivs1_pred, y_derivs2_pred, y_derivs3_pred
    
    def predict(self,
        x,
        jacobian1,
        jacobian2 = None,
        jacobian3 = None,
        ):
        
        if not self.state.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Run `fit` to fit the model before prediction."
            )
        return self._predict(
        params=self.state.params,
        x_train=self.state.x_train,
        jacobian1_train=self.state.jacobian1_train,
        jacobian2_train=self.state.jacobian2_train,
        jacobian3_train=self.state.jacobian3_train,
        x=x,
        jacobian1=jacobian1,
        jacobian2=jacobian2,
        jacobian3=jacobian3,
        c=self.state.c,
        mu=self.state.mu,
        kernel=self.state.kernel
        )
        
    def print(self) -> None:
        return self.state.print_params()
    
    def save(self, state_file):
        return self.state.save(state_file)
    
    def load(self, state_file):
        self.state = self.state.load(state_file)
        return self
