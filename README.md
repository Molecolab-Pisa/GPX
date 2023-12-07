<div align="center">
<img src="images/GPX_logo.png" alt="logo"></img>
</div>

# GPX

GPX is [Gaussian Process Regression](https://gaussianprocess.org/gpml/chapters/RW.pdf), written in [JAX](https://github.com/google/jax).

GPX currently supports:

* [Standard GPR](https://gaussianprocess.org/gpml/chapters/RW.pdf#page=25)
* Sparse GPR (SGPR) in the [Projected Processes Approximation](https://gaussianprocess.org/gpml/chapters/RW.pdf#page=196)
* SGPR in the Projected Processes Approximation, with landmark selection using the [Randomly Pivoted Cholesky Decomposition](https://arxiv.org/abs/2207.06503)
* [Radial Basis Function Networks](https://en.wikipedia.org/wiki/Radial_basis_function_network)
* Training on target values or on [derivative values](https://gaussianprocess.org/gpml/chapters/RW.pdf#page=209) (using the Hessian kernel)
* Kernels with automatic support for gradient and Hessian
* Dense and sparse operations, the latter of which are important to [scale GP](https://proceedings.neurips.cc/paper_files/paper/2019/file/01ce84968c6969bdd5d51c5eeaa3946a-Paper.pdf) to large datasets.
* [Iterative estimation](https://epubs.siam.org/doi/pdf/10.1137/16M1104974) of the log marginal likelihood with stochastic trace estimation and Lanczos quadrature.
* Interface to [scipy](https://scipy.org/), [nlopt](https://nlopt.readthedocs.io/en/latest/), and [optax](https://github.com/google-deepmind/optax) optimizers


## Installation

An environment with python 3.10 is recommended. You can create it with `conda`, `virtualenv`, or `pyenv`.
Then simply clone the project and install it with `pip`.

For example, using conda:

```shell
conda create -n gpx-env python=3.10
conda activate gpx-env
git clone https://github.com/Molecolab-Pisa/GPX
cd GPX
pip install .
```

If you need to install JAX with GPU support, install JAX first following the [instructions provided by JAX](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu).


## Getting Started

You may want to look at our list of examples:

* [GPR](./examples/gpr.ipynb)
* [SGPR](./examples/sgpr.ipynb)
* [SGPR with RPCholesky](./examples/sgpr_with_rpcholesky.ipynb)
* [GPR with derivatives](./examples/gpr_with_derivatives_only.ipynb)
* [Simple Multioutput GP](./examples/multioutput.ipynb)
* [Interface to NLOpt](./examples/nlopt_optimizers.ipynb)
* [Kernelizers](./examples/kernelizers.ipynb) and [Kernel Operations](./examples/kernel_operations.ipynb)
* [Maximum a Posteriori estimate](./examples/map_estimate.ipynb)
* [Model Persistence in GPX](./examples/model_persistence.ipynb)
* [Kernel derivatives](./examples/derivatives_solak.ipynb)

# Citing GPX

In order to cite GPX you can use the following bibtex entry:

```
@software{gpx2023github,
  author = {Edoardo Cignoni and Amanda Arcidiacono and Patrizia Mazzeo and Lorenzo Cupellini and Benedetta Mennucci},
  title = {GPX: Gaussian Process Regression in JAX},
  url = {https://github.com/Molecolab-Pisa/GPX},
  version = {0.1.0},
  year = {2023},
}
```
