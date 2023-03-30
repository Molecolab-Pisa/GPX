<div align="center">
<img src="images/GPX_logo_250px.png" alt="logo"></img>
</div>

# GPX

GPX is Gaussian Process Regression, written in JAX.


## Installation

An environment with python 3.10 is recommended. You can create it with `conda`, `virtualenv`, or `pyenv`.

For example, with conda run the following code:

```shell
conda create -n gpx-env python=3.10
```
```shell
conda activate gpx-env
```

#### JAX installation with GPU (CUDA) support

You can skip this section if you only run JAX on the CPU.

In order to install JAX with GPU support you must have CUDA and CuDNN installed. 
CUDA 11 is required (available on gpumachine, molimen6, molimen7). 
You can install CuDNN with:

```shell
conda install -c conda-forge cudnn=8.4
```

Then run:

```shell
pip install --upgrade pip
```

```shell
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For more information refer to [JAX installation on GPU (CUDA)](https://github.com/google/jax#pip-installation-gpu-cuda)

#### GPX installation

When installing GPX all the dependencies are included, however in case you have specific requirements for the JAX installation,
you may want to install JAX separately, before installing GPX.
For instance, if you want to run on GPU, see the previous section. 

To clone the GPX module, run:

```shell
git clone git@molimen1.dcci.unipi.it:molecolab/gpx.git
```

This will create the `gpx` repository.

From the `gpx` folder, you can install the module with pip:

```shell
pip install .
```

#### Developer mode installation

It is recommended to fork this repository and clone the "forked" one as `origin`. The `upstream` version
can be added by running the following code:

```shell
git remote add upstream git@molimen1.dcci.unipi.it:molecolab/gpx.git
```

More details on [configuring a remote repository for a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork).

You can install it locally with:

```shell
pip install -e .
```

## Getting Started

You may want to look at the examples.


## Notes for Developers

We recommend making changes in a branch of your local version. 
Make sure that your main branch is up to date with the upstream:

```shell
git pull upstream main
```

If you feel your work is completed and want to merge it with the `main` branch of GPX, you can
make a merge request and ask for a review of your work.

If, when contributing with some feature, you want to write some unit test for it, we are all super
happy. We use `pytest` to write our tests.

To run all the tests you can use `tox` (`pip install tox` if you don't have it).

```shell
tox -e tests
```

It is recommended to run the tests before pushing your changes to the upstream repository.

We also use some pre-commit hooks, to format the code (with `black` and `isort`) and to immediately
catch plain errors and bad practices (with `flake8`). To install the `pre-commit` hooks, follow
the instructions in the `.pre-commit-config.yaml` file.

You can check for code quality with tox, typing:

```shell
tox -e lint
```

If you want to manually format the code using `black` and `isort`, you can also type:

```shell
tox -e format
```
