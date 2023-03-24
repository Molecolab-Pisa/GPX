
# GPX

GPX is Gaussian Process Regression, written in JAX.


## Installation

An environment with python 3.10 is recommended. You can create it with `conda`, `virtualenv`, or `pyenv`.

You can install the module with pip:

```python
pip install .
```

If you are a developer of the module, you can install it locally with:

```python
pip install -e .
```

This should install all the dependencies, JAX included.
However, in case you have specific requirements for the JAX installation (e.g., you want to run on GPU),
you may want to install JAX separately, before installing GPX.


## Getting Started

You may want to look at the examples.


## Notes for Developers

It is recommended to fork this repository, and make changes in a branch of your local version.
If you feel your work is completed and want to merge it with the `main` branch of GPX, you can
make a pull request and ask for a review of your work.

If, when contributing with some feature, you want to write some unit test for it, we are all super
happy. We use `pyenv` to write our tests.

To run all the tests you can use `tox` (`pip install tox` if you don't have it).

```python
tox -e tests
```

It is recommended to run the tests before pushing your changes to the upstream repository.

We also use some pre-commit hooks, to format the code (with `black`) and to immediately catch plain
errors and bad practices (with `flake8`). To install the `pre-commit` hooks, follow the instructions
in the `.pre-commit-config.yaml` file.
