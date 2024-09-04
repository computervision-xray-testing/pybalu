![Build and upload to PyPI](hhttps://github.com/computervision-xray-testing/pybalu/workflows/Build%20and%20upload%20to%20PyPI/badge.svg)


# About the Project

This project is a Python3 implementation for Balu, a computer vision, pattern recognition, and image processing library. Initially implemented in Matlab&reg; by Domingo Mery.


# Installation

Python 3.10 or higher is required to use this package. Also, it requires to have installed compilation tools:

- Windows: Microsoft C++ build tools (vsbuildtools). You can install it directly from Microsoft [website](https://aka.ms/vs/17/release/vs_buildtools.exe) or using [Chocolately](https://chocolatey.org/).
- Linux/MacOS: GCC.


In order to install pybalu, run

```bash
$ pip install pybalu
```

If installation fails, check if the environment has installed `setuptools` and `Cython`. In this case, install them:

```bash
$ python -m pip install --upgrade setuptools Cyhton
```


# Contributing

We follow [github flow](https://www.atlassian.com/es/git/tutorials/comparing-workflows/gitflow-workflow) standard. For contributions:

- Fork the repo
- Create a new branch called `feature/<feature-desc>` or `fix/<fix-desc>` depending on the nature of your contribution
- Perform a pull request and wait for maintainers accept or reject the contribution

Possible and valuable contributions:

- Tests
- More feature extraction, analysis, and transformation functions
- Fixes
- Documentation
- Examples

The project has switched to [Poetry](https://python-poetry.org/) as a packaging and dependency management framework. Please install Poetry before starting work on the contribution. The easiest way to install it is using `pipx`, but check the Poetry documentation to find the more convenient way for your configuration.

After you have cloned the repository, install it by:

```bash
$ poetry install
```

The installation will include all the dependencies specified in the pyproject.toml

At the end of the contribution, please verify the format running [Ruff](https://docs.astral.sh/ruff/):

```bash
$ poetry run ruff format
````

## Roadmap

- Documentation: _TODO_
