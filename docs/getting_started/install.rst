.. _getting_started_install:

Installation
======================

Prerequisites
----------------
* Linux (Tested on Ubuntu 22.04)
* Python >= 3.10 (Tested on 3.10)
* torch >= 2.4.0

PyTorch
^^^^^^^^

This project requires PyTorch version 2.4.0 or higher. Since the specific PyTorch
build (CPU, CUDA 11.x, CUDA 12.x, etc.) depends on your system, **PyTorch must be
installed manually before installing this package.** Refer to the
`official PyTorch installation guide <https://pytorch.org/get-started/locally/>`_
to choose the right command for your environment. For example:

.. code-block:: shell

    # CUDA 12.1 (default)
    pip install torch torchvision

    # CUDA 11.8
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # CUDA 12.4
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # CPU only
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Package Installer
^^^^^^^^^^^^^^^^^^

Both **pip** and **uv** are supported. The ``Makefile`` uses ``pip`` by default.
To use ``uv`` instead, copy ``.env.example`` to ``.env`` (git-ignored) and
customize as needed:

.. code-block:: shell

    cp .env.example .env           # enables uv pip for all make targets
    make install-editable          # uses pip (default)
    make install-editable PIP="uv pip"  # or override directly


Install from pip
--------------------------------

The release version of this package is available on PyPI. You can install
it using pip:

.. code-block:: shell

    pip install robo_orchard_lab

Note that some features may require additional dependencies that are not
included in the base package. For example, to install the dependencies
for BIP3D algorithm, you can run:

.. code-block:: shell

    pip install robo_orchard_lab[bip3d]

If you want to install the development version, please install it from source
as described below.


Install from source
--------------------------------

For the latest development version, clone the repository and install from source.

.. code-block:: shell

    cd /path/to/robo_orchard_lab
    make version
    make install-editable

Or directly:

.. code-block:: shell

    pip install -e .                                                         # pip
    uv pip install --config-settings editable_mode=compat -e .  # uv


Dev Environment
--------------------------------

Install dev dependencies (lint, test tools) and pre-commit hooks:

.. code-block:: shell

    make dev-env


Optional Extras
--------------------------------

Install additional dependencies for specific projects. Replace ``pip`` with
``uv pip`` to use uv instead.

.. code-block:: shell

    pip install ".[bip3d]"         # BIP3D 3D perception
    pip install ".[holobrain_0]"   # HoloBrain VLA
    pip install ".[finegrasp]"     # FineGrasp grasp detection
    pip install ".[sem]"           # SEM policy
    pip install ".[aux_think]"     # Aux-Think VLN
    pip install ".[mcap_datasets]" # MCAP dataset support


Known Issues
--------------------------------

We have tried to make the installation process as easy as possible, but there are
some dependencies that may not be available on PyPI. You have to install them
manually.

For example, you have to install `pytorch3d <https://github.com/facebookresearch/pytorch3d>`_ manually because the
default pip package is not compatible with your PyTorch version.
