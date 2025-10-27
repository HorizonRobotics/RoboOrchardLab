.. _overview_lab_arch:

Framework Architecture
------------------------

The **RoboOrchardLab** framework is engineered with modularity and clarity at its core, enabling researchers and developers to efficiently build, train, and evaluate embodied AI agents.
The architecture is designed to separate concerns, promote reusability, and facilitate easy extension.

Below is a high-level diagram illustrating the main components and their interactions:

.. image:: ../_static/overview/lab_architecture.png
   :alt: System Architecture Diagram
   :align: center
   :width: 100%

This diagram highlights the following key components and their roles:

Configuration System
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the foundational layer that drives the entire framework, powered by `Pydantic <https://docs.pydantic.dev/>`_. It handles the definition, validation, serialization, and deserialization of all experiment parameters and component configurations.
It provides settings for data processing, model architectures, optimizer and scheduler choices, and the behavior of the training engine itself.

Data Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Responsible for all aspects of data handling. This includes defining Dataset objects, specifying data transforms and augmentations, and configuring DataLoader and Sampler instances for efficient batching and iteration.

Models Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This module houses the neural network architectures for various tasks. These are broadly categorized into:

* **Perception**: Specialized models for understanding the environment, such as `BIP3D <https://horizonrobotics.github.io/robot_lab/bip3d/index.html>`__ for advanced 3D object detection from visual input.

* **Embodied AI Algorithms**: Higher-level policies and models for agent interaction, including `SEM <https://arxiv.org/abs/2505.16196>`__, `FineGrasp <https://horizonrobotics.github.io/robot_lab/finegrasp/index.html>`__, **Whole-body Control** (algorithms designed for complex robot motion planning and coordination, under development)

Pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The central orchestrator of the framework. The Pipelines takes the data, models, optimizer, and scheduler, and executes the defined pipelines for training and evaluation.
It manages the main training loop, incorporates various hooks (for functionalities like checkpointing, TensorBoard logging, statistics monitoring, etc.), and handles the overall execution flow.

Accelerator
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Leveraging Hugging Face Accelerate, this component provides seamless support for distributed training (e.g., DDP, FSDP) and mixed-precision training.
It works in conjunction with the Engine to abstract away the complexities of different hardware setups (single GPU, multi-GPU, multi-node) and optimize training performance.
