.. _trainer_tutorials:

Trainer Tutorials
==================

This tutorial will guide you through understanding and customizing the training pipeline in **RoboOrchardLab**,
using the provided `ResNet50 on ImageNet example <https://github.com/HorizonRobotics/robo_orchard_lab/tree/master/examples/resnet50_imagenet>`_
as our foundation. We'll start with the basics of running the script and gradually delve into customizing each component.

Here are the key highlights:

* Configuration-Driven (Pydantic + Argparse): Uses Pydantic (SettingConfig, DatasetConfig, TrainerConfig) for typed, validated, and hierarchical configurations. This is excellent for clarity, reducing errors, and IDE support.

* Hugging Face Accelerator Integration: Uses Accelerator for abstracting device management (CPU/GPU/TPU), distributed training (DDP, FSDP, etc.), and mixed precision.

* Modular Training Pipeline: A clear abstraction for the main training loop, encapsulating the core logic. A hook system allows injecting custom logic at various points in the training loop without modifying the pipeline core.
