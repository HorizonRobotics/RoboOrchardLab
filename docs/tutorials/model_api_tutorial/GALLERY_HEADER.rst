.. _model_api_tutorial:

Model API Tutorials
====================================

This tutorial guides you through using the **RoboOrchardLab** ``ModelMixin`` API to build, save, and load your PyTorch models in a standardized and reproducible way.

The core idea behind ``ModelMixin`` is to **tightly couple a model's architecture (Configuration) with its learned weights (State Dictionary)**.
This practice ensures complete model reproducibility: given a saved model directory, you can restore the exact model instance with a single command.

Key Advantages:

* **Reproducibility**: Bundles configuration with weights, preventing mismatches or forgotten hyperparameters.
* **Easy Integration with huggingface**: Uses the ``safetensors`` format by default for secure and fast file handling and offers an out-of-the-box hook for frameworks like Hugging Face Accelerate.
