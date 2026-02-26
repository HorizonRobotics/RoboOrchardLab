.. _model_zoo_tutorials:

Model Zoo Tutorials
====================================

These tutorials demonstrate the two core APIs of the framework: the **Model Zoo API** and the **Inference API**. Our tutorials are split into two tracks:

1. **Core API Tutorials**: For developers. This track demonstrates how to use our two pillar APIs. You will learn to use the
   **Model Zoo API** (:py:class:`~robo_orchard_lab.models.torch_model.TorchModelMixin`) to standardize any model by decoupling its **configuration**
   from its **weights** (``.safetensors``). Then, you will master the **Inference API** (:py:class:`~robo_orchard_lab.inference.mixin.InferencePipelineMixin`)
   to encapsulate complex **pre-processing** and **post-processing** logic into a single, serializable, and portable pipeline.

2. **Model Zoo Applications**: For users. This track highlights the immediate power and accessibility of our framework. See how to
   use a single command (:py:meth:`~robo_orchard_lab.inference.mixin.InferencePipelineMixin.load`) to download, cache, and run pre-trained SOTA models
   directly from the **Hugging Face Hub** using the simple ``hf://`` protocol. All available models can be found in `huggingface face <https://huggingface.co/HorizonRobotics/models>`__.
