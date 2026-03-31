# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "projects"
    / "holobrain"
    / "scripts"
    / "inference_server.py"
)


class _FakeSettingConfig:
    pass


class _FakeLogger:
    def info(self, *_args, **_kwargs):
        return None


class _FakeLoggerManager:
    def get_child(self, _name):
        return _FakeLogger()


class _FakeModel:
    def eval(self):
        return None


class _FakePipeline:
    load_kwargs = None

    def __init__(self):
        self.model = _FakeModel()

    @classmethod
    def load_pipeline(cls, **kwargs):
        cls.load_kwargs = kwargs
        return cls()


class _FakeMultiArmManipulationInput:
    pass


class _FakeFlaskApp:
    def route(self, *_args, **_kwargs):
        def _decorator(func):
            return func

        return _decorator


def _load_script_module():
    fake_cli_module = types.ModuleType("robo_orchard_core.utils.cli")
    fake_cli_module.SettingConfig = _FakeSettingConfig
    fake_cli_module.pydantic_from_argparse = (
        lambda _config, _parser: types.SimpleNamespace(
            model_dir="/tmp/model",
            inference_prefix="inference",
            port=2000,
            server_name="holobrain",
            num_joints=7,
            valid_action_step=64,
        )
    )

    fake_logging_module = types.ModuleType("robo_orchard_core.utils.logging")
    fake_logging_module.LoggerManager = _FakeLoggerManager

    fake_pipeline_module = types.ModuleType(
        "robo_orchard_lab.models.holobrain.pipeline"
    )
    fake_pipeline_module.HoloBrainInferencePipeline = _FakePipeline

    fake_processor_module = types.ModuleType(
        "robo_orchard_lab.models.holobrain.processor"
    )
    fake_processor_module.MultiArmManipulationInput = (
        _FakeMultiArmManipulationInput
    )

    fake_flask_module = types.ModuleType("flask")
    fake_flask_module.Flask = lambda *_args, **_kwargs: _FakeFlaskApp()
    fake_flask_module.Response = object
    fake_flask_module.jsonify = lambda payload: payload
    fake_flask_module.request = object()

    fake_gevent_module = types.ModuleType("gevent.pywsgi")
    fake_gevent_module.WSGIServer = object

    spec = importlib.util.spec_from_file_location(
        "holobrain_inference_server_script", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None

    patched_modules = {
        "robo_orchard_core.utils.cli": fake_cli_module,
        "robo_orchard_core.utils.logging": fake_logging_module,
        "robo_orchard_lab.models.holobrain.pipeline": fake_pipeline_module,
        "robo_orchard_lab.models.holobrain.processor": fake_processor_module,
        "flask": fake_flask_module,
        "gevent.pywsgi": fake_gevent_module,
    }
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, patched_modules):
        spec.loader.exec_module(module)
    return module


class InferenceServerScriptTest(unittest.TestCase):
    def test_loads_pipeline_with_explicit_default_prefix(self):
        _FakePipeline.load_kwargs = None

        _load_script_module()

        self.assertIsNotNone(_FakePipeline.load_kwargs)
        self.assertEqual(
            _FakePipeline.load_kwargs["inference_prefix"],
            "inference",
        )
        self.assertEqual(_FakePipeline.load_kwargs["directory"], "/tmp/model")


if __name__ == "__main__":
    unittest.main()
