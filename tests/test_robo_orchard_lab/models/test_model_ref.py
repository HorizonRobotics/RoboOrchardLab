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
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from robo_orchard_core.utils.config import ClassType_co
from transformers import (
    BertConfig,
    BertModel,
    PretrainedConfig,
    PreTrainedModel,
)

from robo_orchard_lab.models.mixin import ModelMixin, TorchModuleCfg
from robo_orchard_lab.models.model_ref import (
    HFPretrainedModelRef,
    TorchModelLoadConfig,
    TorchModelRef,
)
from robo_orchard_lab.utils.path import in_cwd


class RefModel(ModelMixin):
    def __init__(self, cfg: "RefModelCfg"):
        super().__init__(cfg)
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RefModelCfg(TorchModuleCfg[RefModel]):
    class_type: ClassType_co[RefModel] = RefModel
    tag: str = "default"


class RelativeAssetModel(ModelMixin):
    def __init__(self, cfg: "RelativeAssetModelCfg"):
        super().__init__(cfg)
        self.linear = nn.Linear(4, 2)
        self.asset_contents = Path(cfg.asset_path).read_text()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RelativeAssetModelCfg(TorchModuleCfg[RelativeAssetModel]):
    class_type: ClassType_co[RelativeAssetModel] = RelativeAssetModel
    asset_path: str = "artifact.txt"


class UnregisteredHFConfig(PretrainedConfig):
    model_type = "unregistered_ref_model"

    def __init__(self, hidden_size: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class UnregisteredHFModel(PreTrainedModel):
    config_class = UnregisteredHFConfig

    def __init__(self, config: UnregisteredHFConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def forward(self, x: torch.Tensor | None = None):
        return x


class PublicFromConfigHFModel(UnregisteredHFModel):
    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        **kwargs,
    ) -> "PublicFromConfigHFModel":
        if not isinstance(config, UnregisteredHFConfig):
            raise TypeError(
                "PublicFromConfigHFModel expects UnregisteredHFConfig."
            )
        if kwargs:
            raise TypeError("Unexpected kwargs for from_config.")
        return cls(config)


class TorchDtypeOnlyPrivateConfigHFModel(UnregisteredHFModel):
    from_config = None
    last_from_private_kwargs: dict[str, object] | None = None

    @classmethod
    def _from_config(
        cls,
        config: UnregisteredHFConfig,
        **kwargs,
    ) -> "TorchDtypeOnlyPrivateConfigHFModel":
        cls.last_from_private_kwargs = dict(kwargs)
        if "dtype" in kwargs:
            raise TypeError(
                "`dtype` should be normalized before `_from_config`."
            )
        torch_dtype = kwargs.pop("torch_dtype", None)
        if kwargs:
            raise TypeError("Unexpected kwargs for `_from_config`.")

        model = cls(config)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        return model


class TestTorchModelRef:
    def test_load_config_rejects_empty_directory(self):
        with pytest.raises(ValueError, match="directory"):
            TorchModelLoadConfig(directory="   ")

    def test_builds_model_from_cfg(self):
        ref = TorchModelRef[RefModel](cfg=RefModelCfg(tag="from_cfg"))

        model = ref.resolve()

        assert isinstance(model, RefModel)
        assert isinstance(model.cfg, RefModelCfg)
        assert model.cfg.tag == "from_cfg"

    def test_loads_model_from_saved_directory(self, tmp_path):
        model = RefModel(RefModelCfg(tag="saved"))
        save_dir = tmp_path / "saved_model"
        model.save_model(str(save_dir))

        ref = TorchModelRef[RefModel](
            load_from=TorchModelLoadConfig(directory=str(save_dir)),
            ensure_type=RefModel,
        )

        loaded = ref.resolve()

        assert isinstance(loaded, RefModel)
        assert isinstance(loaded.cfg, RefModelCfg)
        assert loaded.cfg.tag == "saved"
        for key, value in model.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key])

    def test_prefers_cfg_as_structure_source_of_truth(self, tmp_path):
        saved_model = RefModel(RefModelCfg(tag="saved"))
        with torch.no_grad():
            saved_model.linear.weight.fill_(1.25)
            saved_model.linear.bias.fill_(-0.5)
        save_dir = tmp_path / "saved_model"
        saved_model.save_model(str(save_dir))

        ref = TorchModelRef[RefModel](
            cfg=RefModelCfg(tag="override"),
            load_from=TorchModelLoadConfig(directory=str(save_dir)),
        )

        loaded = ref.resolve()

        assert isinstance(loaded, RefModel)
        assert isinstance(loaded.cfg, RefModelCfg)
        assert loaded.cfg.tag == "override"
        for key, value in saved_model.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key])

    def test_cfg_plus_load_from_normalizes_hf_directory(
        self, tmp_path, monkeypatch
    ):
        saved_model = RefModel(RefModelCfg(tag="saved"))
        save_dir = tmp_path / "saved_model"
        saved_model.save_model(str(save_dir))

        monkeypatch.setattr(
            "robo_orchard_lab.models.model_ref.resolve_hf_compatible_path",
            lambda _: str(save_dir),
        )

        ref = TorchModelRef[RefModel](
            cfg=RefModelCfg(tag="override"),
            load_from=TorchModelLoadConfig(directory="hf://org/repo"),
        )

        loaded = ref.resolve()

        assert isinstance(loaded, RefModel)
        assert isinstance(loaded.cfg, RefModelCfg)
        assert loaded.cfg.tag == "override"
        for key, value in saved_model.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[key])

    def test_cfg_plus_load_from_builds_from_checkpoint_directory(
        self, tmp_path
    ):
        save_dir = tmp_path / "saved_model_with_artifact"
        save_dir.mkdir()
        (save_dir / "artifact.txt").write_text("artifact-from-checkpoint")

        with in_cwd(str(save_dir)):
            saved_model = RelativeAssetModel(
                RelativeAssetModelCfg(asset_path="artifact.txt")
            )
            saved_model.save_model(str(save_dir), required_empty=False)

        ref = TorchModelRef[RelativeAssetModel](
            cfg=RelativeAssetModelCfg(asset_path="artifact.txt"),
            load_from=TorchModelLoadConfig(directory=str(save_dir)),
        )

        loaded = ref.resolve()

        assert isinstance(loaded, RelativeAssetModel)
        assert loaded.asset_contents == "artifact-from-checkpoint"

    def test_serializes_device_map_with_torch_device_values(self):
        load_from = TorchModelLoadConfig.model_validate(
            {
                "directory": "/tmp/model",
                "device_map": {"encoder": torch.device("cpu")},
            }
        )
        ref = TorchModelRef[RefModel](
            cfg=RefModelCfg(),
            load_from=load_from,
        )

        dumped = ref.model_dump_json()

        assert ref.load_from is not None
        assert ref.load_from.device_map == {"encoder": "cpu"}
        assert json.loads(dumped)["load_from"]["device_map"] == {
            "encoder": "cpu"
        }

    def test_cfg_plus_load_from_requires_existing_directory(self, tmp_path):
        missing_dir = tmp_path / "missing_model_dir"
        ref = TorchModelRef[RefModel](
            cfg=RefModelCfg(tag="from_cfg"),
            load_from=TorchModelLoadConfig(
                directory=str(missing_dir),
                load_weights=False,
            ),
        )

        with pytest.raises(FileNotFoundError, match="does not exist"):
            ref.resolve()


class TestHFPretrainedModelRef:
    @staticmethod
    def _create_tiny_bert(tmp_path):
        config = BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
        saved_model = BertModel(config)
        with torch.no_grad():
            next(saved_model.parameters()).fill_(3.0)
        save_dir = tmp_path / "tiny_bert"
        saved_model.save_pretrained(str(save_dir))
        return saved_model, save_dir

    @staticmethod
    def _create_unregistered_hf_model(tmp_path):
        config = UnregisteredHFConfig(hidden_size=4)
        saved_model = UnregisteredHFModel(config)
        with torch.no_grad():
            next(saved_model.parameters()).fill_(2.0)
        save_dir = tmp_path / "tiny_unregistered_hf_model"
        saved_model.save_pretrained(str(save_dir))
        return saved_model, save_dir

    def test_loads_hf_pretrained_model_weights(self, tmp_path):
        saved_model, save_dir = self._create_tiny_bert(tmp_path)

        ref = HFPretrainedModelRef[BertModel](
            class_type=BertModel,
            path=str(save_dir),
        )

        loaded = ref.resolve()

        assert isinstance(loaded, BertModel)
        assert torch.allclose(
            next(loaded.parameters()),
            torch.full_like(next(loaded.parameters()), 3.0),
        )

    def test_hf_model_ref_rejects_empty_path(self):
        with pytest.raises(ValueError, match="path"):
            HFPretrainedModelRef(class_type=BertModel, path="   ")

    def test_builds_hf_model_from_config_without_weights(self, tmp_path):
        saved_model, save_dir = self._create_tiny_bert(tmp_path)

        ref = HFPretrainedModelRef[BertModel](
            class_type=BertModel,
            path=str(save_dir),
            load_weights=False,
        )

        built = ref.resolve()

        assert isinstance(built, BertModel)
        assert built.config.vocab_size == saved_model.config.vocab_size
        assert built.config.hidden_size == saved_model.config.hidden_size
        assert (
            built.config.num_hidden_layers
            == saved_model.config.num_hidden_layers
        )
        assert (
            built.config.num_attention_heads
            == saved_model.config.num_attention_heads
        )
        assert (
            built.config.intermediate_size
            == saved_model.config.intermediate_size
        )
        assert not torch.allclose(
            next(built.parameters()),
            next(saved_model.parameters()),
        )

    def test_hf_model_ref_supports_hf_uri(self, tmp_path, monkeypatch):
        saved_model, save_dir = self._create_tiny_bert(tmp_path)

        monkeypatch.setattr(
            "robo_orchard_lab.models.model_ref.resolve_hf_compatible_path",
            lambda _: str(save_dir),
        )

        ref = HFPretrainedModelRef[BertModel](
            class_type=BertModel,
            path="hf://org/repo",
        )

        loaded = ref.resolve()

        assert isinstance(loaded, BertModel)
        assert torch.allclose(
            next(loaded.parameters()),
            next(saved_model.parameters()),
        )

    def test_hf_model_ref_resolve_path_preserves_missing_local_like_string(
        self,
    ):
        ref = HFPretrainedModelRef[BertModel](
            class_type=BertModel,
            path="./missing_local_model_dir",
        )

        assert ref.resolve_path() == "./missing_local_model_dir"

    def test_hf_model_ref_applies_config_kwargs_when_loading_weights(
        self, tmp_path
    ):
        _, save_dir = self._create_tiny_bert(tmp_path)

        ref = HFPretrainedModelRef[BertModel](
            class_type=BertModel,
            path=str(save_dir),
            config_kwargs={"output_hidden_states": True},
        )

        loaded = ref.resolve()

        assert isinstance(loaded, BertModel)
        assert loaded.config.output_hidden_states is True

    def test_hf_model_ref_serializes_dtype_kwargs(self, tmp_path):
        _, save_dir = self._create_tiny_bert(tmp_path)
        ref = HFPretrainedModelRef(
            class_type=BertModel,
            path=str(save_dir),
            load_weights=False,
            build_kwargs={"dtype": torch.float16},
        )

        dumped = ref.model_dump_json()
        restored = HFPretrainedModelRef.model_validate_json(dumped)
        built = restored.resolve()

        assert restored.build_kwargs["dtype"] == "float16"
        assert isinstance(built, BertModel)
        assert next(built.parameters()).dtype == torch.float16

    def test_hf_model_ref_routes_legacy_model_kwargs_to_active_branch(
        self, tmp_path
    ):
        _, save_dir = self._create_tiny_bert(tmp_path)
        ref = HFPretrainedModelRef.model_validate(
            {
                "class_type": BertModel,
                "path": str(save_dir),
                "load_weights": False,
                "model_kwargs": {"dtype": torch.float16},
            }
        )

        built = ref.resolve()

        assert ref.build_kwargs["dtype"] == "float16"
        assert ref.load_kwargs == {}
        assert isinstance(built, BertModel)
        assert next(built.parameters()).dtype == torch.float16

    def test_hf_model_ref_routes_legacy_model_kwargs_with_string_bool(
        self, tmp_path
    ):
        _, save_dir = self._create_tiny_bert(tmp_path)
        ref = HFPretrainedModelRef.model_validate(
            {
                "class_type": BertModel,
                "path": str(save_dir),
                "load_weights": "false",
                "model_kwargs": {"dtype": torch.float16},
            }
        )

        built = ref.resolve()

        assert ref.build_kwargs["dtype"] == "float16"
        assert ref.load_kwargs == {}
        assert isinstance(built, BertModel)
        assert next(built.parameters()).dtype == torch.float16

    def test_hf_model_ref_translates_dtype_alias_for_private_from_config(
        self, tmp_path, monkeypatch
    ):
        _, save_dir = self._create_unregistered_hf_model(tmp_path)
        monkeypatch.setattr(
            "robo_orchard_lab.models.model_ref._runtime_hf_dtype_kwarg_name",
            lambda: "torch_dtype",
        )
        TorchDtypeOnlyPrivateConfigHFModel.last_from_private_kwargs = None

        ref = HFPretrainedModelRef[TorchDtypeOnlyPrivateConfigHFModel](
            class_type=TorchDtypeOnlyPrivateConfigHFModel,
            path=str(save_dir),
            load_weights=False,
            build_kwargs={"dtype": torch.float16},
        )

        built = ref.resolve()

        assert isinstance(built, TorchDtypeOnlyPrivateConfigHFModel)
        assert TorchDtypeOnlyPrivateConfigHFModel.last_from_private_kwargs == {
            "torch_dtype": "float16"
        }
        assert next(built.parameters()).dtype == torch.float16

    def test_hf_model_ref_rejects_conflicting_dtype_aliases(self, tmp_path):
        _, save_dir = self._create_tiny_bert(tmp_path)

        with pytest.raises(ValueError, match="must match"):
            HFPretrainedModelRef(
                class_type=BertModel,
                path=str(save_dir),
                load_weights=False,
                build_kwargs={
                    "dtype": torch.float16,
                    "torch_dtype": torch.float32,
                },
            )

    def test_hf_model_ref_rejects_load_kwargs_on_build_only_path(
        self, tmp_path
    ):
        _, save_dir = self._create_tiny_bert(tmp_path)

        with pytest.raises(ValueError, match="load_kwargs"):
            HFPretrainedModelRef(
                class_type=BertModel,
                path=str(save_dir),
                load_weights=False,
                load_kwargs={"device_map": "auto"},
            )

    def test_hf_model_ref_rejects_build_kwargs_on_load_path(self, tmp_path):
        _, save_dir = self._create_tiny_bert(tmp_path)

        with pytest.raises(ValueError, match="build_kwargs"):
            HFPretrainedModelRef(
                class_type=BertModel,
                path=str(save_dir),
                build_kwargs={"dtype": torch.float16},
            )

    def test_hf_model_ref_loads_custom_model_with_config_overrides(
        self, tmp_path
    ):
        _, save_dir = self._create_unregistered_hf_model(tmp_path)

        ref = HFPretrainedModelRef[UnregisteredHFModel](
            class_type=UnregisteredHFModel,
            path=str(save_dir),
            config_kwargs={"output_hidden_states": True},
        )

        loaded = ref.resolve()

        assert isinstance(loaded, UnregisteredHFModel)
        assert loaded.config.output_hidden_states is True
        assert torch.allclose(
            next(loaded.parameters()),
            torch.full_like(next(loaded.parameters()), 2.0),
        )

    def test_hf_model_ref_builds_custom_model_without_autoconfig(
        self, tmp_path
    ):
        saved_model, save_dir = self._create_unregistered_hf_model(tmp_path)

        ref = HFPretrainedModelRef[UnregisteredHFModel](
            class_type=UnregisteredHFModel,
            path=str(save_dir),
            load_weights=False,
        )

        built = ref.resolve()

        assert isinstance(built, UnregisteredHFModel)
        assert built.config.hidden_size == saved_model.config.hidden_size
        assert not torch.allclose(
            next(built.parameters()),
            next(saved_model.parameters()),
        )

    def test_hf_model_ref_uses_public_from_config_when_available(
        self, tmp_path
    ):
        _, save_dir = self._create_unregistered_hf_model(tmp_path)
        ref = HFPretrainedModelRef[PublicFromConfigHFModel](
            class_type=PublicFromConfigHFModel,
            path=str(save_dir),
            load_weights=False,
        )

        built = ref.resolve()

        assert isinstance(built, PublicFromConfigHFModel)
        assert built.config.hidden_size == 4
