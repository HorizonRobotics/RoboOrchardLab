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
import importlib
from collections import OrderedDict
from pathlib import Path

import torch.nn as nn
from transformers import (
    BertConfig,
    BertModel as HFBertModel,
    BertTokenizerFast,
)

from robo_orchard_lab.models.bip3d.bert import BertModel as BIP3DBertModel
from robo_orchard_lab.models.bip3d.layers import MultiScaleDeformableAttention
from robo_orchard_lab.utils.path import in_cwd


class TestBIP3DTransformersCompatibility:
    def test_bip3d_core_modules_import(self):
        modules = [
            "robo_orchard_lab.models.bip3d.bert",
            "robo_orchard_lab.models.bip3d.layers",
            "robo_orchard_lab.models.bip3d.feature_enhancer",
            "robo_orchard_lab.models.bip3d.spatial_enhancer",
            "robo_orchard_lab.models.bip3d.structure",
        ]

        for module_name in modules:
            module = importlib.import_module(module_name)
            assert module is not None

    def test_multiscale_deformable_attention_instantiates(self):
        layer = MultiScaleDeformableAttention()

        assert isinstance(layer, MultiScaleDeformableAttention)


class TestBIP3DMetadataSave:
    def test_save_metadata_uses_default_hf_save_kwargs(self, tmp_path):
        calls = []

        class FakeTokenizer:
            def save_pretrained(self, directory):
                Path(directory, "tokenizer.json").write_text("{}")

        class FakeHFModel:
            def save_pretrained(self, directory, **kwargs):
                calls.append(kwargs)
                Path(directory, "model.safetensors").write_text("weights")

        class FakeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeHFModel()

        model = BIP3DBertModel.__new__(BIP3DBertModel)
        nn.Module.__init__(model)
        model.name = "bert"
        model.tokenizer = FakeTokenizer()
        model.language_backbone = nn.Sequential(
            OrderedDict([("body", FakeBackbone())])
        )

        model.save_metadata(tmp_path)

        assert calls == [{}]
        assert (tmp_path / "bert" / "tokenizer.json").exists()
        assert (tmp_path / "bert" / "model.safetensors").exists()

    def test_save_metadata_default_output_round_trips_from_pretrained(
        self,
        tmp_path,
    ):
        source_dir = tmp_path / "tiny_bert"
        source_dir.mkdir()
        (source_dir / "vocab.txt").write_text(
            "\n".join(
                [
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]",
                    "[MASK]",
                    "hello",
                    "world",
                ]
            )
        )
        tokenizer = BertTokenizerFast(vocab_file=str(source_dir / "vocab.txt"))
        tokenizer.save_pretrained(str(source_dir))
        hf_model = HFBertModel(
            BertConfig(
                vocab_size=7,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            )
        )
        hf_model.save_pretrained(str(source_dir))

        with in_cwd(str(tmp_path)):
            model = BIP3DBertModel(name="tiny_bert", max_tokens=8)
            export_dir = tmp_path / "export"
            model.save_metadata(str(export_dir))

        restored = BIP3DBertModel(name=str(export_dir / "tiny_bert"))
        output = restored(["hello world"])

        assert "embedded" in output
        assert output["embedded"].shape[-1] == 8

    def test_sub_sentence_attention_mask_round_trips_through_hf_bert(
        self,
        tmp_path,
    ):
        source_dir = tmp_path / "tiny_bert"
        source_dir.mkdir()
        (source_dir / "vocab.txt").write_text(
            "\n".join(
                [
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                    "[SEP]",
                    "[MASK]",
                    "hello",
                    "world",
                ]
            )
        )
        tokenizer = BertTokenizerFast(vocab_file=str(source_dir / "vocab.txt"))
        tokenizer.save_pretrained(str(source_dir))
        hf_model = HFBertModel(
            BertConfig(
                vocab_size=7,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            )
        )
        hf_model.save_pretrained(str(source_dir))

        model = BIP3DBertModel(
            name=str(source_dir),
            max_tokens=8,
            use_sub_sentence_represent=True,
            special_tokens_list=["[CLS]", "[SEP]"],
            add_pooling_layer=False,
        )
        output = model(["hello world"])

        assert "embedded" in output
        assert output["embedded"].shape[-1] == 8
        assert output["masks"].dim() == 3
