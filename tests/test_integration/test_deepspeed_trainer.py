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

from __future__ import annotations
import json
import os
import subprocess
import sys
import textwrap

import pytest

_DEEPSPEED_MASTER_PORT_BASE = 29600
_DEEPSPEED_MASTER_PORT_WORKER_STRIDE = 1000
_DEEPSPEED_MASTER_PORT_RETRY_STRIDE = 100
_DEEPSPEED_MASTER_PORT_ATTEMPTS = 3
_DISTRIBUTED_PORT_CONFLICT_MARKERS = (
    "eaddrinuse",
    "address already in use",
)


@pytest.mark.parametrize(
    ("zero_optimization", "expected_basic_optimizer", "port_offset"),
    [
        ({"stage": 0}, "FusedAdam", 0),
        (
            {
                "stage": 3,
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
            },
            "DeepSpeedCPUAdam",
            10,
        ),
    ],
    ids=["no-offload", "cpu-offload"],
)
def test_hook_based_trainer_deepspeed_observer_smoke(
    PROJECT_ROOT: str,
    zero_optimization: dict[str, object],
    expected_basic_optimizer: str,
    port_offset: int,
) -> None:
    """Smoke test real Accelerate + DeepSpeed optimizer-step ownership."""
    pytest.importorskip(
        "deepspeed",
        reason="DeepSpeed integration smoke requires optional deepspeed.",
    )

    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    for retry_index in range(_DEEPSPEED_MASTER_PORT_ATTEMPTS):
        subprocess_env = {
            **_single_process_distributed_env(
                port_offset=port_offset,
                retry_index=retry_index,
            ),
            "TEST_DEEPSPEED_ZERO_CONFIG": json.dumps(zero_optimization),
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    r"""
                    import json
                    import os
                    import torch
                    from accelerate import Accelerator
                    from accelerate.utils import DeepSpeedPlugin
                    from torch.utils.data import DataLoader

                    from robo_orchard_lab.pipeline.training import (
                        HookBasedTrainer,
                    )
                    from robo_orchard_lab.pipeline.hooks.grad_clip import (
                        GradientClippingHookConfig,
                    )
                    from robo_orchard_lab.pipeline.hooks.stats import (
                        StatsMonitorConfig,
                    )
                    from robo_orchard_lab.processing.step_processor import (
                        SimpleStepProcessor,
                    )


                    class TinyModel(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.linear = torch.nn.Linear(2, 1)
                            self.extra = torch.nn.ParameterList(
                                [
                                    torch.nn.Parameter(torch.ones(1))
                                    for _ in range(3)
                                ]
                            )

                        def forward(self, batch):
                            extra = sum(self.extra) * 0.01
                            return self.linear(batch) + extra


                    class TinyBatchProcessor(SimpleStepProcessor):
                        def forward(self, model, batch):
                            outputs = model(batch)
                            loss = torch.mean((outputs - 1.0) ** 2)
                            return outputs, loss


                    def build_scheduler(optimizer):
                        return torch.optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=1,
                            gamma=0.1,
                        )


                    torch.manual_seed(0)
                    model = TinyModel()
                    dataloader = DataLoader(
                        torch.tensor(
                            [
                                [0.1, 0.2],
                                [0.2, 0.3],
                                [0.3, 0.4],
                                [0.4, 0.5],
                            ],
                            dtype=torch.float32,
                        ),
                        batch_size=1,
                    )
                    base_lr = 0.1
                    deepspeed_plugin = DeepSpeedPlugin(
                        hf_ds_config={
                            "train_micro_batch_size_per_gpu": "auto",
                            "train_batch_size": "auto",
                            "gradient_clipping": "auto",
                            "gradient_accumulation_steps": 2,
                            "zero_optimization": json.loads(
                                os.environ["TEST_DEEPSPEED_ZERO_CONFIG"]
                            ),
                        },
                        gradient_accumulation_steps=2,
                    )
                    accelerator = Accelerator(
                        deepspeed_plugin=deepspeed_plugin,
                        gradient_accumulation_steps=2,
                    )

                    def fail_local_clip(*args, **kwargs):
                        raise RuntimeError(
                            "trainer-owned local clipping must not run "
                            "when DeepSpeed is active"
                        )

                    accelerator.clip_grad_norm_ = fail_local_clip
                    logged_lrs = {}

                    def record_log(values, step):
                        logged_lrs.setdefault(str(step), {}).update(values)

                    accelerator.log = record_log
                    optimizer = torch.optim.AdamW(
                        [
                            {"params": [model.linear.weight], "lr": 0.1},
                            {"params": [model.linear.bias], "lr": 0.2},
                            {"params": [model.extra[0]], "lr": 0.3},
                            {"params": [model.extra[1]], "lr": 0.4},
                            {"params": [model.extra[2]], "lr": 0.5},
                        ],
                        lr=base_lr,
                        weight_decay=0.0005,
                    )

                    trainer = HookBasedTrainer(
                        accelerator=accelerator,
                        model=model,
                        dataloader=dataloader,
                        batch_processor=TinyBatchProcessor(),
                        optimizer=optimizer,
                        lr_scheduler=build_scheduler,
                        grad_clip=GradientClippingHookConfig(
                            clip_mode="norm",
                            max_norm=0.2,
                        ),
                        hooks=[
                            StatsMonitorConfig(
                                batch_size=1,
                                step_log_freq=1,
                            )
                        ],
                        max_step=2,
                    )
                    trainer()

                    engine = accelerator.deepspeed_engine_wrapped.engine
                    result = {
                        "trainer_global_step_id": (
                            trainer.trainer_progress_state.global_step_id
                        ),
                        "trainer_micro_step_id": (
                            trainer.trainer_progress_state
                            .micro_step
                            .global_step_id
                        ),
                        "last_optimizer_step_size": (
                            trainer.trainer_progress_state
                            .micro_step
                            .last_optimizer_step_size
                        ),
                        "engine_global_steps": engine.global_steps,
                        "engine_skipped_steps": engine.skipped_steps,
                        "engine_lrs": engine.get_lr(),
                        "engine_gradient_clipping": engine.gradient_clipping(),
                        "scheduler_type": type(engine.lr_scheduler).__name__,
                        "basic_optimizer_type": type(
                            engine.basic_optimizer
                        ).__name__,
                        "runtime_optimizer_config": (
                            accelerator.deepspeed_config["optimizer"]
                        ),
                        "logged_lrs": logged_lrs,
                    }
                    accelerator.free_memory()
                    print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
                    """
                ),
            ],
            cwd=PROJECT_ROOT,
            env=subprocess_env,
            text=True,
            capture_output=True,
        )
        attempts.append((subprocess_env["MASTER_PORT"], completed))
        if completed.returncode == 0:
            break
        if not _completed_with_distributed_port_conflict(completed):
            break

    assert completed.returncode == 0, (
        "DeepSpeed smoke subprocess failed.\n"
        + _format_deepspeed_subprocess_attempts(attempts)
    )
    result = _parse_result_json(completed.stdout)

    assert result["trainer_global_step_id"] == 2
    assert result["trainer_micro_step_id"] == 4
    assert result["last_optimizer_step_size"] == 2
    assert result["engine_global_steps"] == 2
    assert result["engine_skipped_steps"] == 0
    assert result["scheduler_type"] == "StepLR"
    assert result["basic_optimizer_type"] == expected_basic_optimizer
    assert result["runtime_optimizer_config"] == {
        "type": "AdamW",
        "params": {
            "lr": 0.1,
            "weight_decay": 0.0005,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    }
    assert result["engine_gradient_clipping"] == pytest.approx(0.2)
    expected_lrs = [0.001, 0.002, 0.003, 0.004, 0.005]
    assert result["engine_lrs"] == pytest.approx(expected_lrs)
    assert result["logged_lrs"]["2"] == pytest.approx(
        {f"LR/group{idx}": lr for idx, lr in enumerate(expected_lrs)}
    )


def test_single_process_distributed_env_assigns_retry_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw2")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example")

    ports = [
        int(
            _single_process_distributed_env(
                port_offset=10,
                retry_index=retry_index,
            )["MASTER_PORT"]
        )
        for retry_index in range(3)
    ]

    assert len(set(ports)) == len(ports)
    assert ports[0] != 12345
    assert ports[1] - ports[0] == 100
    assert ports[2] - ports[1] == 100
    assert "HTTP_PROXY" not in _single_process_distributed_env()


def test_distributed_port_conflict_detects_torch_error() -> None:
    assert _completed_with_distributed_port_conflict(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="DistNetworkError: EADDRINUSE, address already in use",
        )
    )
    assert not _completed_with_distributed_port_conflict(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="RuntimeError: unrelated training failure",
            stderr="",
        )
    )


def _single_process_distributed_env(
    *,
    port_offset: int = 0,
    retry_index: int = 0,
) -> dict[str, str]:
    env = os.environ.copy()
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["MASTER_PORT"] = str(
        _stable_master_port(
            env,
            port_offset=port_offset,
            retry_index=retry_index,
        )
    )
    for proxy_key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        env.pop(proxy_key, None)
    return env


def _stable_master_port(
    env: dict[str, str],
    *,
    port_offset: int = 0,
    retry_index: int = 0,
) -> int:
    worker = env.get("PYTEST_XDIST_WORKER", "gw0")
    if worker.startswith("gw") and worker[2:].isdigit():
        worker_index = int(worker[2:])
    else:
        worker_index = 0
    return (
        _DEEPSPEED_MASTER_PORT_BASE
        + worker_index * _DEEPSPEED_MASTER_PORT_WORKER_STRIDE
        + port_offset
        + retry_index * _DEEPSPEED_MASTER_PORT_RETRY_STRIDE
    )


def _completed_with_distributed_port_conflict(
    completed: subprocess.CompletedProcess[str],
) -> bool:
    output = f"{completed.stdout}\n{completed.stderr}".lower()
    return any(
        marker in output for marker in _DISTRIBUTED_PORT_CONFLICT_MARKERS
    )


def _format_deepspeed_subprocess_attempts(
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]],
) -> str:
    sections = []
    for index, (master_port, completed) in enumerate(attempts, start=1):
        sections.append(
            "\n".join(
                [
                    (
                        f"attempt {index} MASTER_PORT={master_port} "
                        f"returncode={completed.returncode}"
                    ),
                    "stdout:",
                    completed.stdout,
                    "stderr:",
                    completed.stderr,
                ]
            )
        )
    return "\n\n".join(sections)


def _parse_result_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.removeprefix("RESULT_JSON="))
    raise AssertionError(f"RESULT_JSON marker not found in stdout:\n{stdout}")
