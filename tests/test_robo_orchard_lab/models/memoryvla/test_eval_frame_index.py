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

"""Where inference gets its frame index, and what breaks if it does not.

`HoloBrainProcessor` derives `step_index` from `len(history_joint_state) - 1`
(processor.py:158). Training feeds it the real frame index from the dataset;
`HoloBrainRoboDojoPolicy.data_preprocess` always builds that list with
exactly one entry, so on the deploy path the processor's answer is 0 on every
frame of every episode. `TimestepEmbedder` then hands the whole episode one
positional encoding: the bank keeps its contents and loses all sense of when
anything in it happened. Nothing raised, and nothing would have.

The policy counts `update_obs` instead. `deploy.py::eval_one_episode` calls
it once per env step and `get_action` once per `valid_action_step` of them,
so the count is the env frame index exactly -- including for episodes that
end early, where multiplying decisions by 32 would not be.

CPU only, no model, no GPU, no simulator.
"""

import os
import sys

import pytest

_COMMON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    *([os.pardir] * 4),
    "projects",
    "holobrain_internal",
    "common",
)
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from holobrain_robodojo_policy.deploy_policy import (  # noqa: E402
    HoloBrainRoboDojoPolicy,
)

VALID_ACTION_STEP = 32


class FakeMemory:
    """Stands in for structure.py's `memoryvla`, which is None when off."""


class FakeModel:
    def __init__(self, with_memory):
        self.memoryvla = FakeMemory() if with_memory else None
        self.seen = []

    def __call__(self, model_input):
        self.seen.append(dict(model_input))
        return "outputs"


class FakeProcessor:
    """pre_process ends in UnsqueezeBatch, so scalars arrive as 1-lists."""

    def __init__(self, with_step_index=True):
        self.with_step_index = with_step_index

    def pre_process(self, data):
        out = {"imgs": ["<tensor>"], "text": ["pick up the cube"]}
        if self.with_step_index:
            # what processor.py:158 computes on this path, every time
            out["step_index"] = [0]
        return out

    def post_process(self, outputs, model_input):
        return outputs


def bare(with_memory=True, with_step_index=True, pipeline=None):
    """A policy without __init__; building one needs a checkpoint on disk."""
    p = object.__new__(HoloBrainRoboDojoPolicy)
    p.cfg = None
    p.processor = FakeProcessor(with_step_index)
    p.model = None if pipeline is not None else FakeModel(with_memory)
    p.pipeline = pipeline
    p._obs = None
    p._batch_obs = {}
    p._env_step = 0
    return p


def step_env(policy, n):
    """Run n env steps, as deploy.py drives them: one update_obs each."""
    for _ in range(n):
        policy.update_obs({"env_idx": 0})


# -- the frame index ----------------------------------------------------------
def test_first_decision_of_an_episode_is_frame_zero():
    p = bare()
    step_env(p, 1)
    p._run_holobrain("data")
    assert p.model.seen[-1]["step_index"] == [0]


def test_frame_index_advances_by_valid_action_step():
    """One get_action per 32 update_obs is what eval_one_episode does."""
    p = bare()
    for decision in range(4):
        step_env(p, VALID_ACTION_STEP)
        p._run_holobrain("data")
        assert p.model.seen[-1]["step_index"] == [
            decision * VALID_ACTION_STEP + VALID_ACTION_STEP - 1
        ]


def test_frame_index_is_not_stuck_at_zero():
    """The whole defect, stated once: it used to be 0 forever."""
    p = bare()
    seen = []
    for _ in range(5):
        step_env(p, VALID_ACTION_STEP)
        p._run_holobrain("data")
        seen.append(p.model.seen[-1]["step_index"][0])
    assert len(set(seen)) == len(seen)
    assert seen == sorted(seen)


def test_an_episode_that_ends_early_still_reports_real_frames():
    """An episode cut short must still report the frames it really ran.

    A chunk ended early by is_episode_end gives fewer than 32 update_obs, so
    decisions * 32 would overcount; counting observations does not.
    """
    p = bare()
    step_env(p, VALID_ACTION_STEP)
    p._run_holobrain("data")
    step_env(p, 7)                       # episode ended mid-chunk
    p._run_holobrain("data")
    assert p.model.seen[-1]["step_index"] == [VALID_ACTION_STEP + 7 - 1]


def test_reset_puts_the_next_episode_back_at_frame_zero():
    p = bare()
    step_env(p, 100)
    p._run_holobrain("data")
    p.reset()
    step_env(p, 1)
    p._run_holobrain("data")
    assert p.model.seen[-1]["step_index"] == [0]


def test_frame_index_never_goes_negative():
    """A forward before the first observation must not underflow.

    wrapper.py's eval-side boundary guard rejects a step_index that goes
    backwards, so -1 here would be read as a missed reset().
    """
    p = bare()
    p.reset()
    p._run_holobrain("data")
    assert p.model.seen[-1]["step_index"] == [0]


# -- a baseline package must be untouched -------------------------------------
def test_a_processor_without_step_index_is_left_alone():
    """A baseline package must come through byte-for-byte as before.

    step_index is whitelisted only when the memory is on, so its absence is
    itself the switch; nothing may be injected into a baseline batch.
    """
    p = bare(with_memory=False, with_step_index=False)
    step_env(p, 10)
    p._run_holobrain("data")
    assert "step_index" not in p.model.seen[-1]


# -- the two paths that would silently be wrong -------------------------------
class FakePipeline:
    def __init__(self, with_memory):
        self.model = FakeModel(with_memory)

    def __call__(self, data):
        return "outputs"


def test_pipeline_path_refuses_a_memory_model():
    """The pipeline path cannot carry a frame index, so it must refuse.

    It calls pre_process and the model in one step, leaving nowhere to
    correct the index between them.
    """
    p = bare(pipeline=FakePipeline(with_memory=True))
    with pytest.raises(RuntimeError, match="frame index"):
        p._run_holobrain("data")


def test_pipeline_path_still_works_without_memory():
    p = bare(pipeline=FakePipeline(with_memory=False))
    assert p._run_holobrain("data") == "outputs"


def test_batched_eval_refuses_a_memory_model():
    """N envs, one episode identity: their memories would merge."""
    p = bare()
    p._batch_obs = {0: {"env_idx": 0}, 1: {"env_idx": 1}}
    with pytest.raises(RuntimeError, match="eval_batch"):
        p.get_action_batch([0, 1])


def test_batched_eval_of_one_env_is_allowed():
    """num_envs=1 is what main.py forces while eval_batch is false."""
    p = bare()
    p._batch_obs = {0: {"env_idx": 0}}
    p.predict_actions = lambda obs: __import__("numpy").zeros((32, 14))
    assert len(p.get_action_batch([0])) == 1


def test_batched_eval_of_many_envs_is_fine_without_memory():
    p = bare(with_memory=False, with_step_index=False)
    p._batch_obs = {0: {"env_idx": 0}, 1: {"env_idx": 1}}
    p.predict_actions = lambda obs: __import__("numpy").zeros((32, 14))
    assert len(p.get_action_batch([0, 1])) == 2
