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

"""The inference path, which until now nothing covered.

Four rounds of review hardened the training path until "switched on and
computing nothing" became unreachable there. None of it applied at inference:
all three consumer-side guards return early when ``self.training`` is unset,
and the deployed input does not even carry the fields the module reads. A
memoryvla-enabled model sent through RoboDojo eval raised ``KeyError`` on the
first forward -- and the message told the reader to edit a dataset config,
which at inference is an instruction that cannot be carried out.

What is asserted here:

  * the deployed batch shape (no ``uuid``, scalar ``step_index``) is accepted,
    and the training batch shape still behaves exactly as before;
  * episode identity at inference comes from ``reset()`` and has the two
    properties ``uuid`` has during training -- constant within an episode,
    distinct across episodes;
  * a caller that forgets ``reset()`` is caught rather than silently
    retrieving one episode's history while acting in the next;
  * an episode that never retrieved anything says so.

Each guard is shown firing AND shown staying quiet. A probe never seen to trip
is not evidence, and one never seen to stay quiet is a liability.

CPU only, no dataset, no GPU.
"""

import logging

import pytest

from robo_orchard_lab.models.memoryvla.wrapper import MemoryVLAMemory

#: Phrases that must never appear in an inference-side error. Each one sends
#: the reader somewhere that does not exist at inference time. This is the
#: same class of defect as P1-B and P1-C, on the path that had no guard.
EVAL_FORBIDDEN = (
    "ItemSelection",
    "dataset config",
)


class CaptureLogs(logging.Handler):
    """Collect this module's log records.

    Not ``caplog``: the offline runner this box uses in place of pytest
    (``.git/run_tests_nopytest.py``, since installing pytest into the host env
    would break the zero-change rule) supplies no fixtures, and a test it
    cannot resolve is reported SKIP. Three silently skipped assertions about
    guard output would be exactly the hole these tests exist to close.
    """

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def __enter__(self):
        self.logger = logging.getLogger(
            "robo_orchard_lab.models.memoryvla.wrapper"
        )
        self._old_level = self.logger.level
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self)
        self.logger.setLevel(self._old_level)
        return False

    def warnings(self):
        return [r for r in self.records if r.levelno >= logging.WARNING]


class FakeBank:
    """Only what reset()/memory_stats()/_history_will_be_read touch."""

    def __init__(self, lengths=()):
        self.bank = {"ep{}".format(i): [None] * n
                     for i, n in enumerate(lengths)}

    def reset(self):
        self.bank = {}

    def clear_episode(self, eid):
        self.bank.pop(eid, None)


def bare_eval(training=False, banks=None, use_timestep_pe=True):
    """A MemoryVLAMemory without __init__; building one needs a config."""
    m = object.__new__(MemoryVLAMemory)
    m.training = training
    m.dataloader_type = "stream"
    m.episode_id_key = "uuid"
    m.timestep_key = "step_index"
    m.use_timestep_pe = use_timestep_pe
    m.use_perceptual = True
    m.use_cognitive = True
    m.per_mem_bank = banks if banks is not None else FakeBank()
    m.cog_mem_bank = banks if banks is not None else FakeBank()
    m._last_episode_ids = None
    m._eval_episode = 0
    m._eval_forwards = 0
    m._eval_history_reads = 0
    m._last_timestep_seen = None
    return m


#: Exactly what deploy_policy hands the model: MultiArmManipulationInput has
#: no uuid field, and processor.struction_to_dict sets step_index to a scalar
#: because nothing collates between pre_process and model().
def deployed_batch(step_index=0):
    return {
        "image": "<tensor>",
        "instruction": "pick up the cube",
        "step_index": step_index,
    }


# -- the deployed batch shape -------------------------------------------------
def test_deployed_batch_has_no_uuid_and_that_is_not_an_error():
    m = bare_eval()
    ids = m._episode_ids(deployed_batch(), batch_size=1)
    assert len(ids) == 1
    assert ids[0] is not None


def test_scalar_step_index_is_accepted_at_inference():
    """No collate runs between pre_process and model() on the eval path."""
    m = bare_eval()
    assert m._timesteps(deployed_batch(step_index=7), batch_size=1) == [7]


def test_scalar_step_index_is_broadcast_to_the_batch():
    m = bare_eval()
    got = m._timesteps(deployed_batch(step_index=3), batch_size=3)
    assert got == [3, 3, 3]


def test_list_step_index_still_works_unchanged():
    m = bare_eval(training=True)
    batch = {"step_index": [0, 1, 2]}
    assert m._timesteps(batch, batch_size=3) == [0, 1, 2]


# -- the training path must not have moved ------------------------------------
def test_training_batch_without_uuid_still_raises():
    m = bare_eval(training=True)
    with pytest.raises(KeyError):
        m._episode_ids(deployed_batch(), batch_size=1)


def test_training_message_says_it_is_about_the_training_batch():
    """It used to be phrased as if there were only one path."""
    m = bare_eval(training=True)
    try:
        m._episode_ids(deployed_batch(), batch_size=1)
    except KeyError as e:
        assert "training batch" in str(e)
    else:
        raise AssertionError("expected KeyError")


# -- episode identity at inference --------------------------------------------
def test_eval_episode_id_is_constant_within_one_episode():
    m = bare_eval()
    first = m._episode_ids(deployed_batch(0), 1)
    later = m._episode_ids(deployed_batch(5), 1)
    assert first == later


def test_eval_episode_id_changes_after_reset():
    """The property uuid provides during training: distinct across episodes."""
    m = bare_eval()
    before = m._episode_ids(deployed_batch(0), 1)
    m.reset()
    after = m._episode_ids(deployed_batch(0), 1)
    assert before != after


def test_reset_empties_the_banks():
    m = bare_eval(banks=FakeBank([3, 4]))
    assert m.per_mem_bank.bank
    m.reset()
    assert m.per_mem_bank.bank == {}


# -- the caller that forgets reset() ------------------------------------------
def test_step_index_going_backwards_without_reset_is_caught():
    m = bare_eval()
    m._check_eval_episode_boundary([5])
    with pytest.raises(RuntimeError) as excinfo:
        m._check_eval_episode_boundary([0])
    assert "reset()" in str(excinfo.value)


def test_reset_between_episodes_is_not_a_false_positive():
    """The whole point: this must stay quiet when the caller behaves."""
    m = bare_eval()
    for t in range(4):
        m._check_eval_episode_boundary([t])
    m.reset()
    for t in range(4):
        m._check_eval_episode_boundary([t])          # must not raise


def test_monotonic_step_index_is_not_a_false_positive():
    m = bare_eval()
    for t in (0, 1, 2, 9, 40):
        m._check_eval_episode_boundary([t])


def test_boundary_message_does_not_send_the_reader_to_a_dataset_config():
    m = bare_eval()
    m._check_eval_episode_boundary([5])
    try:
        m._check_eval_episode_boundary([0])
    except RuntimeError as e:
        text = str(e)
        for phrase in EVAL_FORBIDDEN:
            assert phrase not in text, f"eval-side message mentions {phrase!r}"
        assert "step_index" in text
    else:
        raise AssertionError("expected RuntimeError")


def test_boundary_check_is_inert_without_timestep_pe():
    m = bare_eval(use_timestep_pe=False)
    m._check_eval_episode_boundary(None)
    m._check_eval_episode_boundary(None)


# -- did this episode use its memory at all -----------------------------------
def test_episode_that_never_retrieved_history_says_so():
    m = bare_eval()
    m._eval_forwards = 12
    m._eval_history_reads = 0
    with CaptureLogs() as cap:
        m.reset()
    assert len(cap.warnings()) == 1
    assert "never once retrieved history" in cap.warnings()[0].getMessage()


def test_episode_that_did_retrieve_history_stays_quiet():
    m = bare_eval()
    m._eval_forwards = 12
    m._eval_history_reads = 11
    with CaptureLogs() as cap:
        m.reset()
    assert cap.warnings() == []


def test_reset_before_any_forward_says_nothing():
    """Policies reset at startup too; that must not warn."""
    m = bare_eval()
    with CaptureLogs() as cap:
        m.reset()
    assert cap.warnings() == []
    assert cap.records == []


def test_memory_stats_reports_the_counters():
    m = bare_eval(banks=FakeBank([2, 3]))
    m._eval_forwards = 5
    m._eval_history_reads = 4
    stats = m.memory_stats()
    assert stats["eval_forwards"] == 5
    assert stats["eval_history_reads"] == 4
    assert stats["bank_lengths"]["per_mem_bank"] == [2, 3]


def test_counters_restart_with_each_episode():
    m = bare_eval()
    m._eval_forwards = 9
    m._eval_history_reads = 9
    m.reset()
    assert m.memory_stats()["eval_forwards"] == 0
    assert m.memory_stats()["eval_history_reads"] == 0
