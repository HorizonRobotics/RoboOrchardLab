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

"""Consumer-side guards inside MemoryVLAMemory.forward.

A probe that has never been seen to trip is not evidence, and a probe that has
never been seen to stay quiet is a liability. Each guard here is driven with
synthetic state so it is shown firing AND shown staying quiet -- the pair is
the point. CPU only, no dataset, no GPU.

The guards, and what each is for:

  _check_episode_stream   the first training batch. Two reasons to stop: the
                          configuration cannot hold memory at all (`group` with
                          min(group_size, batch_size) == 1), or the batch that
                          arrived has one sample per episode so nothing can
                          ever read history.
  _check_bank_liveness    after K training forwards, did any episode's memory
                          ever exceed one entry. This is the only guard that
                          does not need history to already exist, which is
                          exactly what made the other two silent in the failure
                          they were written for.
  _assert_not_identity    did the module return its input back.
  _history_will_be_read   would a probe be meaningful on this batch at all.
"""

import pytest
import torch

from robo_orchard_lab.models.memoryvla.wrapper import (
    IDENTITY_TOL,
    MemoryVLAMemory,
)

K = MemoryVLAMemory.BANK_LIVENESS_FORWARDS


class FakeBank:
    """Just the attribute the guards read: .bank, episode id -> entry list."""

    def __init__(self, lengths):
        self.bank = {"ep{}".format(i): [None] * n
                     for i, n in enumerate(lengths)}


def bare(dl_type="stream", banks=None, training=True, group_size=16,
         mem_length=16):
    """A MemoryVLAMemory without __init__; building one needs a config."""
    m = object.__new__(MemoryVLAMemory)
    m.training = training
    m.dataloader_type = dl_type
    m.group_size = group_size
    m.mem_length = mem_length
    m._episode_check_done = False
    m._identity_check_done = False
    m._bank_liveness_checked = False
    m._train_forwards = 0
    m._max_bank_len_seen = 0
    m._batch_sizes_seen = set()
    m._distinct_in_batch_seen = set()
    m.per_mem_bank = banks
    m.cog_mem_bank = banks
    m.use_perceptual = True
    m.use_cognitive = True
    return m


def drive(m, per_call_lengths, ids=("a",), batch_size=1):
    """Run the watchdog once per entry, with the bank in the given shape."""
    for lengths in per_call_lengths:
        m.per_mem_bank = FakeBank(lengths)
        m.cog_mem_bank = FakeBank(lengths)
        m._check_bank_liveness(list(ids), batch_size)


def raised(fn):
    try:
        fn()
    except RuntimeError as e:
        return str(e)
    return None


# ------------------------------------------- 1. bank liveness watchdog -----

def test_bank_stuck_at_one_for_k_forwards_raises():
    """The P0-1 / P1-B signature: history never appears, nothing notices."""
    msg = raised(lambda: drive(bare(), [[1]] * K))
    assert msg is not None and "grew past a single entry" in msg


def test_healthy_growth_is_left_alone():
    assert raised(lambda: drive(bare(), [[i + 1] for i in range(K)])) is None


def test_bank_reaching_two_on_the_last_watched_forward_is_healthy():
    assert raised(lambda: drive(bare(), [[1]] * (K - 1) + [[2]])) is None


def test_no_verdict_before_k_forwards():
    """K is a time gate. This test documents the gap rather than hiding it.

    Nothing is decided before the K-th forward, so a run shorter than K gets
    nothing from this guard. That was half of P1-C: a 4-step smoke run through
    a configuration that could not hold memory finished rc=0 in silence. The
    configurations whose failure is decidable without running are now rejected
    earlier -- by assert_episode_stream_wired and by _check_episode_stream on
    the very first forward -- so what is left to K is the class that genuinely
    needs forwards to observe: batches that were supposed to be
    episode-contiguous and are not.
    """
    m = bare()
    assert raised(lambda: drive(m, [[1]] * (K - 1))) is None
    assert m._bank_liveness_checked is False


def test_group_mode_stuck_at_one_raises():
    msg = raised(lambda: drive(bare("group"), [[1]] * K, batch_size=1))
    assert msg is not None and "grew past a single entry" in msg


def test_group_mode_reaching_batch_size_is_healthy():
    assert raised(
        lambda: drive(bare("group"), [[4]] * K, ids=("a",) * 4, batch_size=4)
    ) is None


def test_eval_mode_is_never_judged():
    assert raised(lambda: drive(bare(training=False), [[1]] * K)) is None


def test_a_future_dataloader_type_is_still_covered():
    """The criterion is the consequence, so unknown modes inherit the cover."""
    msg = raised(lambda: drive(bare("some_new_mode"), [[1]] * K))
    assert msg is not None and "grew past a single entry" in msg


def test_mem_length_of_one_is_not_a_false_positive():
    """mem_length=1 caps bank length at 1 while memory still works.

    Consolidation trims the bank back to a single entry after every write, so
    length can never exceed 1 -- but that entry is a real merged history and IS
    retrieved (memory_bank.py: `hist = self.bank.get(eid, [])` then
    `if len(hist) > 0`). Raising here would fail a run that is behaving as
    documented, which is the most complete way for a guard to point at the
    wrong cause.
    """
    m = bare(mem_length=1)
    assert raised(lambda: drive(m, [[1]] * K)) is None
    assert m._bank_liveness_checked is True   # it ruled; it just did not fail


def test_mem_length_of_one_says_it_is_standing_down():
    """Standing down silently would be indistinguishable from approving."""
    import logging

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.levelname + ":" + record.getMessage())

    lg = logging.getLogger("robo_orchard_lab.models.memoryvla.wrapper")
    h = _Capture()
    lg.addHandler(h)
    if lg.level > logging.INFO or lg.level == logging.NOTSET:
        lg.setLevel(logging.INFO)
    try:
        drive(bare(mem_length=1), [[1]] * K)
    finally:
        lg.removeHandler(h)
    assert any(r.startswith("WARNING") and "standing down" in r
               for r in records), records


# ----------------------------------- 2. first-batch checks (P1-C, group) ---

SPAN_DEAD = [
    ("group, batch 1", "group", ["a"], 1, 16),
    ("group, batch 1, group_size 1", "group", ["a"], 1, 1),
    ("group, group_size 1 at batch 4", "group", ["a"] * 4, 4, 1),
]


@pytest.mark.parametrize("name,dl,ids,bs,gs", SPAN_DEAD)
def test_group_span_of_one_raises_on_the_first_forward(name, dl, ids, bs, gs):
    m = bare(dl, group_size=gs)
    msg = raised(lambda: m._check_episode_stream(ids, bs))
    assert msg is not None, name
    assert "cannot hold any memory" in msg, msg[:160]


@pytest.mark.parametrize("name,dl,ids,bs,gs", SPAN_DEAD)
def test_group_span_message_does_not_blame_the_sampler(name, dl, ids, bs, gs):
    """P1-C: the sampler is wired and the batches are contiguous here."""
    m = bare(dl, group_size=gs)
    msg = raised(lambda: m._check_episode_stream(ids, bs))
    assert "episode sampler is NOT the problem" in msg
    assert "the fix is memoryvla.episode_stream_sampler" not in msg
    assert "dataloader_type='stream'" in msg
    assert "batch_size >= 2 AND group_size >= 2" in msg


def test_stream_at_batch_one_is_legal():
    """Stream carries the bank across calls, so batch 1 is a fine config."""
    m = bare("stream")
    assert raised(lambda: m._check_episode_stream(["a"], 1)) is None


def test_stream_at_batch_one_with_group_size_one_is_still_legal():
    """group_size is inert outside `group` and must not trip anything."""
    m = bare("stream", group_size=1)
    assert raised(lambda: m._check_episode_stream(["a"], 1)) is None


def test_group_with_a_workable_span_stays_quiet():
    m = bare("group", group_size=2)
    assert raised(lambda: m._check_episode_stream(["a", "a"], 2)) is None


SPREAD = [
    ("stream, 4 samples from 4 different episodes",
     "stream", ["a", "b", "c", "d"], 4, "different episodes"),
    ("stream, 4 samples from 1 episode", "stream", ["a"] * 4, 4, None),
    ("group, 4 samples from 4 different episodes -- was unchecked before",
     "group", ["a", "b", "c", "d"], 4, "different episodes"),
    ("group, two groups of two -- a legitimate group layout",
     "group", ["a", "a", "b", "b"], 4, None),
    ("batch of 1 cannot be judged by the spread criterion",
     "stream", ["a"], 1, None),
]


@pytest.mark.parametrize("name,dl,ids,bs,expect", SPREAD)
def test_batch_episode_spread(name, dl, ids, bs, expect):
    msg = raised(lambda: bare(dl)._check_episode_stream(ids, bs))
    if expect is None:
        assert msg is None, "{}: raised {}".format(name, (msg or "")[:120])
    else:
        assert msg is not None and expect in msg, name


def test_eval_mode_may_span_many_episodes():
    m = bare(training=False)
    ids = ["a", "b", "c", "d"]
    assert raised(lambda: m._check_episode_stream(ids, 4)) is None


# ------------------------------------------------- 3. identity probe -------

X = torch.randn(2, 4, 8)


def ident(per_gap, cog_gap):
    bare()._assert_not_identity(X, X + per_gap, X, X + cog_gap)


@pytest.mark.parametrize("name,per,cog,must_raise", [
    ("both streams exactly identity", 0.0, 0.0, True),
    ("both 6e-08 apart -- the measured degenerate gap", 6e-8, 6e-8, True),
    ("perceptual alive, cognitive dead -- min not max", 1.0, 0.0, True),
    ("both streams alive", 1.0, 1.0, False),
    ("gap exactly half the tolerance",
     IDENTITY_TOL / 2, IDENTITY_TOL / 2, True),
])
def test_identity_probe(name, per, cog, must_raise):
    msg = raised(lambda: ident(per, cog))
    assert (msg is not None) is must_raise, name


# -------------------------------------------- 4. _history_will_be_read -----

def hwr(dl_type, ids, bank_lengths):
    m = bare(dl_type)
    b = FakeBank([])
    b.bank = dict(bank_lengths)
    m.per_mem_bank = b
    m.cog_mem_bank = b
    return m._history_will_be_read(ids)


@pytest.mark.parametrize("name,dl,ids,bank,want", [
    ("stream: repeated id inside the batch", "stream", ["a", "a"], {}, True),
    ("stream: distinct ids but the bank already holds one",
     "stream", ["a", "b"], {"a": [None]}, True),
    ("stream: distinct ids, empty bank -- episode first frame",
     "stream", ["a", "b"], {}, False),
    ("group: repeated id inside the batch is written and read in this call",
     "group", ["a", "a"], {}, True),
    ("group: leftover bank entries do not count -- group clears on entry",
     "group", ["a", "b"], {"a": [None]}, False),
])
def test_history_will_be_read(name, dl, ids, bank, want):
    assert hwr(dl, ids, bank) is want, name


# ------------------------------------------ 5. watchdog message hygiene ----

def watchdog_message():
    return raised(lambda: drive(bare(), [[1]] * K))


def test_watchdog_does_not_assert_a_cause_it_cannot_distinguish():
    """P1-C, the other half.

    "bank never exceeded 1" has two possible causes -- the batches are broken,
    or this configuration could never hold memory -- and bank length cannot
    tell them apart. The previous text asserted the first ("The batches
    reaching this module are not episode-contiguous") in a configuration where
    the batches were, measurably, contiguous.

    Naming non-contiguity is fine; naming it as *the* cause is not. So the
    assertion is on the sentence, not on the words: the declarative form has
    to be gone, both candidates have to be present, and the message has to
    say it cannot choose between them.
    """
    msg = watchdog_message()
    assert "The batches reaching this module are" not in msg
    assert "Two different things produce this" in msg
    assert "cannot tell them apart" in msg
    assert "(a)" in msg and "(b)" in msg


def test_watchdog_does_not_prescribe_an_already_effective_switch():
    """It may mention the switch, but only conditionally on it being off."""
    msg = watchdog_message()
    assert "so the fix is memoryvla.episode_stream_sampler=True" not in msg
    low = msg.lower()
    assert "if memoryvla.episode_stream_sampler is off, turn it on" in low
    assert "if it is already on" in low


def test_watchdog_reports_what_it_observed():
    """Observations first, advice second, so the advice is checkable."""
    msg = watchdog_message()
    for token in ("dataloader_type=", "group_size=", "mem_length=",
                  "batch sizes seen=", "distinct episodes per batch seen=",
                  "longest bank="):
        assert token in msg, token


FORBIDDEN = [
    "episode sampler is only meaningful",
    "turn the episode sampler off",
    "switch the bank to dataloader_type='group'",
    "episode_stream_sampler=False",
]


@pytest.mark.parametrize("phrase", FORBIDDEN)
def test_no_consumer_side_message_routes_the_reader_into_the_hole(phrase):
    blob = "\n".join(
        m for m in (
            watchdog_message(),
            raised(lambda: bare("group", group_size=1)
                   ._check_episode_stream(["a"] * 4, 4)),
            raised(lambda: bare()._check_episode_stream(["a", "b"], 2)),
            raised(lambda: ident(0.0, 0.0)),
        ) if m
    )
    assert phrase not in blob
