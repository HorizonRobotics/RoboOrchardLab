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

"""Assembly-time guard for the memoryvla port: assert_episode_stream_wired.

Two of the guard's branches cannot be reached through train.py --kwargs
(`dataset_sample_weights` is not a declared top-level config key, so train.py
rejects it first; a type mismatch needs a dataloader the wiring would never
build), so they are exercised directly here. No dataset, no GPU, no
config file.

Why this lives in the repository rather than beside the port notes: the guard's
*text* is part of the guard. P1-B was not a missing check -- it was a check
whose message told the reader to disable the one thing that was working. P1-C
was not a missing check either -- it was a check that fired and named the wrong
cause, recommending a switch that was already on. Neither is visible to ruff,
to autoapi, or to any other gate. Only an assertion keeps those sentences from
coming back, and an assertion that no runner executes is a note, not a test.
"""

import pytest

from robo_orchard_lab.models.memoryvla.sampler import (
    MemoryVLAEpisodeStreamBatchSampler,
    _episode_spans,
    assert_episode_stream_wired,
)


class FakeLoader:
    def __init__(self, batch_sampler=None, sampler=None):
        self.batch_sampler = batch_sampler
        self.sampler = sampler


class FakeShard:
    """Stands in for accelerate's BatchSamplerShard."""

    def __init__(self, inner):
        self.batch_sampler = inner


class HostSampler:
    """Stands in for DistributedBatchFlagSampler."""

    batch_size = 4


def episode_sampler(batch_size=4):
    """An instance without __init__, which would scan a real dataset.

    ``_emit_repeat`` is left to the class default of 1 -- single process --
    which keeps every test in this file about the wiring checks rather than
    the shard composition; test_sampler_ddp.py covers that.
    """
    s = object.__new__(MemoryVLAEpisodeStreamBatchSampler)
    s.batch_size = batch_size
    return s


def cfg(enable=True, dl="stream", ess=True, weights=None, group_size=16,
        batch_size=None):
    c = {
        "memoryvla": {
            "enable": enable,
            "dataloader_type": dl,
            "episode_stream_sampler": ess,
            "group_size": group_size,
        }
    }
    if weights is not None:
        c["dataset_sample_weights"] = weights
    if batch_size is not None:
        c["batch_size"] = batch_size
    return c


def raise_text(config, loader):
    """The guard's message, or None when it returns cleanly."""
    try:
        assert_episode_stream_wired(config, loader)
    except RuntimeError as e:
        return str(e)
    return None


# --------------------------------------------------------------- wiring ----

QUIET = [
    ("off: switch disabled, nothing else matters",
     cfg(enable=False, dl="stream", ess=False), FakeLoader(HostSampler())),
    ("off: not even a dataloader", cfg(enable=False), None),
    ("off: group + no sampler is still none of the guard's business",
     cfg(enable=False, dl="group", ess=False), FakeLoader(HostSampler())),
    ("eval_only: enabled but no train dataloader", cfg(), None),
    ("wired: episode sampler directly on the loader",
     cfg(), FakeLoader(episode_sampler())),
    ("wired: episode sampler behind an accelerate-style shard",
     cfg(), FakeLoader(FakeShard(episode_sampler()))),
    ("wired: group mode with the episode sampler -- a working config",
     cfg(dl="group", ess=True), FakeLoader(episode_sampler())),
    ("wired: group behind a shard too",
     cfg(dl="group", ess=True), FakeLoader(FakeShard(episode_sampler()))),
    ("wired: the chain decides, not the key -- ess=False but sampler present",
     cfg(dl="stream", ess=False), FakeLoader(episode_sampler())),
]

LOUD = [
    ("stream, trainer is iterating the host sampler",
     cfg(dl="stream", ess=False), FakeLoader(HostSampler()),
     "not in the chain"),
    ("group + sampler off -- the P1-B hole",
     cfg(dl="group", ess=False), FakeLoader(HostSampler()),
     "not in the chain"),
    ("group + sampler on but host sampler actually iterated",
     cfg(dl="group", ess=True), FakeLoader(HostSampler()), "not in the chain"),
    ("no sampler at all", cfg(), FakeLoader(None), "no sampler at all"),
    ("sample weights would be silently dropped",
     cfg(weights=[1.0]), FakeLoader(episode_sampler()),
     "dataset_sample_weights"),
    ("sample weights, group mode",
     cfg(dl="group", weights=[1.0]), FakeLoader(episode_sampler()),
     "dataset_sample_weights"),
]


@pytest.mark.parametrize("name,config,loader", QUIET)
def test_guard_stays_quiet(name, config, loader):
    assert raise_text(config, loader) is None, name


@pytest.mark.parametrize("name,config,loader,expect", LOUD)
def test_guard_raises(name, config, loader, expect):
    got = raise_text(config, loader)
    assert got is not None, "{}: did not raise".format(name)
    assert expect in got, "{}: wrong message: {}".format(name, got[:120])


# ------------------------------------------------- P1-C: the memory span ----
# `group` clears the bank at the top of every training call and again every
# group_size samples inside the batch, so memory reaches
# min(group_size, batch_size) samples. At 1 nothing can ever be retrieved: the
# module is on, holds 7.47M parameters, and computes an exact identity. This is
# decidable here, with no forward -- which is the point. The consumer-side
# watchdog needs K forwards before it can rule, so a 4-step smoke run used to
# pass this configuration in silence.

SPAN_DEAD = [
    ("group, batch 1", "group", 1, 16),
    ("group, batch 1, group_size 1", "group", 1, 1),
    ("group, group_size 1 at batch 4 -- same failure, different key",
     "group", 4, 1),
    ("group, group_size 0", "group", 4, 0),
]

SPAN_ALIVE = [
    ("group, batch 2, group_size 2 -- smallest working group",
     "group", 2, 2),
    ("group, batch 4, group_size 16", "group", 4, 16),
    ("group, batch 4, group_size 2", "group", 4, 2),
    ("stream, batch 1 is legal -- the bank carries across calls",
     "stream", 1, 16),
    ("stream, batch 1, group_size 1 -- group_size is inert in stream",
     "stream", 1, 1),
    ("stream, batch 4", "stream", 4, 16),
]


@pytest.mark.parametrize("name,dl,bs,gs", SPAN_DEAD)
def test_span_of_one_is_rejected(name, dl, bs, gs):
    got = raise_text(
        cfg(dl=dl, group_size=gs), FakeLoader(episode_sampler(bs))
    )
    assert got is not None, "{}: did not raise".format(name)
    assert "cannot hold any memory" in got, got[:160]


@pytest.mark.parametrize("name,dl,bs,gs", SPAN_ALIVE)
def test_workable_spans_are_left_alone(name, dl, bs, gs):
    got = raise_text(
        cfg(dl=dl, group_size=gs), FakeLoader(episode_sampler(bs))
    )
    assert got is None, "{}: unexpected raise: {}".format(
        name, (got or "")[:160])


def test_span_read_through_an_accelerate_shard():
    """prepare() re-wraps, so batch size must be found through the shard."""
    got = raise_text(
        cfg(dl="group"), FakeLoader(FakeShard(episode_sampler(1)))
    )
    assert got is not None and "cannot hold any memory" in got


def test_span_falls_back_to_the_config_batch_size():
    """A sampler with no batch_size attribute must not silence the check."""
    s = object.__new__(MemoryVLAEpisodeStreamBatchSampler)   # no batch_size
    got = raise_text(cfg(dl="group", batch_size=1), FakeLoader(s))
    assert got is not None and "config['batch_size']" in got


def test_span_check_skips_loudly_when_no_batch_size_is_findable():
    """Unreadable batch size must warn, not silently pass.

    A check that cannot run and says nothing is indistinguishable from a check
    that ran and approved -- that equivalence is what P0-1 was made of.
    """
    import logging

    s = object.__new__(MemoryVLAEpisodeStreamBatchSampler)
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    lg = logging.getLogger("robo_orchard_lab.models.memoryvla.sampler")
    h = _Capture()
    lg.addHandler(h)
    try:
        got = raise_text(cfg(dl="group"), FakeLoader(s))
    finally:
        lg.removeHandler(h)
    assert got is None
    assert any("being skipped" in m for m in records), records


def test_span_message_names_the_batch_size_it_used():
    got = raise_text(cfg(dl="group"), FakeLoader(episode_sampler(1)))
    assert "batch_size=1" in got
    assert "group_size=16" in got
    assert "min(16, 1) = 1" in got


# ----------------------------------------------------- message hygiene -----
# Two failures in this port were caused by raise text, not by missing checks.
# These assertions are the only thing stopping either from returning.

FORBIDDEN = [
    # P1-B: routed the reader into the one unguarded cell
    "episode sampler is only meaningful",
    "turn the episode sampler off",
    "switch the bank to dataloader_type='group'",
    "episode_stream_sampler=False",
]


def all_guard_messages():
    msgs = [raise_text(c, ld) for _, c, ld, _ in LOUD]
    msgs += [
        raise_text(cfg(dl=dl, group_size=gs), FakeLoader(episode_sampler(bs)))
        for _, dl, bs, gs in SPAN_DEAD
    ]
    try:
        _episode_spans(object())
    except TypeError as e:
        msgs.append(str(e))
    except Exception:                                          # noqa: BLE001
        pass
    return [m for m in msgs if m]


@pytest.mark.parametrize("phrase", FORBIDDEN)
def test_no_message_routes_the_reader_into_the_hole(phrase):
    for m in all_guard_messages():
        assert phrase not in m, "forbidden phrase back in: {}".format(m[:160])


def test_wrong_sampler_messages_name_the_actual_fix():
    for _, config, loader, expect in LOUD:
        if expect != "not in the chain":
            continue
        assert "episode_stream_sampler=True" in raise_text(config, loader)


@pytest.mark.parametrize("name,dl,bs,gs", SPAN_DEAD)
def test_span_message_does_not_recommend_an_already_effective_switch(
    name, dl, bs, gs
):
    """P1-C in one assertion.

    The span failure is reached with episode_stream_sampler already True and
    the batches already episode-contiguous. Telling the reader to set it to
    True is advice that changes nothing -- the exact aggravating factor that
    made P1-B a P1. The message must say the sampler is not the problem.
    """
    got = raise_text(
        cfg(dl=dl, group_size=gs), FakeLoader(episode_sampler(bs)))
    assert "so set memoryvla.episode_stream_sampler=True" not in got
    assert "the fix is memoryvla.episode_stream_sampler" not in got
    assert "episode sampler is NOT the problem" in got


@pytest.mark.parametrize("name,dl,bs,gs", SPAN_DEAD)
def test_span_message_offers_actions_that_work_here(name, dl, bs, gs):
    got = raise_text(
        cfg(dl=dl, group_size=gs), FakeLoader(episode_sampler(bs)))
    assert "dataloader_type='stream'" in got
    assert "batch_size >= 2 AND group_size >= 2" in got
