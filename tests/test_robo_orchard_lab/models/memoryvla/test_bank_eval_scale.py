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

"""Consolidation, at the length an episode actually runs.

Every run of this port so far has been 4 to 20 training steps. At batch 4
that fills the bank to `mem_length` and stops -- the `while len(bank) >
mem_length` loop in `_memory_consolidate` was entered a handful of times at
most, and the branch that decides *which* two entries merge has never been
exercised more than a few times in a row. Evaluation is not like that: a
RoboDojo episode runs 276 to 1203 frames and the policy acts every 32, so
consolidation runs on nearly every forward for the whole episode.

The sharp question is not "does the bank stay bounded" -- fifo does that too
-- but "does it still know about the start of the episode". `tome` merges the
two most similar *adjacent* entries and keeps the earlier one's timestep, so
timestep 0 survives to the end. `fifo` drops from the front, so it cannot.
That difference is what makes this memory rather than a window, and nothing
until now checked it past 20 steps.

CPU only, small dims, no dataset, no GPU.
"""

import torch

from robo_orchard_lab.models.memoryvla.memory_bank import CogMemBank

TOKEN_SIZE = 64
MEM_LENGTH = 16
N_TOKENS = 3
#: 1200 env frames at one decision per 32 -- the longest Memory task
#: (imitate_sorting_sequence) rounded to something a test can afford.
EVAL_FORWARDS = 40


def build(consolidate_type="tome", mem_length=MEM_LENGTH):
    bank = CogMemBank(
        dataloader_type="stream",
        group_size=MEM_LENGTH,
        token_size=TOKEN_SIZE,
        mem_length=mem_length,
        retrieval_layers=1,
        use_timestep_pe=True,
        fusion_type="gate",
        consolidate_type=consolidate_type,
    )
    bank.eval()
    return bank


def run_episode(bank, forwards=EVAL_FORWARDS, stride=32, eid="ep-0"):
    """One episode of inference: batch of 1, decisions `stride` frames apart.

    That spacing is what valid_action_step makes the eval loop do.
    """
    torch.manual_seed(0)
    for k in range(forwards):
        tokens = torch.randn(1, N_TOKENS, TOKEN_SIZE)
        with torch.no_grad():
            bank.process_batch(tokens, [eid], [k * stride])
    return bank.bank[eid]


def test_bank_stays_bounded_across_a_whole_episode():
    bank = build()
    entries = run_episode(bank)
    assert len(entries) == MEM_LENGTH


def test_consolidation_actually_ran():
    """Without it the bank would be EVAL_FORWARDS long."""
    bank = build()
    entries = run_episode(bank)
    assert EVAL_FORWARDS > MEM_LENGTH
    assert len(entries) < EVAL_FORWARDS


def test_tome_still_holds_the_start_of_the_episode():
    """The property that makes this memory and not a sliding window."""
    bank = build("tome")
    entries = run_episode(bank)
    assert entries[0][0] == 0, (
        "timestep 0 fell out of the bank; tome merges adjacent entries and "
        "keeps the earlier timestep, so it should never leave"
    )


def test_fifo_does_not_hold_the_start_of_the_episode():
    """The contrast that gives the test above its meaning."""
    bank = build("fifo")
    entries = run_episode(bank)
    assert entries[0][0] > 0


def test_timesteps_stay_ordered_after_many_merges():
    bank = build()
    entries = run_episode(bank)
    stamps = [t for t, _ in entries]
    assert stamps == sorted(stamps)


def test_the_newest_frames_timestamp_is_not_guaranteed_to_survive():
    """Written asserting the opposite, which is how this was found.

    `tome` takes the most similar *adjacent* pair anywhere in the bank --
    the last one included -- and keeps the earlier entry's timestep
    (memory_bank.py:313-318). So the frame just observed can be averaged
    into its predecessor and lose its timestamp immediately. That is
    upstream's behaviour, not a porting slip, and it costs the current
    decision nothing, because retrieval runs before consolidation. What it
    does mean is that the newest timestamp in the bank is not a reliable
    "how far has this episode got" signal for anything reading it later.

    Forced rather than observed: the last two frames are identical, so
    cosine similarity puts the argmax on the final pair with no dependence
    on the seed or on how torch initialises anything.
    """
    bank = build(mem_length=4)
    torch.manual_seed(3)
    frames = [torch.randn(1, N_TOKENS, TOKEN_SIZE) for _ in range(4)]
    frames.append(frames[-1].clone())          # identical to its predecessor
    for k, tokens in enumerate(frames):
        with torch.no_grad():
            bank.process_batch(tokens, ["ep-0"], [k * 32])

    stamps = [t for t, _ in bank.bank["ep-0"]]
    assert stamps == [0, 32, 64, 96], stamps
    assert 128 not in stamps


def test_a_short_episode_never_consolidates():
    """Below mem_length nothing merges, so every frame is kept verbatim."""
    bank = build()
    entries = run_episode(bank, forwards=MEM_LENGTH - 1)
    assert [t for t, _ in entries] == [k * 32 for k in range(MEM_LENGTH - 1)]


def test_reset_between_episodes_leaves_nothing_behind():
    bank = build()
    run_episode(bank, eid="ep-0")
    bank.reset()
    entries = run_episode(bank, forwards=3, eid="ep-1")
    assert list(bank.bank) == ["ep-1"]
    assert len(entries) == 3


def test_two_episodes_kept_apart_when_not_reset():
    """clear_episode is what the wrapper uses; keys must not collide."""
    bank = build()
    run_episode(bank, forwards=5, eid="ep-0")
    run_episode(bank, forwards=3, eid="ep-1")
    assert len(bank.bank["ep-0"]) == 5
    assert len(bank.bank["ep-1"]) == 3
    bank.clear_episode("ep-0")
    assert list(bank.bank) == ["ep-1"]


def test_retrieval_output_keeps_its_shape_over_a_long_episode():
    bank = build()
    torch.manual_seed(1)
    for k in range(EVAL_FORWARDS):
        tokens = torch.randn(1, N_TOKENS, TOKEN_SIZE)
        with torch.no_grad():
            out = bank.process_batch(tokens, ["ep-0"], [k * 32])
        assert out.shape == (1, N_TOKENS, TOKEN_SIZE)


def test_output_stops_being_an_identity_once_there_is_history():
    """Retrieval must change the output once there is anything to retrieve.

    The first forward has no history and must pass through; later ones must
    not, or the module is switched on and computing nothing.
    """
    bank = build()
    torch.manual_seed(2)
    first = torch.randn(1, N_TOKENS, TOKEN_SIZE)
    with torch.no_grad():
        out0 = bank.process_batch(first, ["ep-0"], [0])
        later = torch.randn(1, N_TOKENS, TOKEN_SIZE)
        out1 = bank.process_batch(later, ["ep-0"], [32])
    gap0 = (out0 - first).abs().max().item()
    gap1 = (out1 - later).abs().max().item()
    assert gap1 > gap0
