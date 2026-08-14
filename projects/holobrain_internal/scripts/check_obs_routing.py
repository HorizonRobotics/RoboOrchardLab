#!/usr/bin/env python3
"""Do the per-env observation readings say what they claim?

Pure python, no GPU and no Isaac Sim.

These readings exist because E5 could only point. It measured commanded motion
per env and found, inside a single batch of four envs where nothing differs
but the slot, two envs thrashing at 7-9x the single-env reference (818, 692
against 94) and two below its minimum (71, 37). That is an output-side symptom
of an input-side fault; obs_jump and obs_dup make the input readable instead.

obs_dup is the one that can end the guessing: above zero is direct evidence
that two envs were handed the same observation, not an inference from what the
policy did afterwards. So its negative control matters more than the rest --
a duplicate detector that never fires would read exactly like clean routing.
"""
import importlib.util
import os
import pathlib
import sys
import types

import numpy as np

ROL = pathlib.Path(__file__).resolve().parents[3]
DP = str(ROL / "projects/holobrain_internal/common/holobrain_robodojo_policy"
               "/deploy_policy.py")

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


spec = importlib.util.spec_from_file_location("dp", DP)
dp = importlib.util.module_from_spec(spec)
sys.modules["dp"] = dp
spec.loader.exec_module(dp)
cls = next(v for k, v in vars(dp).items()
           if isinstance(v, type) and hasattr(v, "get_action_batch"))
print(f"class = {cls.__name__}")

HORIZON, DIM = 32, 14
rng = np.random.default_rng(0)


def obs(env_idx, joints, scene=0):
    """A RoboDojo-shaped observation. `scene` decides the image content."""
    j = np.asarray(joints, dtype=np.float32)
    img = np.full((48, 64, 3), scene, dtype=np.uint8)
    img[0, 0] = scene  # keep scenes distinguishable after striding
    return {
        "env_idx": env_idx,
        "state": {
            "left_arm_joint_state": j[:6],
            "left_ee_joint_state": j[6:7],
            "right_arm_joint_state": j[7:13],
            "right_ee_joint_state": j[13:14],
        },
        "vision": {"cam_head": {"color": img}},
    }


def stub():
    os.environ["HOLOBRAIN_ACTION_MODE"] = "chunk"
    s = object.__new__(cls)
    s._obs = None
    s._batch_obs = {}
    s._reset_count = 1
    s.pipeline = None
    s._action_mode, s._te_m, s._step_index_stride = cls._resolve_modes(HORIZON)
    s._init_runtime_state()
    return s


HOME = np.zeros(DIM, dtype=np.float32)

# ============================================ 1. coherent per-env streams
print("--- [1] coherent, distinct streams read as coherent and distinct")
s = stub()
for step in range(5):
    for env in (0, 1):
        # env 0 creeps, env 1 creeps twice as fast; different scenes.
        s.update_obs(obs(env, HOME + step * (env + 1) * 0.1, scene=env + 1))
st = s.memory_stats.__wrapped__(s) if hasattr(s.memory_stats, "__wrapped__") else None
jumps = {k: round(v["jump"] / max(1, v["n"] - 1), 3) for k, v in s._obs_stats.items()}
dups = {k: v["dup"] for k, v in s._obs_stats.items()}
check("obs_jump tracks each env's own step size (env1 moves 2x env0)",
      abs(jumps[0] - 14 * 0.1) < 1e-3 and abs(jumps[1] - 14 * 0.2) < 1e-3,
      str(jumps))
check("no duplicates reported when every env has its own stream",
      dups == {0: 0, 1: 0}, str(dups))

print("    negative control -- hand env1 the observation env0 just got:")
s = stub()
shared = obs(0, HOME + 0.5, scene=7)
s.update_obs(shared)
s.update_obs({**shared, "env_idx": 1})
check("obs_dup fires when two envs receive the same observation",
      s._obs_stats[1]["dup"] == 1, str({k: v["dup"] for k, v in s._obs_stats.items()}))

print("    negative control -- the home-pose false positive the image guards:")
s = stub()
# Every robot starts an episode at the same joint state; only the scene
# differs. A joints-only signature would call this a duplicate.
s.update_obs(obs(0, HOME, scene=1))
s.update_obs(obs(1, HOME, scene=2))
check("identical joint state + different scene is NOT a duplicate",
      all(v["dup"] == 0 for v in s._obs_stats.values()),
      str({k: v["dup"] for k, v in s._obs_stats.items()}))

# ================================================== 2. the two failure shapes
print("--- [2] the readings separate a frozen stream from a spliced one")
s = stub()
for _ in range(6):
    s.update_obs(obs(0, HOME + 0.5, scene=1))       # frozen: same pose forever
frozen = s._obs_stats[0]["jump"] / (s._obs_stats[0]["n"] - 1)
check("a frozen stream reads ~0", frozen == 0.0, f"{frozen:.4f}")

s = stub()
for step in range(6):
    # spliced: alternating between two robots that are far apart
    far = HOME + (0.0 if step % 2 == 0 else 3.0)
    s.update_obs(obs(0, far, scene=1 + step))
spliced = s._obs_stats[0]["jump"] / (s._obs_stats[0]["n"] - 1)
check("a spliced stream reads large", spliced > 10.0, f"{spliced:.2f}")
check("and the two are far apart, so the reading discriminates",
      spliced > 100 * max(frozen, 1e-6), f"{frozen:.4f} vs {spliced:.2f}")

# ========================================================= 3. act_gap
print("--- [3] act_gap measures how far a chunk starts from its own state")


class Proc:
    def pre_process(self, data):
        return {"step_index": [0]}

    def post_process(self, out, model_input):
        return out


def make_model(offset):
    def fake(model_input):
        return types.SimpleNamespace(
            action=np.full((HORIZON, DIM), offset, dtype=np.float32)
        )
    return fake


for offset, want_small in ((0.5, True), (9.0, False)):
    s = stub()
    s.processor = Proc()
    s.model = make_model(offset)
    s.cfg = types.SimpleNamespace(valid_action_step=HORIZON)
    s.data_preprocess = lambda o: o
    s.update_obs(obs(0, HOME + 0.5, scene=1))
    s.predict_actions(s._obs, 0)
    gap = s._act_stats[0]["gap"]
    ok = (gap < 1.0) if want_small else (gap > 100.0)
    check(f"chunk starting at {offset} from a state of 0.5 -> "
          f"{'small' if want_small else 'large'} gap", ok, f"{gap:.2f}")

# ================================================ 4. reported and cleared
print("--- [4] memory_stats reports them, reset clears them")
s = stub()
s.processor = Proc()
s.model = make_model(0.5)
s.cfg = types.SimpleNamespace(valid_action_step=HORIZON)
s.data_preprocess = lambda o: o
for env in (0, 1):
    s.update_obs(obs(env, HOME + 0.1 * (env + 1), scene=env + 1))
    s.predict_actions(s._obs, env)
stats = s.memory_stats()
for key in ("obs_jump_by_env", "obs_dup_by_env", "act_gap_by_env"):
    check(f"memory_stats carries {key}",
          set(stats.get(key, {})) == {"0", "1"}, str(stats.get(key)))
s.reset()
check("reset clears the observation-side state",
      not s._obs_stats and not s._last_js and not s._obs_sig,
      f"stats={len(s._obs_stats)} js={len(s._last_js)} sig={len(s._obs_sig)}")

# ============================================ 5. non-RoboDojo obs is tolerated
print("--- [5] an observation without RoboDojo's state is skipped, not fatal")
s = stub()
try:
    s.update_obs({"env_idx": 0})
    check("no state -> recorded nothing and did not raise",
          s._obs_stats == {}, str(s._obs_stats))
except Exception as exc:
    check("no state -> recorded nothing and did not raise", False,
          f"{type(exc).__name__}: {exc}"[:70])

os.environ.pop("HOLOBRAIN_ACTION_MODE", None)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
