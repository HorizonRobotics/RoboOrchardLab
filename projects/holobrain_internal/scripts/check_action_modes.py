#!/usr/bin/env python3
"""Per-step forwarding and ACT temporal ensembling: does the policy deliver
what each mode claims?

Pure python, no GPU and no Isaac Sim -- stubs built with object.__new__,
because __init__ loads a 2.4 GB checkpoint.

Every check carries a negative control. The failures these modes can hide are
silent ones: a chunk of the wrong length still runs, and a blend with the
weights reversed is still a valid action.
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

HORIZON = 32
DIM = 14


class Proc:
    def pre_process(self, data):
        return {"step_index": [0]}

    def post_process(self, out, model_input):
        return out


captured = []
_call = [0]


def fake_model(model_input):
    """A different chunk every call, so blending is visible."""
    captured.append(dict(model_input))
    _call[0] += 1
    base = float(_call[0])
    chunk = np.tile(
        np.arange(HORIZON, dtype=np.float32)[:, None], (1, DIM)
    ) + base * 100.0
    return types.SimpleNamespace(action=chunk)


def wired(mode, te_m=None):
    os.environ["HOLOBRAIN_ACTION_MODE"] = mode
    os.environ.pop("HOLOBRAIN_STEP_INDEX_MODE", None)
    if te_m is not None:
        os.environ["HOLOBRAIN_TE_M"] = str(te_m)
    m, tm, stride = cls._resolve_modes(HORIZON)
    s = object.__new__(cls)
    s._obs = None
    s._batch_obs = {}
    s._reset_count = 1
    s.pipeline = None
    s._action_mode, s._te_m, s._step_index_stride = m, tm, stride
    s._init_runtime_state()
    s.processor, s.model = Proc(), fake_model
    s.cfg = types.SimpleNamespace(valid_action_step=HORIZON)
    s.data_preprocess = lambda obs: obs
    # Fresh chunk numbering per policy: the counter is module level, and
    # a check that hard-codes the first chunk value silently depends on
    # how many checks ran before it.
    _call[0] = 0
    return s


# ================================================== 1. mode/stride resolution
print("--- [1] _resolve_modes reads the environment")
for var in ("HOLOBRAIN_ACTION_MODE", "HOLOBRAIN_STEP_INDEX_MODE",
            "HOLOBRAIN_TE_M"):
    os.environ.pop(var, None)
check("default is chunk with the full stride",
      cls._resolve_modes(32) == ("chunk", 0.01, 32), str(cls._resolve_modes(32)))

os.environ["HOLOBRAIN_ACTION_MODE"] = "perstep"
check("perstep forces stride 1 (one forward is one frame)",
      cls._resolve_modes(32)[2] == 1, str(cls._resolve_modes(32)))
os.environ["HOLOBRAIN_ACTION_MODE"] = "ensemble"
check("ensemble forces stride 1", cls._resolve_modes(32)[2] == 1)

os.environ["HOLOBRAIN_ACTION_MODE"] = "chunk"
os.environ["HOLOBRAIN_STEP_INDEX_MODE"] = "forward"
check("chunk + STEP_INDEX_MODE=forward still reproduces the old numbering",
      cls._resolve_modes(32)[2] == 1)
os.environ.pop("HOLOBRAIN_STEP_INDEX_MODE")

print("    negative control -- an unknown mode must not be silently accepted:")
os.environ["HOLOBRAIN_ACTION_MODE"] = "perstepp"
try:
    cls._resolve_modes(32)
    check("typo in the mode name raises", False, "it was accepted")
except ValueError as exc:
    check("typo in the mode name raises", True, str(exc)[:50])

# ============================================================ 2. chunk length
print("--- [2] each mode delivers the number of actions it promises")
for mode, want in (("chunk", HORIZON), ("perstep", 1), ("ensemble", 1)):
    s = wired(mode)
    s.update_obs({"env_idx": 0})
    out = s.predict_actions(s._obs, 0)
    check(f"{mode} delivers {want} action(s)", out.shape[0] == want,
          f"got {out.shape}")

s = wired("perstep")
s.update_obs({"env_idx": 0})
out = s.predict_actions(s._obs, 0)
check("perstep delivers exactly the first predicted action",
      float(out[0, 0]) == 100.0 + 0.0, str(out[0, 0]))

print("    negative control -- RoboDojo's loop only re-forwards on a short "
      "chunk, so length is the whole mechanism:")
check("chunk mode is NOT length 1 (or per-step would be indistinguishable)",
      wired("chunk").__class__ is cls and (
          lambda t: (t.update_obs({"env_idx": 0}),
                     t.predict_actions(t._obs, 0).shape[0])[1]
      )(wired("chunk")) == HORIZON)

# ======================================================= 3. the ensemble math
print("--- [3] ACT temporal ensembling blends what earlier forwards said "
      "about this frame")
M = 0.01
s = wired("ensemble", te_m=M)
s.update_obs({"env_idx": 0})
first = s.predict_actions(s._obs, 0).copy()      # chunk A = 100 + [0..31]
check("first frame has only one prediction, so it passes through",
      float(first[0, 0]) == 100.0, str(first[0, 0]))

s.update_obs({"env_idx": 0})
second = s.predict_actions(s._obs, 0).copy()     # chunk B = 200 + [0..31]
# frame 1 is covered by A (offset 1 -> 101) and B (offset 0 -> 200).
w = np.exp(-M * np.arange(2, dtype=np.float32))
w /= w.sum()
want = 101.0 * w[0] + 200.0 * w[1]
check("second frame blends A[1] and B[0] with ACT weights",
      abs(float(second[0, 0]) - want) < 1e-3,
      f"got {float(second[0, 0]):.4f} want {want:.4f}")

print("    negative control -- the weights must favour the OLDER prediction:")
flipped = 101.0 * w[1] + 200.0 * w[0]
check("blend differs from the newest-weighted version",
      abs(float(second[0, 0]) - flipped) > 1e-3,
      f"flipped would be {flipped:.4f}")
check("and it sits nearer the older prediction",
      abs(float(second[0, 0]) - 101.0) < abs(float(second[0, 0]) - 200.0))

# ==================================================== 4. per-env buffering
print("--- [4] the ensemble buffer is per env")
s = wired("ensemble", te_m=M)
for _ in range(2):
    for env in (0, 1):
        s.update_obs({"env_idx": env})
        s.predict_actions(s._obs, env)
bufs = {k: [t0 for t0, _ in v] for k, v in s._te_buf.items()}
check("each env keeps its own prediction history",
      bufs == {0: [0, 1], 1: [0, 1]}, str(bufs))
check("and its own frame counter, unscaled by env count",
      s._env_step == {0: 2, 1: 2}, str(s._env_step))

print("    negative control -- one shared buffer would hold four entries:")
check("no env's buffer holds all four forwards",
      all(len(v) == 2 for v in s._te_buf.values()),
      str({k: len(v) for k, v in s._te_buf.items()}))

# ================================================= 5. step_index under perstep
print("--- [5] step_index advances one per frame under per-step forwarding")
captured.clear()
s = wired("perstep")
for _ in range(4):
    s.update_obs({"env_idx": 0})
    s.predict_actions(s._obs, 0)
steps = [c["step_index"][0] for c in captured]
check("step_index is 0,1,2,3 -- the spacing stride-1 training used",
      steps == [0, 1, 2, 3], str(steps))

print("    negative control -- chunk mode must still produce the 32 spacing:")
captured.clear()
s = wired("chunk")
for _ in range(4):
    s.update_obs({"env_idx": 0})
    s.predict_actions(s._obs, 0)
check("chunk mode gives 0,32,64,96",
      [c["step_index"][0] for c in captured] == [0, 32, 64, 96],
      str([c["step_index"][0] for c in captured]))

# ================================================ 6. the motion observable
print("--- [6] commanded motion separates 'barely moves' from 'moves wrong'")


def still_model(model_input):
    return types.SimpleNamespace(
        action=np.zeros((HORIZON, DIM), dtype=np.float32)
    )


s = wired("chunk")
s.model = still_model
for _ in range(3):
    s.update_obs({"env_idx": 0})
    s.predict_actions(s._obs, 0)
still_path = s.memory_stats()["action_path_by_env"]["0"]
check("an arm commanded to hold still records ~0 path",
      still_path == 0.0, str(still_path))

s = wired("chunk")
for _ in range(3):
    s.update_obs({"env_idx": 0})
    s.predict_actions(s._obs, 0)
moving = s.memory_stats()
check("a moving arm records a large path",
      moving["action_path_by_env"]["0"] > 100.0,
      str(moving["action_path_by_env"]))
check("and the chunk-boundary discontinuity is reported separately",
      moving["action_jump_by_env"]["0"] > 0.0,
      str(moving["action_jump_by_env"]))
check("memory_stats names the active mode",
      moving.get("action_mode") == "chunk", str(moving.get("action_mode")))

# ========================================================== 7. reset clears
print("--- [7] reset drops the per-env ensemble state")
s = wired("ensemble", te_m=M)
s.update_obs({"env_idx": 0})
s.predict_actions(s._obs, 0)
s.reset()
check("buffers cleared", not s._te_buf and not s._act_stats and not s._last_cmd,
      f"te={len(s._te_buf)} stats={len(s._act_stats)} cmd={len(s._last_cmd)}")

for var in ("HOLOBRAIN_ACTION_MODE", "HOLOBRAIN_TE_M"):
    os.environ.pop(var, None)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
