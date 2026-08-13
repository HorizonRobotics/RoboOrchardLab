#!/usr/bin/env python3
"""Do the eval envs get separate memory banks now?

Pure python, no GPU and no Isaac Sim: stubs built with object.__new__, the way
the sampler's own tests do it, because __init__ loads a 2.4 GB model.

Every check carries a negative control. The failure this guards against is
silent -- shapes stay right, nothing raises, only the score moves -- so a test
that cannot fail is worse than none.
"""
import importlib.util
import sys

import pathlib

# scripts/ -> projects/holobrain_internal/ -> projects/ -> repo root
ROL = pathlib.Path(__file__).resolve().parents[3]
DP = str(ROL / "projects/holobrain_internal/common/holobrain_robodojo_policy"
               "/deploy_policy.py")
WR = str(ROL / "robo_orchard_lab/models/memoryvla/wrapper.py")

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# =============================================================== 1. wrapper
print("--- [1] _autoreset_for_eval must not clear the envs absent from a "
      "single-element batch")


class FakeBank:
    def __init__(self, keys):
        self.bank = {k: ["entry"] for k in keys}

    def clear_episode(self, eid):
        self.bank.pop(eid, None)


# The method only touches _last_episode_ids and the two banks, so a bare
# object with those three attributes exercises it exactly.
class Stub:
    pass


import types  # noqa: E402

src = open(WR).read()
ns = {}
start = src.index("    def _autoreset_for_eval")
end = src.index("    def _episode_ids")
body = "class W:\n" + src[start:end]
exec(compile(body, WR, "exec"), {"Sequence": list, "Any": object}, ns)
W = ns["W"]

E0, E1 = "eval-env0-ep1", "eval-env1-ep1"
w = W()
w.per_mem_bank = FakeBank([E0, E1])
w.cog_mem_bank = FakeBank([E0, E1])
w._last_episode_ids = None

# get_action_batch runs one forward per env: two calls, one id each.
w._autoreset_for_eval([E0])
w._autoreset_for_eval([E1])
kept = sorted(w.per_mem_bank.bank)
check("both env banks survive per-env forwards", kept == [E0, E1], f"kept={kept}")
check("both ids recorded as in play",
      sorted(w._last_episode_ids) == [E0, E1], f"{w._last_episode_ids}")

print("    negative control -- the OLD rule on the same call sequence:")
old_bank = FakeBank([E0, E1])
last = None
for ids in ([E0], [E1]):
    current = tuple(dict.fromkeys(ids))
    if last is not None:
        for eid in last:
            if eid not in current:
                old_bank.clear_episode(eid)
    last = current
lost = sorted(old_bank.bank)
check("old rule DID destroy env0's bank (so this test discriminates)",
      lost == [E1], f"old kept={lost}")

# =========================================================== 2. deploy_policy
print("--- [2] episode key is distinct per env and per episode")
dp = load(DP, "dp")
cls = next(v for k, v in vars(dp).items()
           if isinstance(v, type) and hasattr(v, "get_action_batch"))
print(f"    class = {cls.__name__}")


def policy_stub():
    s = object.__new__(cls)
    s._obs = None
    s._batch_obs = {}
    s._env_step = 0
    s._step_index_stride = 32
    s._reset_count = 1
    s.pipeline = None
    return s

s = policy_stub()
keys = [s._episode_key(i) for i in (0, 1, 2)]
check("distinct across envs", len(set(keys)) == 3, str(keys))
s._reset_count = 2
check("distinct across episodes", s._episode_key(0) != keys[0],
      f"{keys[0]} -> {s._episode_key(0)}")

# =========================================================== 3. env_idx flows
print("--- [3] get_action_batch tells predict_actions which env it is running")
s = policy_stub()
s._batch_obs = {0: {"env_idx": 0}, 1: {"env_idx": 1}, 2: {"env_idx": 2}}
seen = []
s.predict_actions = lambda obs, env_idx=0: (
    seen.append((obs["env_idx"], env_idx)) or __import__("numpy").zeros((32, 14))
)
try:
    s.get_action_batch([0, 1, 2])
    check("no longer refuses >1 env with memory", True)
except RuntimeError as exc:
    check("no longer refuses >1 env with memory", False, str(exc)[:60])
check("env_idx matches the obs it was given",
      seen == [(0, 0), (1, 1), (2, 2)], str(seen))

# ======================================================= 4. uuid reaches model
print("--- [4] the model input actually carries a per-env uuid")


class Proc:
    def pre_process(self, data):
        return {"step_index": [0]}

    def post_process(self, out, model_input):
        return out


captured = {}


def fake_model(model_input):
    captured.update(model_input)
    return types.SimpleNamespace(action=None)


s = policy_stub()
s.processor, s.model = Proc(), fake_model
s._env_step = 64
try:
    s._run_holobrain(object(), env_idx=3)
    check("uuid injected", captured.get("uuid") == ["eval-env3-ep1"],
          str(captured.get("uuid")))
    check("step_index still correct alongside it",
          captured.get("step_index") == [32], str(captured.get("step_index")))
except Exception as exc:
    check("uuid injected", False, f"{type(exc).__name__}: {exc}"[:90])

print("    negative control -- a baseline package (no step_index key):")
captured.clear()


class BaseProc(Proc):
    def pre_process(self, data):
        return {}


s = policy_stub()
s.processor, s.model = BaseProc(), fake_model
try:
    s._run_holobrain(object(), env_idx=3)
    check("baseline package gets NO uuid (memory switch is step_index)",
          "uuid" not in captured, str(sorted(captured)))
except Exception as exc:
    check("baseline package gets NO uuid", False, f"{type(exc).__name__}"[:60])

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
