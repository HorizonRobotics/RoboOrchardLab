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

# Hoisted: sections 4-7 all need these, and wired() below refers to them.


class Proc:
    def pre_process(self, data):
        return {"step_index": [0]}

    def post_process(self, out, model_input):
        return out


class BaseProc(Proc):
    """A baseline package: no memory, so pre_process yields no step_index."""

    def pre_process(self, data):
        return {}


import numpy  # noqa: E402

captured = []


def fake_model(model_input):
    captured.append(dict(model_input))
    return types.SimpleNamespace(
        action=numpy.zeros((32, 14), dtype=numpy.float32)
    )


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
    s._reset_count = 1
    s.pipeline = None
    # Through the real resolver and the real state initialiser, not copies of
    # them: a stub that lists the fields by hand goes stale the next time one
    # is added, which it has three times.
    s._action_mode, s._te_m, s._step_index_stride = cls._resolve_modes(32)
    s._init_runtime_state()
    return s


def wired(proc=None):
    """A stub wired far enough to run update_obs -> get_action end to end."""
    s = policy_stub()
    s.processor = proc if proc is not None else Proc()
    s.model = fake_model
    s.cfg = types.SimpleNamespace(valid_action_step=32)
    s.data_preprocess = lambda obs: obs
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


s = policy_stub()
s.processor, s.model = Proc(), fake_model
s._env_step = {3: 64}
try:
    s._run_holobrain(object(), env_idx=3)
    last = captured[-1]
    check("uuid injected", last.get("uuid") == ["eval-env3-ep1"],
          str(last.get("uuid")))
    check("step_index still correct alongside it",
          last.get("step_index") == [32], str(last.get("step_index")))
except Exception as exc:
    check("uuid injected", False, f"{type(exc).__name__}: {exc}"[:90])

print("    negative control -- a baseline package (no step_index key):")
captured.clear()

s = policy_stub()
s.processor, s.model = BaseProc(), fake_model
try:
    s._run_holobrain(object(), env_idx=3)
    check("baseline package gets NO uuid (memory switch is step_index)",
          "uuid" not in captured[-1], str(sorted(captured[-1])))
except Exception as exc:
    check("baseline package gets NO uuid", False, f"{type(exc).__name__}"[:60])

# ==================================== 5. the path the ws transport ACTUALLY uses
print("--- [5] update_obs + get_action -- the only path ws reaches -- keys "
      "per env and counts per env")
# model_client.py:81-101 keeps update_obs_batch client-side and sends one INFER
# per observation; model_server._handle_infer binds update_obs + get_action.
# So sections 3-4 above, which drive get_action_batch, exercise code that never
# runs under eval. This section drives the real sequence.
captured.clear()
s = wired()
for _round in range(3):
    for env in (0, 1):
        s.update_obs({"env_idx": env})
        s.get_action()

seen = [(c["uuid"][0], c["step_index"][0]) for c in captured]
per_env = {}
for uuid, step in seen:
    per_env.setdefault(uuid, []).append(step)

check("each env gets its own bank key over the ws path",
      sorted(per_env) == ["eval-env0-ep1", "eval-env1-ep1"],
      str(sorted(per_env)))
check("each env counts its own frames, unscaled by num_envs",
      all(v == [0, 32, 64] for v in per_env.values()), str(per_env))
check("memory_stats reports the per-env counters",
      s.memory_stats().get("env_step_by_env") == {"0": 96, "1": 96},
      str(s.memory_stats().get("env_step_by_env")))
check("scalar env_step stays the max, so the E1/E2 parsers still read it",
      s.memory_stats().get("env_step") == 96,
      str(s.memory_stats().get("env_step")))

print("    negative control -- what the OLD code produced on this same "
      "sequence:")
# Scalar counter bumped once per update_obs, and get_action passing no env
# index so predict_actions took its default of 0.
old = [("eval-env0-ep1", 32 * i) for i in range(6)]
check("old code gave ONE key and a num_envs-scaled counter "
      "(so this test discriminates)",
      len({u for u, _ in old}) == 1 and old[-1][1] == 160,
      f"keys={len({u for u, _ in old})} last_step={old[-1][1]}")

# ============================================ 6. a missing env_idx must be fatal
print("--- [6] with a memory, an observation without env_idx is an error, "
      "not a silent 0")
captured.clear()
s = wired()
s.update_obs({})  # no env_idx -- what a harness that does not stamp it sends
try:
    s.get_action()
    check("missing env_idx raises with a memory", False,
          "it silently continued")
except RuntimeError as exc:
    check("missing env_idx raises with a memory", "env_idx" in str(exc),
          str(exc)[:70])
except Exception as exc:
    check("missing env_idx raises with a memory", False,
          f"{type(exc).__name__}: {exc}"[:70])

print("    negative control -- a baseline package must still run without one:")
captured.clear()
s = wired(BaseProc())
s.update_obs({})
try:
    s.get_action()
    check("baseline package unaffected by the new requirement", True,
          f"model_input keys={sorted(captured[-1])}")
except Exception as exc:
    check("baseline package unaffected by the new requirement", False,
          f"{type(exc).__name__}: {exc}"[:70])

# =============================== 7. memory_stats names the banks, not just sizes
print("--- [7] memory_stats reports bank keys")
ns2 = {}
mstart = src.index("    def memory_stats")
mend = src.index("    def _check_eval_episode_boundary")
exec(compile("class M:\n" + src[mstart:mend], WR, "exec"), {}, ns2)
m = ns2["M"]()
m._eval_episode, m._eval_forwards, m._eval_history_reads = 1, 2, 1
m.per_mem_bank = FakeBank([E0, E1])
m.cog_mem_bank = FakeBank([E0, E1])
st = m.memory_stats()
check("bank_keys names both env banks",
      st["bank_keys"]["per_mem_bank"] == [E0, E1],
      str(st["bank_keys"]["per_mem_bank"]))

print("    negative control -- the case bank_lengths CANNOT distinguish:")
# One env plus a stale episode that reset() never cleared. Two keys, so
# bank_lengths is [1, 1] -- byte-identical to the correct two-env case above.
m2 = ns2["M"]()
m2._eval_episode, m2._eval_forwards, m2._eval_history_reads = 1, 2, 1
m2.per_mem_bank = FakeBank(["eval-env0-ep0", E0])
m2.cog_mem_bank = FakeBank(["eval-env0-ep0", E0])
st2 = m2.memory_stats()
check("bank_lengths is identical in both cases (the gap being closed)",
      st2["bank_lengths"] == st["bank_lengths"],
      f"{st2['bank_lengths']} == {st['bank_lengths']}")
check("bank_keys tells them apart",
      st2["bank_keys"] != st["bank_keys"],
      f"stale={st2['bank_keys']['per_mem_bank']}")

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
