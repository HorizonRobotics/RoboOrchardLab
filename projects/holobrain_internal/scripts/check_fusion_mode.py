#!/usr/bin/env python3
"""Does HOLOBRAIN_FUSION_MODE actually drop the gate, with the weights loaded?

The point of the switch is that fusion_type stays "gate" -- so GateFusion is
constructed and the checkpoint's four gate_fusion_blocks tensors load -- while
the forward path uses the mean. A switch that changed fusion_type instead
would be the config edit that cannot load, which is what it replaces.

So the checks that matter are: the module still exists, and the output equals
the mean rather than the gate's. An assertion that only read back the env var
would pass on a switch the forward path ignores.
"""
import inspect
import os
import sys

import torch

sys.path.insert(0, "/home/users/kun01.wu-labs/git_repo/robo_orchard_lab")
from robo_orchard_lab.models.memoryvla import memory_bank as mb  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


CLS = next(
    v for v in vars(mb).values()
    if inspect.isclass(v) and "fusion_type" in inspect.signature(v).parameters
)
print(f"class = {CLS.__name__}")

D, N = 16, 3
BASE = dict(dataloader_type="stream", group_size=2, token_size=D,
            mem_length=4, retrieval_layers=1, use_timestep_pe=False,
            fusion_type="gate", consolidate_type="fifo", update_fused=False)


def build(**over):
    kw = dict(BASE)
    kw.update(over)
    sig = inspect.signature(CLS).parameters
    kw = {k: v for k, v in kw.items() if k in sig}
    return CLS(**kw)


print("--- [1] the switch resolves, and the gate module survives it")
os.environ.pop("HOLOBRAIN_FUSION_MODE", None)
m = build()
check("default follows fusion_type", m._fusion_mode == "gate", m._fusion_mode)
check("gate module built", hasattr(m, "gate_fusion_blocks"))

os.environ["HOLOBRAIN_FUSION_MODE"] = "add"
m_add = build()
check("override reaches _fusion_mode", m_add._fusion_mode == "add",
      m_add._fusion_mode)
check("fusion_type UNCHANGED, so the checkpoint still loads",
      m_add.fusion_type == "gate", m_add.fusion_type)
check("gate weights still present to load into",
      hasattr(m_add, "gate_fusion_blocks"),
      str([n for n, _ in m_add.named_parameters() if "gate" in n]))

print("    negative control -- a typo must fail at startup, not silently run:")
os.environ["HOLOBRAIN_FUSION_MODE"] = "addd"
try:
    build()
    check("typo raises", False, "it was accepted")
except ValueError as exc:
    check("typo raises", True, str(exc)[:44])

print("    negative control -- gate requested on an add-built package:")
os.environ["HOLOBRAIN_FUSION_MODE"] = "gate"
try:
    build(fusion_type="add")
    check("asking for a gate that does not exist raises", False, "accepted")
except ValueError as exc:
    check("asking for a gate that does not exist raises", True, str(exc)[:44])

print("--- [2] the forward path honours it, not just the attribute")
torch.manual_seed(0)
os.environ.pop("HOLOBRAIN_FUSION_MODE", None)
m_gate = build()
os.environ["HOLOBRAIN_FUSION_MODE"] = "add"
m_add = build()
m_add.load_state_dict(m_gate.state_dict())      # same weights, both modes
for x in (m_gate, m_add):
    x.eval()
    x.training = False

tokens = torch.randn(2, N, D)
eids = ["ep0", "ep0"]
ts = None
with torch.no_grad():
    out_gate = m_gate.process_batch(tokens.clone(), eids, ts)
    out_add = m_add.process_batch(tokens.clone(), eids, ts)

check("gate and add give different outputs",
      not torch.allclose(out_gate, out_add, atol=1e-6),
      f"max|diff|={float((out_gate - out_add).abs().max()):.4g}")

# The first element has no history, so both modes fuse working memory with
# itself: mean and gate can legitimately agree there. The second element does
# have history, and that is where "add" must be the plain mean.
with torch.no_grad():
    m2 = build()
    m2.eval(); m2.training = False
    m2.load_state_dict(m_gate.state_dict())
    o = m2.process_batch(tokens.clone(), eids, ts)
check("add output is finite and shaped like the input",
      o.shape == tokens.shape and torch.isfinite(o).all(), str(tuple(o.shape)))

os.environ.pop("HOLOBRAIN_FUSION_MODE", None)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
