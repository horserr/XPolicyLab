#!/usr/bin/env python
"""
MemoryConflict — real-model smoke test (loads the actual Mem-0 execution module).

Loads the MemConflict adapter (Qwen3-VL-2B + MemoryBank + DiT-B) with the RMBench
`m1_mix` checkpoint and feeds a synthetic episode of random images whose hidden
states drift, so the anchor-vs-window conflict detector fires on *real* model
outputs. Then dumps the MemoryBank and the conflict log.

This is the "full" counterpart to demo_memory_conflict.py (which drives the bare
MemoryBank with synthetic vectors). No simulator is needed.

Run:
  cd XPolicyLab/policy/MemConflict
  CUDA_VISIBLE_DEVICES=0 ../../../.policy-venv/bin/python scripts/smoke_test_real_model.py
"""

import json
import os
import sys

import numpy as np

# --- path setup (mirrors setup_eval_policy_server.sh) ----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.dirname(SCRIPT_DIR)                       # .../policy/MemConflict
XPL_ROOT = os.path.dirname(os.path.dirname(POLICY_DIR))        # .../XPolicyLab
BENCH_ROOT = os.path.dirname(XPL_ROOT)                         # .../RMBench
UPSTREAM_DIR = os.path.join(POLICY_DIR, "Mem_0")

for p in (BENCH_ROOT, XPL_ROOT, UPSTREAM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml  # noqa: E402
import torch  # noqa: E402

CKPT = os.path.join(BENCH_ROOT, "checkpoints", "m1_mix", "checkpoint", "m1_mix_inference.pt")
NORM_STATS = os.path.join(BENCH_ROOT, "checkpoints", "m1_mix", "norm_stats", "norm_stats.json")

ALOHA_STATE_KEYS = {
    "left_arm_joint_state": 6,
    "left_ee_joint_state": 1,
    "right_arm_joint_state": 6,
    "right_ee_joint_state": 1,
}


def fake_obs(seed: int) -> dict:
    """An aloha_agilex/joint observation: random RGB head view + zero-ish state."""
    rng = np.random.default_rng(seed)
    color = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    state = {k: np.zeros(d, dtype=np.float32) for k, d in ALOHA_STATE_KEYS.items()}
    return {
        "vision": {"cam_head": {"color": color}},
        "state": state,
        "instruction": "Swap the two blocks in the tray.",
    }


def main() -> None:
    assert os.path.isfile(CKPT), f"checkpoint missing: {CKPT}"
    assert os.path.isfile(NORM_STATS), f"norm stats missing: {NORM_STATS}"

    with open(os.path.join(POLICY_DIR, "deploy.yml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override the null / eval-script-supplied keys with concrete values.
    cfg.update(
        {
            "policy_name": "MemConflict",
            "task_name": "swap_blocks",
            "bench_name": "RMBench",
            "ckpt_name": "m1_mix",
            "env_cfg_type": "aloha_agilex",
            "action_type": "joint",
            "action_dim": 14,
            "seed": 0,
            "execution_ckpt": CKPT,
            "state_stats_path": NORM_STATS,
            "global_task": "Swap the two blocks in the tray.",
            "task_type": "M1",
        }
    )
    # Surface the per-step anchor-vs-window similarity so we can see the detector
    # computing real values (and confirm random-noise images stay above threshold).
    cfg["execution_module"]["memory_bank"]["dump_memory"] = True

    # Import after path setup so XPolicyLab.policy.MemConflict.model resolves.
    from XPolicyLab.policy.MemConflict.model import Model  # noqa: E402

    print("building Model (loads Qwen3-VL-2B + m1_mix_inference.pt) ...")
    model = Model(cfg)
    print("Model built.\n")

    print("begin_episode ...")
    model.begin_episode(fake_obs(0))

    n_steps = 8
    print(f"running {n_steps} eval steps with drifting (random) observations ...\n")
    for t in range(n_steps):
        # Fresh random image each step -> Qwen hidden states drift -> conflict.
        obs = fake_obs(1000 + t)
        model.update_obs(obs)
        result = model.get_action()
        acts = np.asarray(result) if result is not None else None
        d = model.dump_memory(0)
        log = model.get_conflict_log()
        last_sim = log[-1]["anchor_sim"] if log else float("nan")
        print(
            f"  step {t:2d}: actions={None if acts is None else acts.shape} "
            f"anchors={len(d.get('anchors', []))} window={len(d.get('window', []))} "
            f"conflicts={len(log)} last_anchor_sim={last_sim:.3f}"
        )

    print("\nconflict log (all events):")
    for ev in model.get_conflict_log():
        print("  ", json.dumps(ev, default=str))

    dump = model.dump_memory(0)
    anchors = np.asarray(dump.get("anchors", []))
    window = np.asarray(dump.get("window", []))
    print(f"\nmemory dump: anchors={anchors.shape} window={window.shape}")
    if len(anchors):
        print(f"  anchor[0] norm={np.linalg.norm(anchors[0]):.3f} dtype={anchors.dtype}")

    model.reset()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
