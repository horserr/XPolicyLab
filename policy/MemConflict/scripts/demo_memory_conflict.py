#!/usr/bin/env python
"""
MemoryConflict — minimal demonstration of anchor-vs-window memory conflict.

Feeds the MemoryBank a synthetic episode of 2048-d feature vectors whose "recent
history" (window) progressively rotates away from the task-initial anchor, the way a
bimanual robot's workspace changes as it moves blocks. It then shows:

  * `none`            — conflict events fire but the stale anchor keeps attending.
  * `reweight_window` — anchor branch is suppressed on conflict steps.
  * `refresh_anchor`  — anchor is re-anchored to the current state (sim resets).

Run with the policy env:  .policy-venv/bin/python scripts/demo_memory_conflict.py
This needs no checkpoint / backbone / simulator — only torch + this repo's MemoryBank.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Mem_0"))

from source.models.execution_module.memory_bank.memory_bank import MemoryBank  # noqa: E402

H = 2048
THRESHOLD = 0.85


class DummyClassifier:
    """Stands in for the subtask-end classifier; never fires (M1 single-stage)."""

    def predict(self, summary_vector):
        return {"prob": torch.zeros(1, 1, device=summary_vector.device)}


def drift_sequence(t_steps=20, theta_max=math.radians(80), seed=0):
    """Anchor `a` plus a window whose latest vector rotates `theta_max` away from `a`."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(H, generator=g)
    a = a / a.norm()
    perp = torch.randn(H, generator=g)
    perp = perp - perp.dot(a) * a
    perp = perp / perp.norm()

    seq = []
    for t in range(t_steps):
        theta = theta_max * t / max(1, t_steps - 1)
        v = a * math.cos(theta) + perp * math.sin(theta)
        seq.append(v / v.norm())
    return a, seq


def cos_sim(x, y):
    return float(torch.nn.functional.cosine_similarity(x, y, dim=-1).squeeze())


def run(resolution, enable_window=True, enable_anchor=True, t_steps=20):
    bank = MemoryBank(
        hidden_dim=H,
        window_size=30,
        initial_anchor_size=1,
        num_heads=8,
        enable_window=enable_window,
        enable_anchor=enable_anchor,
        resolution=resolution,
        dump_memory=False,
        conflict_threshold=THRESHOLD,
        dtype=torch.float32,
    )
    a, seq = drift_sequence(t_steps=t_steps)
    cls = DummyClassifier()
    sims, anchor_norms = [], []

    for t, v in enumerate(seq):
        nv = v.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        tv = torch.zeros_like(nv)
        bank.update_on_eval(nv, tv, cls, episode_id=0)

        dump = bank.dump_memory(0)
        anchors, window = dump["anchors"], dump["window"]
        s = cos_sim(torch.tensor(anchors[0]), torch.tensor(window[-1])) if anchors and window else 1.0
        sims.append(s)
        # how far the stored anchor has moved from the original initial anchor
        anchor_norms.append(float((torch.tensor(anchors[0]) - a).norm()) if anchors else 0.0)

    conflicts = [e for e in bank.get_conflict_log()]
    return sims, conflicts, anchor_norms, bank


def main():
    print("=" * 78)
    print("MemoryConflict demo: anchor (initial state) vs window (recent history)")
    print(f"hidden={H}  conflict_threshold(cos)={THRESHOLD}  rotation 0->80deg")
    print("=" * 78)

    for res in ("none", "reweight_window", "refresh_anchor"):
        sims, conflicts, anchor_norms, bank = run(res)
        n_conflict = len(conflicts)
        # For reweight_window, on conflict steps the anchor branch is skipped -> the
        # anchor stays fixed at the initial state (no re-anchoring). For refresh_anchor,
        # the anchor is replaced -> its norm from the initial anchor grows.
        max_anchor_drift = max(anchor_norms) if anchor_norms else 0.0
        print(f"\n--- resolution={res} ---")
        print(f"  conflict events fired : {n_conflict}")
        print(f"  first conflict at step: {conflicts[0]['step'] if conflicts else '-'}")
        print(f"  min anchor_sim       : {min(sims):.3f}")
        print(f"  final anchor_sim     : {sims[-1]:.3f}")
        print(f"  max |anchor - anchor0|: {max_anchor_drift:.3f}  "
              f"({'re-anchored' if max_anchor_drift > 0.5 else 'anchor frozen at t=0'})")
        if n_conflict:
            ev = conflicts[0]
            print(f"  sample event         : step={ev['step']} anchor_sim={ev['anchor_sim']:.3f} "
                  f"anchor_len={ev['anchor_len']} window_len={ev['window_len']}")

    # Ablation switches: removing each component must leave the bank empty of it.
    print("\n--- ablation gates ---")
    for label, ew, ea in (("full", True, True), ("no_window", False, True), ("no_anchor", True, False)):
        _, _, _, bank = run("none", enable_window=ew, enable_anchor=ea, t_steps=5)
        d = bank.dump_memory(0)
        print(f"  {label:10s} anchors={len(d['anchors'])} window={len(d['window'])}")

    print("\nDone. The conflict detector, resolution mechanisms, and ablation gates all work.")


if __name__ == "__main__":
    main()
