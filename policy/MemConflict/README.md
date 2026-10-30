# MemConflict

**Fork of `Mem_0` that instruments, demonstrates, and resolves memory conflict.**

`Mem_0` fuses every step through a two-branch `MemoryBank`:

- **window** — a sliding FIFO of the most recent hidden states (recent history, `window_size=30`),
- **anchor** — the task-initial state (`initial_anchor_size=1`, never replaced during an episode).

The two branches are cross-attended separately and concatenated with the text token into the
`(B, 3, H)` summary that drives the DiT action head. When an episode drifts far from its start —
a bimanual robot swapping blocks, so the workspace is visibly different by the end — the
*recent history* and the *task-initial anchor* point in different directions. The anchor keeps
dragging the fused representation back toward the initial state even though that state is now
stale. That divergence is the **memory conflict** this adapter makes visible.

This fork changes **nothing about `Mem_0`'s default behavior**; every new switch defaults to the
upstream value. The instrumentation lives entirely under `policy/MemConflict/`. The vendored
upstream is kept at its original path `MemConflict/Mem_0/`, so `model.py`'s `UPSTREAM_DIR`
reference is unchanged.

## New `deploy.yml` keys (`execution_module.memory_bank`)

| key | default | meaning |
| --- | --- | --- |
| `enable_window` | `true` | gate the sliding-window branch (cross-attn **and** window write) |
| `enable_anchor` | `true` | gate the anchor branch (cross-attn **and** anchor write) |
| `resolution` | `none` | `none` \| `reweight_window` \| `refresh_anchor` |
| `dump_memory` | `false` | print `anchor_sim` / sizes / conflict flag every eval step |
| `conflict_threshold` | `0.85` | cosine(anchor, latest window) below this ⇒ a conflict event |

These are read in `memorymatters_executor.py` and forwarded into `MemoryBank.__init__`, so they
also work if you drive the executor directly.

Every one of the five keys is also overridable via an env var at eval time (mirroring the
`MEM0_*` convention), so a demo/ablation run can change conflict behavior without editing
`deploy.yml`:

| env var | overrides |
| --- | --- |
| `MEMC_ENABLE_WINDOW` | `enable_window` |
| `MEMC_ENABLE_ANCHOR` | `enable_anchor` |
| `MEMC_RESOLUTION` | `resolution` |
| `MEMC_DUMP_MEMORY` | `dump_memory` |
| `MEMC_CONFLICT_THRESHOLD` | `conflict_threshold` |

Additionally, `MEMC_ADDITIONAL_INFO="test_num=5,instruction_type=unseen"` appends extra
`key=value` args to the env-client eval invocation (e.g. to cap episodes for a demo).

## What "conflict" means here, concretely

At every eval step (`MemoryBank.update_on_eval`), the adapter computes

```
anchor_sim = cosine( mean(anchor), latest_window_vector )
```

where `latest_window_vector` is the most recent hidden state in the FIFO. When
`anchor_sim < conflict_threshold`, a `conflict` event is recorded in `MemoryBank.conflict_log`
(one dict per event: `episode_id`, `step`, `anchor_sim`, `anchor_len`, `window_len`, `resolution`).

This is a **read-only detector** — it never changes the model. The *resolution* switch is what
acts on the event:

- `none` — record but do nothing (baseline; reproduces `Mem_0` exactly).
- `reweight_window` — on a conflict step, **suppress the anchor branch** so the recent window
  drives the action. The stored anchor is left untouched (a soft, per-step reweighting).
- `refresh_anchor` — on a conflict step, **re-anchor to the current state** (replace the stored
  anchor). The stale initial state is dropped and the memory tracks the new regime (a hard memory
  update).

## Probes (read-only)

```python
model.dump_memory()          # -> {episode_id: {"anchors": [ndarray, ...], "window": [...]}}
model.dump_memory(eid)       # -> {"anchors": [...], "window": [...]} for one episode
model.get_conflict_log()     # -> list of conflict-event dicts
```

Vectors are detached, cast to float32, and moved to CPU, so they are ready for PCA/UMAP.

## Minimal demo (no checkpoint, no simulator)

```bash
cd XPolicyLab/policy/MemConflict
../../../.policy-venv/bin/python scripts/demo_memory_conflict.py
```

`scripts/demo_memory_conflict.py` feeds the bare `MemoryBank` a synthetic 2048-d episode whose
window rotates 80° away from the anchor and prints, for each `resolution`, how many conflict
events fire and how the anchor behaves. Expected outcome: `none`/`reweight_window` fire ~11 events
with the anchor frozen at `t=0`; `refresh_anchor` fires ~2 and the anchor visibly re-anchors.

## Running the real policy (M1, `swap_blocks`)

`MemConflict` is a drop-in XPolicyLab adapter. Use the upstream `Mem_0` eval chain with
`env_cfg_type=aloha_agilex` and the RMBench `m1_mix` checkpoint:

- checkpoint: `qiuly/Mem-0-m1mix-RMBench` → reassemble the 9 parts, or use the slimmed
  `m1_mix_inference.pt` (see the bundle `README.md`).
- base VLM: `Qwen/Qwen3-VL-2B-Instruct` (`model.safetensors` must exist at
  `Mem_0/checkpoints/Qwen3-VL-2B-Instruct/`; the config/processor files are enough to instantiate
  the architecture, the checkpoint overwrites the weights).
- `task_config.json` already lists `swap_blocks` as M1 (no planner / vLLM).

## Real sim results (M1, `swap_blocks`, 2 episodes × 999 steps)

Running the real policy (`MEMC_DUMP_MEMORY=true`, `test_num=2`) confirms the conflict is real and
that `refresh_anchor` resolves it:

| resolution | conflict events (E1+E2) | `anchor_sim` end E1 | end E2 | success |
| --- | --- | --- | --- | --- |
| `none` (baseline) | 61 + 39 = **100** | 0.878 | 0.853 | 0/2 |
| `refresh_anchor` | 1 + 1 = **2** | 0.987 | 0.992 | 0/2 |
| `reweight_window` | 45 + 31 = **76** | 0.853 | 0.862 | 0/2 |

- **Conflict**: the task-initial anchor is frozen, so `cosine(anchor, latest window)` decays from
  1.0 into the 0.83–0.87 band as the workspace visibly changes (`swap_blocks`), firing 100 conflict
  events (first crossing of the 0.85 threshold at step ~68 / ~273).
- **Resolution (hard)**: `refresh_anchor` re-anchors on the first conflict (~step 64 / 276);
  `anchor_sim` jumps back to ~0.99 and stays there — a **50× reduction in conflict events**, with no
  re-decay (a sawtooth reset instead of monotonic drift).
- **Resolution (soft)**: `reweight_window` suppresses the anchor's *contribution to the action* on
  conflict steps but leaves the stored anchor untouched, so its `anchor_sim` still decays like the
  baseline (45 + 31 = 76 events). Its effect lives in the action branch, not in `anchor_sim`, and
  needs success-rate differences to show (0/2 here).

Success is 0/2 on both arms: `swap_blocks` is hard within its 999-step budget and two episodes are
far too few to separate the arms on success — the quantitative signal here is the memory-level
conflict count and trajectory, not the success rate.

## Environment notes (no conda)

The upstream `install.sh` pins `torch==2.6.0+cu124` and `flash-attn==2.6.1`, which **do not run on
sm_120 (RTX 5090)**. This fork is run with:

- **policy env** (`.policy-venv`: `transformers==4.57.1`, torch 2.7.1+cu128) — the model host,
- **sim env** (`/opt/venv`: torch + sapien + warp + curobo) — the simulator client.

The Qwen wrapper falls back from `flash_attention_2` to `sdpa` when `flash_attn` is absent, so no
source build of flash-attn is required for inference.

### Policy-env dependency pins (no conda)

The upstream `install.sh` installs the *latest* `diffusers`, which pulls `huggingface-hub>=1.0`
— but `transformers==4.57.1` (required by Qwen3-VL) pins `huggingface-hub<1.0`. The two
conflict. Pin `diffusers==0.31.0` (keep `huggingface-hub<1.0`); the DiT only uses
`ConfigMixin`/`ModelMixin`, which are stable across versions. Also install `h5py`, `rich`,
`timm`, `openai`, `accelerate` — the execution module's import chain pulls them in even for M1
inference (the agent builds its planner unconditionally).

## Problems found while building this (so far)

1. `install.sh`'s torch/cu124 + flash-attn 2.6.1 pin breaks sm_120 — had to reuse cu128.
2. The `m1_mix` checkpoint ships as a 15.3 GB full-training `.pt` (incl. AdamW + scheduler);
   inference only needs the ~5.1 GB `model_state_dict`.
3. `hf download` stalled at ~35 % of `model.safetensors`; resuming with `curl -C -` from
   `hf-mirror.com` (~4 MB/s) was reliable.
4. Naming a config flag `dump_memory` collides with the `dump_memory()` probe method — the flag
   is stored as `dump_memory_enabled`.
5. `norm_stats.json` is 16-dim (model layout) but the adapter normalizes 14 arm dims; the bundle
   README states "14 arm dimensions", so this matches — but verify before assuming.
6. **`scripts` package collision.** The benchmark repo ships an empty `scripts/__init__.py`,
   which turns `RMBench/scripts/` into a *regular* package that shadows the vendored
   `Mem_0/scripts/` namespace, so `from scripts.tools_for_deploy...` can never resolve on
   RMBench. Fixed in `model.py` by putting `scripts/tools_for_deploy` on `sys.path` and importing
   the leaf modules (`image_utils`/`layout_utils`/`normlization`) directly.
7. **Random-noise images don't trigger conflict.** Feeding the real model random images yields
   `anchor_sim ≈ 0.98–1.0` (Qwen embeds "no object" similarly regardless of noise), so the
   0.85 threshold never trips. Demonstrating memory conflict requires *genuine* task-state drift
   (the simulator, where the blocks actually move), not synthetic noise. **Confirmed on the real
   simulator**: `anchor_sim` decays to 0.83–0.87 in `swap_blocks`, so the 0.85 threshold is in the
   right place for real drift.
8. **Stale cuRobo asset paths in RMBench assets → infinite expert-check loop.** The benchmark's
   `assets/embodiments/aloha-agilex/curobo_left.yml` and `curobo_right.yml` ship
   `urdf_path` / `collision_spheres` pointing at `/home/ubuntu/Desktop/RoboHorizon/...` (the
   original dev machine). On the first `play_once`, the scripted expert drives the robot, which
   spawns the cuRobo motion-planner subprocess; that subprocess crashes with `FileNotFoundError`
   on the missing path, the parent's multiprocessing `recv()` then raises
   `ConnectionResetError`/`BrokenPipeError`, and `eval_policy_xpolicylab.py`'s
   `eval_remote_policy` expert-check `except Exception` block just increments the seed and
   `continue`s — so it spins `seed=405548,405549,…` forever without ever counting an episode.
   Fix: rewrite those two lines in both files to `/root/workspace/RMBench/assets/...`. (The
   `*_tmp.yml` templates use `${ASSETS_PATH}` but `robot.py` loads the non-`_tmp`
   `curobo_left.yml`/`curobo_right.yml`, which had the hardcoded path.)

