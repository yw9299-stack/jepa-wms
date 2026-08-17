# PointMaze PI-LTC cross-model training

These configs reuse the existing PointMaze-Medium expert HDF5 and planner
counterfactual sidecar. No new environment data are generated.

Run the two models sequentially on one GPU:

```bash
bash scripts/pointmaze_pi_ltc_cross_model_autodl.sh jepa_wm
bash scripts/pointmaze_pi_ltc_cross_model_autodl.sh dino_wm
```

Each command performs five physical dataset passes:

- 18,224 microbatches per pass;
- gradient accumulation of 8 microbatches;
- effective batch size 128;
- 2,278 optimizer updates per pass;
- 11,390 optimizer updates in total.

Before allocating the long run, the launcher validates the exact source/sidecar
schema, runs the PI-LTC unit tests, and checks the scale insertion against the
corresponding public JEPA-WM checkpoint. The first launch may therefore spend a
few minutes downloading cached DINOv2/public-checkpoint weights.

The ordinary transition MSE updates the transition model while the input scale
is detached. The planner-landscape objective updates only
`log_input_scale`. The frozen DINOv2 encoder receives no gradient from either
objective.

## v5/v6 history-aligned adaptation

The v5 PushT/Cube protocol fixes an implementation mismatch in the earlier v4
transfer runs.  The transition model is trained with three context frames, so
each planner group now supplies the same three visual frames, the corresponding
three action blocks, and three proprio states.  The first two action blocks are
the canonical source-episode prefix and the last block is the counterfactual
branch.  The planner cost is computed from the final predicted endpoint only.
The sidecar's existing `history_size=3` attribute is checked against
`data.custom.num_hist`; no sidecar or source-data regeneration is required.

This does not change the PI-LTC architecture, positive parameterization, loss
equation, or gradient ownership.  It changes only the data presented to that
loss and its optimization protocol:

- `log_input_scale` has a dedicated AdamW parameter group at 0.1x the
  transition-model learning rate and zero weight decay;
- the planner-only group batch is four (24 candidate pairs/update), preventing
  three-frame AdaLN attention from exceeding a 24 GB GPU; the ordinary JEPA
  microbatch 16 and accumulation 8 are unchanged;
- the physical-episode validation split is evaluated at initialization and at
  every completed data pass;
- `jepa-planner-best.pth.tar` is selected only at a complete pass boundary and
  only when held-out normalized landscape loss is below the zero-response
  baseline of 1 and pairwise sign accuracy is above 0.5;
- `planner_validation_history.json` records all boundary metrics and the chosen
  pass; matched evaluation uses the vanilla `jepa-eN.pth.tar` from that same
  pass;
- canaries record held-out before/after metrics and reject excessive log-scale
  drift instead of clamping the model.

The v4 checkpoints remain immutable historical artifacts and must not be
resumed into v5/v6: the optimizer state has a different parameter-group layout.
Start v5/v6 from the same canonical public initialization.  A recommended
first launch is a 2,000-update PI canary:

```bash
NO_FILE_SCAN=1 PREFLIGHT_ONLY=0 CANARY_OPTIMIZER_STEPS=2000 \
  bash scripts/stablewm_task_5pass_autodl.sh pusht pi
```

Only after `canary_summary.json` reports `canary_gate_passed: true` should the
PI arm be launched, followed by its matched vanilla arm.  The launcher now
stops natively at a complete physical-pass boundary: PushT defaults to one pass
and Cube to four.  No relay process or termination signal is required.  To
extend both PushT arms from their saved pass-1 checkpoints through pass 2, run
the same commands with `TARGET_COMPLETE_PASSES=2`; strict resume retains the
configured long-horizon scheduler and all optimizer state.

Every canary, including a rejected one, now preserves
`jepa-canary-stepN.pth.tar` with `checkpoint_role=canary_diagnostic`.  It also
evaluates the frozen final model with learned scale and fixed scales
`[0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]` over the exact same held-out groups.
`canary_scale_sweep.json` records the learned, identity, and best observed
interventions and verifies that the learned checkpoint parameter was unchanged.
This diagnostic never changes the pass/fail gate: a rejected canary remains
ineligible for formal training and checkpoint selection.

If the best fixed intervention lands at the canary grid boundary, do not rerun
the 2,000 optimizer updates.  The rejected canary checkpoint can be loaded in
strict, checkpoint-only mode and evaluated on the same deterministic held-out
split with the extended grid
`[1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 3, 3.5, 4, 5, 6, 8]`:

```bash
PI_LTC_EXPECTED_COMMIT=<full detached evaluator commit> \
  bash scripts/pusht_canary_extended_scale_sweep_autodl.sh
```

The launcher validates the saved checkpoint SHA-256, requires
`checkpoint_role=canary_diagnostic`, `total_optimizer_steps=2000`, and
`training_complete=false`, and requires an exact model-state load.  It hashes
the checkpoint but only locates and byte-size-checks the two large HDF5 files.
W&B is disabled, zero training microbatches and optimizer steps are consumed,
and the immutable result is written to
`extended_scale_sweep_<commit>/planner_scale_sweep_only.json`.  This artifact
is diagnostic only and cannot make a rejected canary eligible for formal
training.  DINOv2 initialization first reuses the existing Torch Hub checkout
(`facebookresearch_dinov2_main`) or an explicit `JEPAWM_DINOV2_REPO`; only a
machine without either local checkout falls back to the pinned GitHub `main`
revision.

The first v5 PushT canary was rejected by that gate and remains an immutable
diagnostic artifact.  The diagnostic-checkpoint/sweep revision therefore uses
fresh `v6` run directories so config manifests and rejected v5 outputs are
never overwritten.  v6 does not change the v5 architecture, objective, data,
optimizer, or gate.

```bash
NO_FILE_SCAN=1 PREFLIGHT_ONLY=0 TARGET_COMPLETE_PASSES=1 \
  bash scripts/stablewm_task_5pass_autodl.sh pusht pi
NO_FILE_SCAN=1 PREFLIGHT_ONLY=0 TARGET_COMPLETE_PASSES=1 \
  bash scripts/stablewm_task_5pass_autodl.sh pusht vanilla
```

With `NO_FILE_SCAN=1`, the launcher never hashes or samples either large HDF5.
It writes a small audit explicitly marked `not_scanned_user_confirmed` from
path, byte size, mtime, and config hashes; evaluation recovers and checks that
same provenance marker instead of silently presenting it as a content hash.

The native `latest.pth.tar` and per-pass `e*.pth.tar` files are retained.
Running the same command again resumes from `latest.pth.tar`; there is no
additional step-aligned checkpoint callback.

Do not launch both configurations concurrently on one RTX 5090. Train
`jepa_wm` first, inspect the five-pass checkpoint and planner-scale metrics,
then run `dino_wm`.

## PushT and Cube StableWorldModel transfer

PushT and Cube use their already verified StableWorldModel HDF5 files directly;
no converted copy is created.  The adapter accepts both the historical per-row
PointMaze metadata and the episode-level `ep_len`/`ep_offset` layout used by
PushT and Cube.  Task configs are generated immutably from the PointMaze
five-pass template so the matched vanilla arm can differ only in output folder,
the PI input-scale module, and the planner-landscape objective.

The launchers hard-stop on source/sidecar SHA-256, dimensions, physical episode
boundaries, exact clip counts, PI/vanilla config restoration, gradient ownership,
checkpoint round trip, and one-episode LEWM evaluator smoke tests.  Training is
disabled by default:

```bash
export PI_LTC_EXPECTED_COMMIT=<full detached JEPA commit>
export PI_LTC_EXPECTED_LEWM_COMMIT=2f4b66934b844a2cacc9454106b9ab163e20c4be
export LEWM_REPO=/root/autodl-tmp/lewm_figure6_crossmodel_clean
export PI_LTC_PUSHT_SOURCE_SHA256=<sha256>
export PI_LTC_PUSHT_SIDECAR_SHA256=<sha256>
bash scripts/stablewm_task_5pass_autodl.sh pusht pi
```

After reviewing every preflight audit, set `PREFLIGHT_ONLY=0` and run PI and
vanilla sequentially.  Final evaluation uses three paired arms per seed:
learned and identity from the same PI checkpoint, plus the matched vanilla
checkpoint.  It pins `H/R/action_block=5/5/5`, budget 50, goal offset 25,
noise 0, and CEM 16/256/top-k 64.  The evaluator records start/state and CEM
generator-state hashes before the summary script computes pooled and
hierarchical bootstrap intervals and exact McNemar tests for the two separate
estimands.

When the data locations and byte sizes have already been externally confirmed,
`NO_FILE_SCAN=1 PREFLIGHT_ONLY=0` provides an explicit user-directed fast-start
mode. It performs only path and byte-size checks, records
`content_hash_mode=not_scanned_user_confirmed`, skips standalone hashes/data
preflight/model smoke/LEWM smoke, and starts the requested arm. Dataset schema,
episode boundaries, dimensions, and clip counts are still checked naturally
during training initialization; action/proprio statistics must still be read
because they are required for normalization.

The native StableWorldModel recorder stores the final observation of every
episode with a full-NaN action row because that state has no outgoing action.
The adapter follows the original LeWM convention exactly: normalization
statistics exclude those terminal rows, and a terminal placeholder encountered
in the final valid clip becomes zero in normalized action space.  Full-NaN rows
at non-terminal positions, partial-NaN rows, and infinities remain hard errors;
the compatibility path therefore does not conceal damaged interior data.  The
terminal clips remain part of the canonical clip count and do not change the
PushT/Cube pass definitions.

Cube runs with this terminal-placeholder adapter are versioned as `v7`; the
rejected pre-adapter `v6` directory is retained as immutable diagnostic history.
PushT remains on `v6` because its accepted runs and action data are unchanged.
