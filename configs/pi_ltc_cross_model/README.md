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

The native `latest.pth.tar` and per-epoch `e*.pth.tar` files are retained.
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
