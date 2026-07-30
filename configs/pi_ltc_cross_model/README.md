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
