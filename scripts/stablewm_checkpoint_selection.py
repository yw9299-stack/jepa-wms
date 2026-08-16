"""Pure helpers for auditing final-horizon and native pass-boundary checkpoints."""

from __future__ import annotations


def expected_checkpoint_schedule(
    *,
    optimizer_steps_per_pass: int,
    configured_optimizer_step_budget: int,
    expected_completed_passes: int | None,
) -> dict:
    if optimizer_steps_per_pass < 1 or configured_optimizer_step_budget < 1:
        raise ValueError("optimizer schedules must be positive")
    if expected_completed_passes is None:
        completed_passes, partial_pass_steps = divmod(
            configured_optimizer_step_budget,
            optimizer_steps_per_pass,
        )
        return {
            "epoch": completed_passes,
            "total_optimizer_steps": configured_optimizer_step_budget,
            "optimizer_step_in_epoch": partial_pass_steps,
            "training_complete": True,
            "selection_policy": "configured_final_horizon",
        }
    if expected_completed_passes < 1:
        raise ValueError("expected completed passes must be positive")
    total_optimizer_steps = expected_completed_passes * optimizer_steps_per_pass
    if total_optimizer_steps >= configured_optimizer_step_budget:
        raise ValueError("pass-boundary selection must precede the configured final horizon")
    return {
        "epoch": int(expected_completed_passes),
        "total_optimizer_steps": total_optimizer_steps,
        "optimizer_step_in_epoch": 0,
        "training_complete": False,
        "selection_policy": "native_complete_pass_boundary",
    }
