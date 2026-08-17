"""Pure helpers for auditable planner-scale canary interventions."""

from __future__ import annotations

import math
from copy import deepcopy


_REQUIRED_METRICS = (
    "planner_landscape_loss",
    "planner_valid_pairwise_sign_accuracy",
    "planner_predicted_to_real_cost_std_ratio",
)


def require_deterministic_scale_sweep_augmentation(config: dict) -> None:
    """Reject stochastic image transforms that would confound interventions."""

    stochastic_flags = (
        "auto_augment",
        "random_horizontal_flip",
        "motion_shift",
    )
    if any(bool(config.get(key, False)) for key in stochastic_flags):
        raise ValueError("canary scale sweep requires deterministic data augmentation")
    resize_aspect = tuple(
        float(value)
        for value in config.get("random_resize_aspect_ratio", (1.0, 1.0))
    )
    resize_scale = tuple(
        float(value)
        for value in config.get("random_resize_scale", (1.0, 1.0))
    )
    reprob = float(config.get("reprob", 0.0))
    if resize_aspect != (1.0, 1.0) or resize_scale != (1.0, 1.0) or reprob != 0.0:
        raise ValueError("canary scale sweep requires deterministic data augmentation")


def normalize_scale_sweep_values(values) -> tuple[float, ...]:
    """Validate an ordered, positive fixed-scale intervention grid."""

    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("canary scale sweep must be a non-empty list")
    normalized = []
    for value in values:
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"canary scale sweep values must be finite and positive, got {value!r}"
            )
        if any(
            math.isclose(scale, existing, rel_tol=0.0, abs_tol=1.0e-12)
            for existing in normalized
        ):
            raise ValueError(f"duplicate canary scale sweep value: {scale}")
        normalized.append(scale)
    if not any(
        math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        for scale in normalized
    ):
        raise ValueError("canary scale sweep must include the identity scale 1.0")
    return tuple(normalized)


def _record(mode: str, scale: float, metrics: dict) -> dict:
    copied = deepcopy(metrics)
    for key in _REQUIRED_METRICS:
        value = float(copied.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"scale sweep metric {key} must be finite")
        copied[key] = value
    copied["planner_scale"] = float(scale)
    copied["planner_log_scale"] = math.log(float(scale))
    return {
        "mode": mode,
        "scale": float(scale),
        "metrics": copied,
    }


def _brief(record: dict) -> dict:
    metrics = record["metrics"]
    return {
        "mode": record["mode"],
        "scale": record["scale"],
        "planner_landscape_loss": metrics["planner_landscape_loss"],
        "planner_valid_pairwise_sign_accuracy": metrics[
            "planner_valid_pairwise_sign_accuracy"
        ],
        "planner_predicted_to_real_cost_std_ratio": metrics[
            "planner_predicted_to_real_cost_std_ratio"
        ],
    }


def build_scale_sweep_diagnostic(
    *,
    learned_scale: float,
    learned_metrics: dict,
    fixed_metrics,
) -> dict:
    """Summarize learned and fixed interventions on one frozen final model."""

    learned_scale = float(learned_scale)
    if not math.isfinite(learned_scale) or learned_scale <= 0.0:
        raise ValueError("learned scale must be finite and positive")
    fixed_metrics = list(fixed_metrics)
    fixed_scales = normalize_scale_sweep_values([scale for scale, _ in fixed_metrics])
    records = [_record("learned", learned_scale, learned_metrics)]
    records.extend(
        _record("fixed", scale, metrics)
        for scale, (_, metrics) in zip(fixed_scales, fixed_metrics)
    )
    identity = next(
        record
        for record in records
        if record["mode"] == "fixed"
        and math.isclose(record["scale"], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    )
    learned = records[0]
    fixed_records = records[1:]
    best_fixed = min(
        fixed_records,
        key=lambda record: record["metrics"]["planner_landscape_loss"],
    )
    best_observed = min(
        records,
        key=lambda record: record["metrics"]["planner_landscape_loss"],
    )
    identity_loss = identity["metrics"]["planner_landscape_loss"]
    learned_loss = learned["metrics"]["planner_landscape_loss"]
    best_fixed_loss = best_fixed["metrics"]["planner_landscape_loss"]
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol": "same-final-checkpoint-fixed-scale-sweep-v1",
        "frozen_final_model": True,
        "zero_difference_baseline": 1.0,
        "learned": _brief(learned),
        "identity": _brief(identity),
        "best_fixed": _brief(best_fixed),
        "best_observed": _brief(best_observed),
        "learned_minus_identity_loss": learned_loss - identity_loss,
        "learned_improvement_over_identity": identity_loss - learned_loss,
        "best_fixed_minus_identity_loss": best_fixed_loss - identity_loss,
        "best_fixed_improvement_over_identity": identity_loss - best_fixed_loss,
        "observed_any_loss_below_zero_difference_baseline": (
            best_observed["metrics"]["planner_landscape_loss"] < 1.0
        ),
        "fixed_scale_range": [min(fixed_scales), max(fixed_scales)],
        "records": records,
    }
