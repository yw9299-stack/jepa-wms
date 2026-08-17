# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Planner-identified latent transition calibration for JEPA-style WMs."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist


def pairwise_cost_differences(cost):
    if cost.ndim != 2 or cost.shape[1] < 2:
        raise ValueError("planner costs must have shape (groups, candidates>=2)")
    pair_i, pair_j = torch.triu_indices(cost.shape[1], cost.shape[1], offset=1, device=cost.device)
    return cost.index_select(1, pair_i) - cost.index_select(1, pair_j)


def normalized_pairwise_landscape_error(
    predicted_cost,
    real_cost,
    eps=1.0e-8,
    *,
    relative_energy_floor=0.0,
    return_diagnostics=False,
):
    """Return the normalized landscape error over numerically identifiable groups.

    A group's normalization energy is the mean squared realized pairwise cost
    difference.  When that energy is at or below the existing denominator
    guard, the realized candidate order is not numerically identifiable.  Such
    groups must not own a scale gradient: replacing their denominator by
    ``eps`` would turn representation noise into an arbitrarily large relative
    error.  A batch-relative floor also rejects groups whose RMS candidate
    separation is negligible beside a typical group in the same planner
    batch.  Every identifiable group retains the exact previous objective.
    """

    if predicted_cost.shape != real_cost.shape:
        raise ValueError("predicted and real planner costs must have identical shapes")
    if not torch.isfinite(torch.tensor(float(eps))) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive")
    if (
        not torch.isfinite(torch.tensor(float(relative_energy_floor)))
        or float(relative_energy_floor) < 0.0
    ):
        raise ValueError("relative_energy_floor must be finite and non-negative")
    predicted_difference = pairwise_cost_differences(predicted_cost)
    real_difference = pairwise_cost_differences(real_cost.detach())
    numerator = (predicted_difference - real_difference).square().mean(dim=1)
    target_energy = real_difference.square().mean(dim=1)
    target_energy_median = target_energy.median()
    effective_energy_floor = torch.maximum(
        target_energy.new_tensor(float(eps)),
        target_energy_median * float(relative_energy_floor),
    )
    valid_group = target_energy > effective_energy_floor
    valid_group_count = valid_group.sum()
    if bool(valid_group.any()):
        loss = (numerator[valid_group] / target_energy[valid_group]).mean()
    else:
        # Preserve a zero gradient path to predicted_cost/log_scale so callers
        # can safely use autograd.grad without special-casing an empty batch.
        loss = predicted_difference.sum() * 0.0

    if not return_diagnostics:
        return loss
    diagnostics = {
        "valid_group": valid_group,
        "valid_group_count": valid_group_count,
        "valid_group_fraction": valid_group.float().mean(),
        "target_energy": target_energy,
        "target_energy_min": target_energy.min(),
        "target_energy_median": target_energy_median,
        "target_energy_max": target_energy.max(),
        "target_energy_effective_floor": effective_energy_floor,
        "target_energy_relative_floor": target_energy.new_tensor(
            float(relative_energy_floor)
        ),
        "target_energy_min_to_median_ratio": target_energy.min()
        / target_energy_median.clamp_min(float(eps)),
    }
    return loss, diagnostics


def clip_grad_norm_with_isolated_parameters(
    parameters,
    isolated_parameters,
    max_norm,
):
    """Clip ordinary and planner-owned gradients without cross-coupling.

    Joint norm clipping lets a large planner-scale gradient shrink ordinary
    transition gradients even though the planner objective has no graph to the
    transition model.  Returning both pre-clip norms keeps that separation
    explicit and auditable.
    """

    parameters = tuple(parameters)
    isolated_parameters = tuple(isolated_parameters)
    isolated_ids = {id(parameter) for parameter in isolated_parameters}
    if len(isolated_ids) != len(isolated_parameters):
        raise ValueError("isolated_parameters contains duplicate parameters")
    if any(not any(parameter is candidate for candidate in parameters) for parameter in isolated_parameters):
        raise ValueError("every isolated parameter must belong to parameters")

    ordinary_parameters = tuple(
        parameter
        for parameter in parameters
        if id(parameter) not in isolated_ids
    )
    ordinary_grad_parameters = tuple(
        parameter for parameter in ordinary_parameters if parameter.grad is not None
    )
    isolated_grad_parameters = tuple(
        parameter for parameter in isolated_parameters if parameter.grad is not None
    )

    reference = next(
        (
            parameter
            for parameter in parameters
            if parameter.grad is not None
        ),
        parameters[0] if parameters else None,
    )
    if reference is None:
        zero = torch.tensor(0.0)
    else:
        zero = reference.new_tensor(0.0)
    ordinary_norm = (
        torch.nn.utils.clip_grad_norm_(ordinary_grad_parameters, max_norm)
        if ordinary_grad_parameters
        else zero
    )
    isolated_norm = (
        torch.nn.utils.clip_grad_norm_(isolated_grad_parameters, max_norm)
        if isolated_grad_parameters
        else zero
    )
    return ordinary_norm, isolated_norm


def pairwise_sign_accuracy(predicted_cost, real_cost):
    predicted_difference = pairwise_cost_differences(predicted_cost.detach())
    real_difference = pairwise_cost_differences(real_cost.detach())
    comparable = real_difference.abs() > 1.0e-12
    correct = predicted_difference.signbit() == real_difference.signbit()
    count = comparable.sum().clamp_min(1)
    return (correct & comparable).sum(dtype=torch.float32) / count


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def planner_input_scale(world_model):
    predictor = _unwrap(world_model.predictor)
    scale = getattr(predictor, "planner_input_scale", None)
    if scale is None:
        raise RuntimeError("planner-identification requires model.predictor." "planner_identified_input_scale=true")
    return scale


def _flatten_group_candidates(value):
    groups, candidates = value.shape[:2]
    return value.reshape(groups * candidates, *value.shape[2:])


@contextmanager
def _preserve_eval_mode(world_model):
    modules = [
        world_model.encoder,
        world_model.predictor,
        world_model.action_encoder,
        world_model.proprio_encoder,
    ]
    states = [(module, module.training) for module in modules if module is not None]
    try:
        for module, _ in states:
            module.eval()
        yield
    finally:
        for module, training in states:
            module.train(training)


@contextmanager
def _scale_only_predictor_graph(world_model):
    predictor = _unwrap(world_model.predictor)
    scale = planner_input_scale(world_model)
    states = [
        (parameter, parameter.requires_grad)
        for parameter in predictor.parameters()
        if parameter is not scale.log_scale
    ]
    try:
        for parameter, _ in states:
            parameter.requires_grad_(False)
        yield predictor, scale
    finally:
        for parameter, requires_grad in states:
            parameter.requires_grad_(requires_grad)


@dataclass
class PlannerLandscapeResult:
    loss: torch.Tensor
    scale_gradient: torch.Tensor
    stats: dict


def planner_landscape_result(
    world_model,
    batch,
    *,
    device,
    dtype,
    mixed_precision,
    target_energy_eps=1.0e-8,
    target_energy_relative_floor=0.0,
    compute_scale_gradient=True,
):
    """Compute the landscape objective and its gradient only for log(scale).

    ``torch.autograd.grad`` explicitly requests the single calibration
    parameter. Predictor/action/proprio weights are present in the forward
    graph but never receive a gradient from this objective.
    """

    context = batch["context_visual"].to(device, non_blocking=True)
    goal = batch["goal_visual"].to(device, non_blocking=True)
    next_visual = batch["next_visual"].to(device, non_blocking=True)
    action = batch["action"].to(device, non_blocking=True)
    context_proprio = batch["context_proprio"].to(device, non_blocking=True)
    groups, candidates = next_visual.shape[:2]
    history_size = context.shape[1]
    if history_size < 1:
        raise ValueError("planner context must contain at least one frame")
    if goal.shape[:2] != (groups, 1):
        raise ValueError("planner goal must contain exactly one frame per group")
    if action.shape[:3] != (groups, candidates, history_size):
        raise ValueError(
            "planner actions must align with every context frame: "
            f"context={tuple(context.shape)}, action={tuple(action.shape)}"
        )
    if context_proprio.shape[:2] != (groups, history_size):
        raise ValueError(
            "planner proprio must align with every context frame: "
            f"context={tuple(context.shape)}, proprio={tuple(context_proprio.shape)}"
        )

    all_visual = torch.cat((context, goal, next_visual), dim=1)
    with _preserve_eval_mode(world_model), _scale_only_predictor_graph(world_model) as (predictor, scale):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            encoded = world_model.encode_obs({"visual": all_visual, "proprio": None})["visual"]
            context_features = encoded[:, :history_size]
            goal_features = encoded[:, history_size : history_size + 1]
            next_features = encoded[:, history_size + 1 :]
            action_features = world_model.encode_act(_flatten_group_candidates(action))
            proprio_features = world_model.encode_proprio(context_proprio)

        repeated_context = context_features[:, None].expand(groups, candidates, *context_features.shape[1:])
        repeated_goal = goal_features[:, None].expand(groups, candidates, *goal_features.shape[1:])
        repeated_proprio = proprio_features[:, None].expand(groups, candidates, *proprio_features.shape[1:])
        repeated_context = _flatten_group_candidates(repeated_context)
        repeated_goal = _flatten_group_candidates(repeated_goal)
        repeated_proprio = _flatten_group_candidates(repeated_proprio)

        scale_context = scale.landscape_gradient() if compute_scale_gradient else nullcontext()
        gradient_context = torch.enable_grad() if compute_scale_gradient else torch.no_grad()
        with scale_context, gradient_context, torch.amp.autocast(
            "cuda",
            dtype=dtype,
            enabled=mixed_precision,
        ):
            predicted, _, _ = world_model.forward_pred(
                repeated_context,
                action_features,
                repeated_proprio,
                predictor_override=predictor,
            )
            if predicted.shape[1] != history_size:
                raise ValueError(
                    "planner predictor output does not match context/action history: "
                    f"predicted={tuple(predicted.shape)}, history_size={history_size}"
                )
            predicted_endpoint = predicted[:, -1:]
            predicted_cost = (
                (predicted_endpoint.float() - repeated_goal.float())
                .square()
                .flatten(1)
                .mean(dim=1)
                .reshape(groups, candidates)
            )
            real_cost = (next_features.float() - goal_features.float()).square().flatten(2).mean(dim=2)
            loss, landscape_diagnostics = normalized_pairwise_landscape_error(
                predicted_cost,
                real_cost,
                eps=target_energy_eps,
                relative_energy_floor=target_energy_relative_floor,
                return_diagnostics=True,
            )

        if compute_scale_gradient:
            scale_gradient = torch.autograd.grad(
                loss,
                scale.log_scale,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0].detach()
        else:
            scale_gradient = scale.log_scale.detach().new_zeros(())
    if (
        compute_scale_gradient
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        dist.all_reduce(scale_gradient, op=dist.ReduceOp.SUM)
        scale_gradient.div_(dist.get_world_size())

    predicted_std = predicted_cost.detach().std(dim=1).mean()
    real_std = real_cost.detach().std(dim=1).mean()
    response_ratio = predicted_std / real_std.clamp_min(float(target_energy_eps) ** 0.5)
    valid_group = landscape_diagnostics["valid_group"]
    if bool(valid_group.any()):
        valid_sign_accuracy = pairwise_sign_accuracy(
            predicted_cost[valid_group],
            real_cost[valid_group],
        )
    else:
        valid_sign_accuracy = predicted_cost.new_tensor(0.0)
    stats = {
        "planner_landscape_loss": loss.detach(),
        "planner_pairwise_sign_accuracy": pairwise_sign_accuracy(predicted_cost, real_cost),
        "planner_valid_pairwise_sign_accuracy": valid_sign_accuracy,
        "planner_predicted_cost_std": predicted_std,
        "planner_real_cost_std": real_std,
        "planner_predicted_to_real_cost_std_ratio": response_ratio,
        "planner_zero_difference_baseline": predicted_cost.new_tensor(1.0),
        "planner_improvement_over_zero_difference": predicted_cost.new_tensor(1.0)
        - loss.detach(),
        "planner_context_frames": predicted_cost.new_tensor(float(history_size)),
        "planner_valid_group_count": landscape_diagnostics["valid_group_count"].float(),
        "planner_valid_group_fraction": landscape_diagnostics["valid_group_fraction"],
        "planner_target_energy_min": landscape_diagnostics["target_energy_min"],
        "planner_target_energy_median": landscape_diagnostics["target_energy_median"],
        "planner_target_energy_max": landscape_diagnostics["target_energy_max"],
        "planner_target_energy_effective_floor": landscape_diagnostics[
            "target_energy_effective_floor"
        ],
        "planner_target_energy_relative_floor": landscape_diagnostics[
            "target_energy_relative_floor"
        ],
        "planner_target_energy_min_to_median_ratio": landscape_diagnostics[
            "target_energy_min_to_median_ratio"
        ],
        "planner_target_energy_eps": predicted_cost.new_tensor(
            float(target_energy_eps)
        ),
        "planner_scale": scale.input_scale.detach(),
        "planner_log_scale": scale.log_scale.detach(),
        "planner_scale_gradient": scale_gradient,
    }
    return PlannerLandscapeResult(
        loss=loss.detach(),
        scale_gradient=scale_gradient,
        stats=stats,
    )


def install_planner_scale_gradient(world_model, scale_gradient):
    """Install a manually isolated gradient in AMP-compatible units."""

    scale = planner_input_scale(world_model)
    if scale.log_scale.grad is not None:
        ordinary_gradient = scale.log_scale.grad.detach()
        if ordinary_gradient.abs().max().item() != 0.0:
            raise RuntimeError("ordinary transition loss reached planner log_scale; " "gradient ownership is violated")
    multiplier = (
        world_model.scaler.get_scale() if world_model.mixed_precision and world_model.scaler is not None else 1.0
    )
    scale.log_scale.grad = scale_gradient.to(scale.log_scale) * float(multiplier)
