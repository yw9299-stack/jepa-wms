# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Planner-identified latent transition calibration for JEPA-style WMs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist


def pairwise_cost_differences(cost):
    if cost.ndim != 2 or cost.shape[1] < 2:
        raise ValueError("planner costs must have shape (groups, candidates>=2)")
    pair_i, pair_j = torch.triu_indices(cost.shape[1], cost.shape[1], offset=1, device=cost.device)
    return cost.index_select(1, pair_i) - cost.index_select(1, pair_j)


def normalized_pairwise_landscape_error(predicted_cost, real_cost, eps=1.0e-8):
    if predicted_cost.shape != real_cost.shape:
        raise ValueError("predicted and real planner costs must have identical shapes")
    predicted_difference = pairwise_cost_differences(predicted_cost)
    real_difference = pairwise_cost_differences(real_cost.detach())
    numerator = (predicted_difference - real_difference).square().mean(dim=1)
    denominator = real_difference.square().mean(dim=1).clamp_min(float(eps))
    return (numerator / denominator).mean()


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


def planner_landscape_result(world_model, batch, *, device, dtype, mixed_precision):
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

    all_visual = torch.cat((context, goal, next_visual), dim=1)
    with _preserve_eval_mode(world_model), _scale_only_predictor_graph(world_model) as (predictor, scale):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            encoded = world_model.encode_obs({"visual": all_visual, "proprio": None})["visual"]
            context_features = encoded[:, 0:1]
            goal_features = encoded[:, 1:2]
            next_features = encoded[:, 2:]
            action_features = world_model.encode_act(_flatten_group_candidates(action))
            proprio_features = world_model.encode_proprio(context_proprio)

        repeated_context = context_features[:, None].expand(groups, candidates, *context_features.shape[1:])
        repeated_goal = goal_features[:, None].expand(groups, candidates, *goal_features.shape[1:])
        repeated_proprio = proprio_features[:, None].expand(groups, candidates, *proprio_features.shape[1:])
        repeated_context = _flatten_group_candidates(repeated_context)
        repeated_goal = _flatten_group_candidates(repeated_goal)
        repeated_proprio = _flatten_group_candidates(repeated_proprio)

        with scale.landscape_gradient(), torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
            predicted, _, _ = world_model.forward_pred(
                repeated_context,
                action_features,
                repeated_proprio,
                predictor_override=predictor,
            )
            predicted_cost = (
                (predicted.float() - repeated_goal.float()).square().flatten(1).mean(dim=1).reshape(groups, candidates)
            )
            real_cost = (next_features.float() - goal_features.float()).square().flatten(2).mean(dim=2)
            loss = normalized_pairwise_landscape_error(predicted_cost, real_cost)

        scale_gradient = torch.autograd.grad(
            loss,
            scale.log_scale,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(scale_gradient, op=dist.ReduceOp.SUM)
        scale_gradient.div_(dist.get_world_size())

    predicted_std = predicted_cost.detach().std(dim=1).mean()
    real_std = real_cost.detach().std(dim=1).mean()
    stats = {
        "planner_landscape_loss": loss.detach(),
        "planner_pairwise_sign_accuracy": pairwise_sign_accuracy(predicted_cost, real_cost),
        "planner_predicted_cost_std": predicted_std,
        "planner_real_cost_std": real_std,
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
