"""Adapter from JEPA-WM checkpoints to StableWorldModel's planner cost API.

This module deliberately adapts only the learned cost model.  Environment
construction, start-state sampling, CEM sampling/update, action de-normalizing,
simulator stepping, and success measurement remain owned by LEWM's canonical
``eval_pointmaze_topdown.py`` evaluator.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from einops import rearrange
from tensordict import TensorDict

from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.stablewm_h5_dset import StableWMH5Metadata
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module


PI_SCALE_ARMS = {
    "learned": ("learned", None),
    "identity": ("fixed", 1.0),
    "fixed06": ("fixed", 0.6),
    "fixed04": ("fixed", 0.4),
}


def _single_candidate_copy(value: torch.Tensor, name: str) -> torch.Tensor:
    """Remove CEM's expanded candidate dimension without copying its payload."""

    if value.ndim < 2:
        raise ValueError(f"{name} must include batch and candidate dimensions; got {value.shape}")
    # CEM constructs this dimension with ``expand``.  A zero stride is a cheap,
    # exact audit that all candidates see the same observation.
    if value.shape[1] > 1 and value.stride(1) != 0:
        reference = value[:, :1]
        if not torch.equal(value, reference.expand_as(value)):
            raise ValueError(f"{name} differs across CEM candidates")
    return value[:, 0]


def _repeat_latent_for_candidates(latent, samples: int):
    def repeat(value: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        expanded = value.unsqueeze(1).expand(batch, samples, *value.shape[1:])
        return expanded.reshape(batch * samples, *value.shape[1:])

    if isinstance(latent, (TensorDict, dict)):
        return TensorDict({key: repeat(value) for key, value in latent.items()})
    return repeat(latent)


class StableWMPointMazeCost(torch.nn.Module):
    """Expose a JEPA-WM predictor through ``get_cost(info, candidates)``."""

    def __init__(
        self,
        model,
        *,
        source_metadata: StableWMH5Metadata,
        arm: str,
        owner: str,
        candidate_chunk_size: int,
    ):
        super().__init__()
        if int(candidate_chunk_size) <= 0:
            raise ValueError("candidate_chunk_size must be positive")
        self.model = model
        self.source_metadata = source_metadata
        self.arm = str(arm)
        self.owner = str(owner)
        self.context_cost_weight = 1.0
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.planner_scale_audit = getattr(model, "planner_scale_audit", None)
        self._encoding_cache_key = None
        self._encoding_cache_value = None
        self._encoding_cache_inputs = None
        self._encoding_cache_hits = 0
        self._encoding_cache_misses = 0
        self._cost_calls = 0
        self._candidate_chunks = 0
        self._max_flattened_candidate_batch = 0

    @property
    def action_mean(self):
        return self.source_metadata.action_mean

    @property
    def action_std(self):
        return self.source_metadata.action_std

    @property
    def proprio_mean(self):
        return self.source_metadata.proprio_mean

    @property
    def proprio_std(self):
        return self.source_metadata.proprio_std

    @property
    def encoding_cache_audit(self) -> dict:
        return {
            "strategy": "same-expanded-observation tensor identity within one CEM population",
            "hits": self._encoding_cache_hits,
            "misses": self._encoding_cache_misses,
        }

    @property
    def candidate_chunk_audit(self) -> dict:
        return {
            "strategy": "candidate/order-preserving contiguous slices; concatenate costs in original order",
            "candidate_chunk_size": self.candidate_chunk_size,
            "cost_calls": self._cost_calls,
            "candidate_chunks": self._candidate_chunks,
            "max_flattened_candidate_batch": self._max_flattened_candidate_batch,
        }

    @staticmethod
    def _tensor_identity(value: torch.Tensor) -> tuple:
        return (
            value.data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.device),
            str(value.dtype),
        )

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        if "pixels" not in info_dict or "goal" not in info_dict:
            raise KeyError("PointMaze planning requires both pixels and goal")
        if action_candidates.ndim != 4:
            raise ValueError(
                "action candidates must have shape [batch, samples, horizon, action_block_dim]; "
                f"got {action_candidates.shape}"
            )

        batch, samples, _, action_dim = action_candidates.shape
        if action_dim != self.model.action_dim:
            raise ValueError(f"candidate action dim={action_dim}, JEPA-WM expects {self.model.action_dim}")

        pixels_expanded = info_dict["pixels"]
        goal_expanded = info_dict["goal"]
        pixels = _single_candidate_copy(pixels_expanded, "pixels")
        goal_pixels = _single_candidate_copy(goal_expanded, "goal")
        proprio_key = "proprio" if "proprio" in info_dict else "state"
        if proprio_key not in info_dict:
            raise KeyError("PointMaze JEPA-WM requires proprio or state in evaluator info")
        proprio_expanded = info_dict[proprio_key]
        proprio = _single_candidate_copy(proprio_expanded, proprio_key)

        if pixels.ndim != 5 or goal_pixels.ndim != 5 or proprio.ndim != 3:
            raise ValueError(
                "unexpected evaluator tensors: "
                f"pixels={pixels.shape}, goal={goal_pixels.shape}, proprio={proprio.shape}"
            )
        if pixels.shape[0] != batch or goal_pixels.shape[0] != batch or proprio.shape[0] != batch:
            raise ValueError("observation batch does not match action candidate batch")

        device = self.model.device
        action_candidates = action_candidates.to(device=device, dtype=torch.float32, non_blocking=True)
        cache_key = (
            self._tensor_identity(pixels_expanded),
            self._tensor_identity(goal_expanded),
            self._tensor_identity(proprio_expanded),
        )
        if cache_key == self._encoding_cache_key:
            context_latent, goal_latent = self._encoding_cache_value
            self._encoding_cache_hits += 1
        else:
            pixels = pixels.to(device=device, non_blocking=True)
            goal_pixels = goal_pixels.to(device=device, non_blocking=True)
            proprio = proprio.to(device=device, dtype=torch.float32, non_blocking=True)
            context_latent = self.model.encode(TensorDict({"visual": pixels, "proprio": proprio}, device=device))
            # The goal contributes only visual latent geometry, matching LEWM's
            # canonical latent planner cost.  Proprio remains predictor context.
            goal_latent = self.model.encode(goal_pixels)
            self._encoding_cache_key = cache_key
            self._encoding_cache_value = (context_latent, goal_latent)
            # Retain the source tensors as well as their identity tuple.  This
            # prevents allocator pointer reuse from creating a false cache hit
            # at the next MPC replan.
            self._encoding_cache_inputs = (
                pixels_expanded,
                goal_expanded,
                proprio_expanded,
            )
            self._encoding_cache_misses += 1
        goal_final_base = goal_latent[:, -1]
        cost_chunks = []
        self._cost_calls += 1
        for sample_start in range(0, samples, self.candidate_chunk_size):
            sample_stop = min(samples, sample_start + self.candidate_chunk_size)
            chunk_samples = sample_stop - sample_start
            flattened_batch = batch * chunk_samples
            self._candidate_chunks += 1
            self._max_flattened_candidate_batch = max(
                self._max_flattened_candidate_batch,
                flattened_batch,
            )

            context_chunk = _repeat_latent_for_candidates(context_latent, chunk_samples)
            action_chunk = action_candidates[:, sample_start:sample_stop]
            actions = rearrange(action_chunk, "b s t a -> t (b s) a")
            predicted = self.model.unroll(context_chunk, actions)
            predicted_visual = predicted["visual"] if isinstance(predicted, (TensorDict, dict)) else predicted
            predicted_final = predicted_visual[-1]

            goal_chunk = goal_final_base.unsqueeze(1).expand(
                batch,
                chunk_samples,
                *goal_final_base.shape[1:],
            )
            goal_chunk = goal_chunk.reshape(flattened_batch, *goal_final_base.shape[1:])
            if predicted_final.shape != goal_chunk.shape:
                raise ValueError(
                    "predicted/goal latent shape mismatch: " f"{predicted_final.shape} vs {goal_chunk.shape}"
                )

            chunk_cost = (predicted_final.float() - goal_chunk.float()).square().flatten(1).sum(dim=1)
            cost_chunks.append(chunk_cost.reshape(batch, chunk_samples))
            # Release the complete six-step rollout before constructing the
            # next chunk so the CUDA allocator can reuse those blocks.
            del (
                context_chunk,
                action_chunk,
                actions,
                predicted,
                predicted_visual,
                predicted_final,
                goal_chunk,
                chunk_cost,
            )

        costs = torch.cat(cost_chunks, dim=1)
        if not torch.isfinite(costs).all():
            raise FloatingPointError("JEPA-WM planner produced non-finite candidate costs")
        return costs


def build_stablewm_pointmaze_cost(
    *,
    checkpoint_dir: str | Path,
    checkpoint: str,
    training_config: dict,
    source_h5: str | Path,
    arm: str,
    owner: str,
    candidate_chunk_size: int = 32,
    device: str | torch.device = "cuda",
) -> StableWMPointMazeCost:
    """Load either the PI or matched-vanilla checkpoint for Medium planning."""

    owner = str(owner).strip().lower()
    if owner not in {"pi", "vanilla"}:
        raise ValueError(f"owner must be pi or vanilla, got {owner!r}")
    if owner == "pi" and arm not in PI_SCALE_ARMS:
        raise ValueError(f"unknown PI scale arm {arm!r}")
    if owner == "vanilla" and arm != "vanilla_5pass":
        raise ValueError("vanilla owner requires arm=vanilla_5pass")

    cfg_data = deepcopy(training_config["data"])
    cfg_data_aug = deepcopy(training_config["data_aug"])
    cfg_model = deepcopy(training_config["model"])
    cfg_data["paths"] = [str(Path(source_h5).resolve())]
    if owner == "vanilla":
        cfg_model["predictor"]["planner_identified_input_scale"] = False

    metadata = StableWMH5Metadata(
        source_h5,
        normalize_action=bool(cfg_data["custom"].get("normalize_action", True)),
    )
    transform = make_transforms(img_size=int(cfg_data["img_size"]), **cfg_data_aug)
    inverse_transform = make_inverse_transforms(img_size=int(cfg_data["img_size"]), **cfg_data_aug)
    preprocessor = Preprocessor(
        action_mean=metadata.action_mean,
        action_std=metadata.action_std,
        state_mean=metadata.state_mean,
        state_std=metadata.state_std,
        proprio_mean=metadata.proprio_mean,
        proprio_std=metadata.proprio_std,
        transform=transform,
        inverse_transform=inverse_transform,
    )

    wrapper_kwargs = {"ctxt_window": 3, "proprio_mode": "predict_proprio"}
    if owner == "pi":
        mode, value = PI_SCALE_ARMS[arm]
        wrapper_kwargs.update(
            {
                "planner_identified_scale_intervention_mode": mode,
                "planner_identified_scale_intervention_value": value,
            }
        )

    wrapped = init_module(
        folder=str(Path(checkpoint_dir).resolve()),
        checkpoint=checkpoint,
        model_kwargs=cfg_model,
        device=torch.device(device),
        action_dim=metadata.action_dim,
        proprio_dim=metadata.proprio_dim,
        preprocessor=preprocessor,
        cfgs_data=cfg_data,
        wrapper_kwargs=wrapper_kwargs,
    )
    wrapped.eval().requires_grad_(False)
    return StableWMPointMazeCost(
        wrapped,
        source_metadata=metadata,
        arm=arm,
        owner=owner,
        candidate_chunk_size=candidate_chunk_size,
    ).eval()
