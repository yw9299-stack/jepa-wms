#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from contextlib import nullcontext

import torch

from app.plan_common.models.AdaLN_vit import VisionTransformerAdaLN
from app.plan_common.models.vit import ViTPredictor


HF_REPO_ID = "facebook/jepa-wms"
SUPPORTED_MODELS = (
    "dino_wm_pusht",
    "dino_wm_pointmaze",
    "jepa_wm_pusht",
    "jepa_wm_pointmaze",
)


def _clean_state_dict(state_dict):
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _checkpoint_path(model_name, checkpoint):
    if checkpoint is not None:
        return checkpoint
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO_ID, filename=f"{model_name}.pth.tar")


def _build_dino_predictor(device):
    return ViTPredictor(
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        num_patches=16**2,
        num_frames=4,
        dim=384 + 20 + 10,
        use_sdpa=False,
        planner_identified_input_scale=True,
        planner_identified_visual_dim=384,
    ).to(device).eval()


def _build_jepa_predictor(device):
    return VisionTransformerAdaLN(
        img_size=224,
        patch_size=14,
        num_frames=4,
        tubelet_size=1,
        embed_dim=384,
        predictor_embed_dim=384,
        depth=6,
        num_heads=16,
        use_rope=True,
        local_window=(3, -1, -1),
        action_dim=10,
        proprio_dim=4,
        proprio_encoder_inpred=False,
        action_encoder_inpred=True,
        proprio_encoding="feature",
        proprio_emb_dim=16,
        proprio_tokens=0,
        init_scale_factor_adaln=10,
        planner_identified_input_scale=True,
    ).to(device).eval()


def _build_predictor(model_name, device):
    if model_name.startswith("dino_wm_"):
        return _build_dino_predictor(device)
    return _build_jepa_predictor(device)


def _synthetic_inputs(model_name, device):
    generator = torch.Generator(device=device).manual_seed(20260730)
    context_frames = 2
    if model_name.startswith("dino_wm_"):
        features = torch.randn(
            1,
            context_frames * 16 * 16,
            384 + 20 + 10,
            generator=generator,
            device=device,
        )
        return (features,)
    visual = torch.randn(1, context_frames, 1, 16, 16, 384, generator=generator, device=device)
    actions = torch.randn(1, context_frames, 10, generator=generator, device=device)
    proprio = torch.randn(1, context_frames, 16 * 16, 16, generator=generator, device=device)
    return visual, actions, proprio


def _visual_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _math_sdpa():
    if not torch.cuda.is_available():
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel(SDPBackend.MATH)


def _native_identity_preflight(model, args):
    scale = model.planner_input_scale
    saved = model.planner_input_scale
    model.planner_input_scale = None
    with torch.no_grad(), _math_sdpa():
        native = _visual_output(model(*args))
    model.planner_input_scale = saved
    with torch.no_grad(), _math_sdpa():
        identity = _visual_output(model(*args))
    error = (native - identity).abs().max().item()
    if error != 0.0:
        raise RuntimeError(f"s=1 is not bit-exact native behavior: max_abs={error:.3e}")
    if scale.input_scale.item() != 1.0:
        raise RuntimeError(f"identity initialization failed: scale={scale.input_scale.item()}")
    return error


def _ordinary_loss_gradient_preflight(model, args):
    scale = model.planner_input_scale
    model.zero_grad(set_to_none=True)
    with _math_sdpa():
        output = _visual_output(model(*args))
        loss = output.float().square().mean()
    gradient = torch.autograd.grad(loss, scale.log_scale, allow_unused=True)[0]
    if gradient is not None and gradient.abs().item() != 0.0:
        raise RuntimeError(f"ordinary transition loss reached log_scale: grad={gradient.item():+.6e}")
    return gradient


def _directional_jvp_preflight(model, args, minimum=1.0e-8):
    scale = model.planner_input_scale

    # Use stateless functional execution so the actual checkpoint weights remain
    # untouched while log_scale is the sole JVP input.
    from torch.func import functional_call, jvp

    parameters = dict(model.named_parameters())
    log_scale_name = "planner_input_scale.log_scale"
    log_scale = parameters[log_scale_name]

    def stateless_prediction(value):
        overridden = dict(parameters)
        overridden[log_scale_name] = value
        return _visual_output(functional_call(model, overridden, args))

    with scale.landscape_gradient(), _math_sdpa():
        output, derivative = jvp(
            stateless_prediction,
            (log_scale,),
            (torch.ones_like(log_scale),),
        )
    output = output.float().flatten(1)
    derivative = derivative.float().flatten(1)
    projection = (derivative * output).sum(dim=1, keepdim=True)
    projection = projection / output.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    directional = derivative - projection * output
    total_norm = derivative.norm(dim=1).mean().item()
    directional_norm = directional.norm(dim=1).mean().item()
    directional_fraction = directional_norm / max(total_norm, 1.0e-12)
    if not torch.isfinite(derivative).all():
        raise RuntimeError("non-finite log-scale JVP")
    if directional_norm <= minimum:
        raise RuntimeError(
            "input scale is directionally invisible at this insertion point: "
            f"directional_jvp_norm={directional_norm:.6e}"
        )
    return total_norm, directional_norm, directional_fraction


def run(model_name, checkpoint, device):
    checkpoint_path = _checkpoint_path(model_name, checkpoint)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "predictor" not in state:
        raise KeyError(f"checkpoint has no predictor state: {checkpoint_path}")

    model = _build_predictor(model_name, device)
    message = model.load_state_dict(_clean_state_dict(state["predictor"]), strict=False)
    expected_missing = {"planner_input_scale.log_scale"}
    if set(message.missing_keys) != expected_missing or message.unexpected_keys:
        raise RuntimeError(
            "checkpoint architecture mismatch: "
            f"missing={message.missing_keys} unexpected={message.unexpected_keys}"
        )

    args = _synthetic_inputs(model_name, device)
    identity_error = _native_identity_preflight(model, args)
    ordinary_gradient = _ordinary_loss_gradient_preflight(model, args)
    total_jvp, directional_jvp, directional_fraction = _directional_jvp_preflight(model, args)
    print(f"[cross-model PI-LTC smoke] PASS model={model_name}")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  identity_max_abs={identity_error:.3e}")
    print(f"  ordinary_loss_scale_grad={ordinary_gradient}")
    print(f"  total_logscale_jvp_norm={total_jvp:.6e}")
    print(f"  directional_logscale_jvp_norm={directional_jvp:.6e}")
    print(f"  directional_fraction={directional_fraction:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-checkpoint preflight")
    run(args.model, args.checkpoint, torch.device(args.device))


if __name__ == "__main__":
    main()
