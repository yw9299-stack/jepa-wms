#!/usr/bin/env python3

"""Verify PI/vanilla ownership and checkpoint round trips at task dimensions."""

from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from app.plan_common.models.AdaLN_vit import VisionTransformerAdaLN
from app.vjepa_wm.planner_landscape import install_planner_scale_gradient
from scripts.generate_stablewm_task_training_configs import TASKS


def _math_sdpa():
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


def _build(task: str, *, enabled: bool, device: torch.device):
    spec = TASKS[task]
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
        action_dim=spec["action_dim"] * 5,
        proprio_dim=spec["proprio_dim"],
        proprio_encoder_inpred=False,
        action_encoder_inpred=True,
        proprio_encoding="feature",
        proprio_emb_dim=16,
        proprio_tokens=0,
        init_scale_factor_adaln=10,
        planner_identified_input_scale=enabled,
    ).to(device).eval()


def _inputs(task: str, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(20260815)
    action_width = TASKS[task]["action_dim"] * 5
    visual = torch.randn(1, 2, 1, 16, 16, 384, generator=generator, device=device)
    actions = torch.randn(1, 2, action_width, generator=generator, device=device)
    proprio = torch.randn(1, 2, 16 * 16, 16, generator=generator, device=device)
    return visual, actions, proprio


def _visual(output):
    return output[0] if isinstance(output, tuple) else output


def run(
    task: str,
    device: torch.device,
    checkpoint_output: Path | None = None,
    vanilla_checkpoint_output: Path | None = None,
) -> dict:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(20260815)
    native = _build(task, enabled=False, device=device)
    torch.manual_seed(20260815)
    pi = _build(task, enabled=True, device=device)
    native_state = native.state_dict()
    pi_state = pi.state_dict()
    common_parameter_count = 0
    for key, value in native_state.items():
        if key not in pi_state or not torch.equal(value, pi_state[key]):
            raise RuntimeError(f"PI/vanilla common initialization differs: {key}")
        common_parameter_count += value.numel()
    message = pi.load_state_dict(native.state_dict(), strict=False)
    if set(message.missing_keys) != {"planner_input_scale.log_scale"} or message.unexpected_keys:
        raise RuntimeError(
            f"PI/vanilla state mismatch: missing={message.missing_keys} "
            f"unexpected={message.unexpected_keys}"
        )
    pi_state = pi.state_dict()
    for key, value in native_state.items():
        if key not in pi_state or not torch.equal(value, pi_state[key]):
            raise RuntimeError(f"PI/vanilla common parameter differs: {key}")

    inputs = _inputs(task, device)
    with torch.no_grad(), _math_sdpa():
        native_prediction = _visual(native(*inputs))
        identity_prediction = _visual(pi(*inputs))
    identity_error = float((native_prediction - identity_prediction).abs().max())
    if identity_error != 0.0:
        raise RuntimeError(f"identity scale is not native behavior: max_abs={identity_error}")

    pi.zero_grad(set_to_none=True)
    with _math_sdpa():
        ordinary_prediction = _visual(pi(*inputs))
        ordinary_loss = ordinary_prediction.float().square().mean()
    ordinary_loss.backward()
    scale = pi.planner_input_scale
    if scale.log_scale.grad is not None:
        raise RuntimeError("ordinary dynamics loss reached log_input_scale")
    ordinary_owned = [
        name
        for name, parameter in pi.named_parameters()
        if name != "planner_input_scale.log_scale" and parameter.grad is not None
    ]
    if not ordinary_owned:
        raise RuntimeError("ordinary dynamics loss reached no predictor parameter")

    pi.zero_grad(set_to_none=True)
    with scale.landscape_gradient(), _math_sdpa():
        landscape_prediction = _visual(pi(*inputs))
        landscape_loss = landscape_prediction.float().square().mean()
    scale_gradient = torch.autograd.grad(landscape_loss, scale.log_scale)[0]
    if not torch.isfinite(scale_gradient) or float(scale_gradient.abs()) == 0.0:
        raise RuntimeError("planner objective has no finite nonzero scale gradient")
    world_model = SimpleNamespace(predictor=pi, mixed_precision=False, scaler=None)
    install_planner_scale_gradient(world_model, scale_gradient)
    planner_owned = [name for name, parameter in pi.named_parameters() if parameter.grad is not None]
    if planner_owned != ["planner_input_scale.log_scale"]:
        raise RuntimeError(f"planner objective owns unexpected parameters: {planner_owned}")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.pth.tar"
        torch.save({"predictor": pi.state_dict(), "epoch": 0}, path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        restored = _build(task, enabled=True, device=device)
        restored.load_state_dict(checkpoint["predictor"], strict=True)
        restored.eval()
        with torch.no_grad(), _math_sdpa():
            restored_prediction = _visual(restored(*inputs))
        roundtrip_error = float((landscape_prediction.detach() - restored_prediction).abs().max())
        if roundtrip_error != 0.0:
            raise RuntimeError(f"checkpoint round trip changed prediction: {roundtrip_error}")

    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if checkpoint_output is not None:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_output.with_suffix(checkpoint_output.suffix + ".tmp")
        torch.save(
            {
                "predictor": pi.state_dict(),
                "epoch": 0,
                "optimizer_steps_per_epoch": 0,
                "gradient_accumulation_steps": 8,
                "total_optimizer_steps": 0,
                "preflight_only": True,
            },
            temporary,
        )
        temporary.replace(checkpoint_output)
    if vanilla_checkpoint_output is not None:
        vanilla_checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = vanilla_checkpoint_output.with_suffix(
            vanilla_checkpoint_output.suffix + ".tmp"
        )
        torch.save(
            {
                "predictor": native.state_dict(),
                "epoch": 0,
                "optimizer_steps_per_epoch": 0,
                "gradient_accumulation_steps": 8,
                "total_optimizer_steps": 0,
                "preflight_only": True,
            },
            temporary,
        )
        temporary.replace(vanilla_checkpoint_output)
    return {
        "status": "PASS",
        "task": task,
        "device": str(device),
        "identity_max_abs": identity_error,
        "common_initialized_parameter_count": common_parameter_count,
        "ordinary_owned_parameter_count": len(ordinary_owned),
        "ordinary_scale_gradient": None,
        "planner_owned_parameters": planner_owned,
        "planner_scale_gradient": float(scale_gradient),
        "checkpoint_roundtrip_max_abs": roundtrip_error,
        "predictor_peak_memory_bytes": int(peak_bytes),
        "preflight_checkpoint": None
        if checkpoint_output is None
        else str(checkpoint_output.resolve()),
        "vanilla_preflight_checkpoint": None
        if vanilla_checkpoint_output is None
        else str(vanilla_checkpoint_output.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--vanilla-checkpoint-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the task-dimension model preflight")
    result = run(
        args.task,
        device,
        checkpoint_output=args.checkpoint_output,
        vanilla_checkpoint_output=args.vanilla_checkpoint_output,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
